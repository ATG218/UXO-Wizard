import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Callable
from scipy.interpolate import griddata
from scipy.spatial.distance import pdist
import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from loguru import logger

from ...base import ScriptInterface, ProcessingResult, ProcessingError

# Try to import Numba for JIT compilation
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    # Define a no-op decorator if Numba is not available
    def jit(*_args, **_kwargs):
        def decorator(func):
            return func
        return decorator
    NUMBA_AVAILABLE = False

# Additional imports for boundary masking
try:
    from scipy.spatial import ConvexHull
    from scipy.spatial.distance import cdist
    from matplotlib.path import Path as MplPath
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


@jit(nopython=True, cache=True)
def _create_influence_weights_numba(x, y, z, grid_x_coords, grid_y_coords, influence_radius):
    """
    JIT-compiled function to create influence weights and target values for soft constraints.
    
    Args:
        x, y, z: Data point coordinates and values (1D arrays)
        grid_x_coords, grid_y_coords: Grid coordinate arrays (1D arrays)
        influence_radius: Influence radius for soft constraints
    
    Returns:
        influence_weights: 2D array of influence weights
        target_values: 2D array of target values
    """
    ny = len(grid_y_coords)
    nx = len(grid_x_coords)
    
    influence_weights = np.zeros((ny, nx))
    target_values = np.zeros((ny, nx))
    
    # Grid spacing
    dx = grid_x_coords[1] - grid_x_coords[0] if nx > 1 else 1.0
    dy = grid_y_coords[1] - grid_y_coords[0] if ny > 1 else 1.0
    
    # For each data point, create influence zone
    for data_idx in range(len(x)):
        xi, yi, zi = x[data_idx], y[data_idx], z[data_idx]
        
        # Convert to grid coordinates
        gi = (yi - grid_y_coords[0]) / dy
        gj = (xi - grid_x_coords[0]) / dx
        
        # Calculate grid range that could be influenced
        i_min = max(0, int(gi - influence_radius/dy))
        i_max = min(ny, int(gi + influence_radius/dy) + 1)
        j_min = max(0, int(gj - influence_radius/dx))
        j_max = min(nx, int(gj + influence_radius/dx) + 1)
        
        # Create influence zone around this data point
        for i in range(i_min, i_max):
            for j in range(j_min, j_max):
                # Calculate distance from grid point to data point
                grid_y_coord = grid_y_coords[i]
                grid_x_coord = grid_x_coords[j]
                distance = np.sqrt((grid_x_coord - xi)**2 + (grid_y_coord - yi)**2)
                
                if distance <= influence_radius:
                    # Gaussian-like weight function
                    weight = np.exp(-(distance / (influence_radius * 0.3))**2)
                    
                    # Accumulate weighted influence
                    old_weight = influence_weights[i, j]
                    new_weight = old_weight + weight
                    
                    if new_weight > 0:
                        # Weighted average of target values
                        target_values[i, j] = (target_values[i, j] * old_weight + zi * weight) / new_weight
                        influence_weights[i, j] = new_weight
    
    return influence_weights, target_values


@jit(nopython=True, cache=True)
def _minimum_curvature_iteration_numba(grid_z, influence_weights, target_values, 
                                       omega, min_allowed, max_allowed):
    """
    JIT-compiled function for a single minimum curvature iteration with soft constraints.
    
    Args:
        grid_z: Current grid values (modified in-place)
        influence_weights: Influence weights for soft constraints
        target_values: Target values for soft constraints
        omega: Relaxation factor
        min_allowed, max_allowed: Value bounds for stability
    
    Returns:
        max_change: Maximum change in this iteration
    """
    ny, nx = grid_z.shape
    max_change = 0.0
    
    # Update interior points using 5-point stencil with soft constraints
    for i in range(1, ny-1):
        for j in range(1, nx-1):
            old_value = grid_z[i, j]
            
            # 5-point stencil for Laplacian
            neighbors = (grid_z[i+1, j] + grid_z[i-1, j] + 
                        grid_z[i, j+1] + grid_z[i, j-1]) / 4.0
            
            # Successive over-relaxation
            new_value = old_value + omega * (neighbors - old_value)
            
            # Apply soft data constraint if there's influence at this point
            if influence_weights[i, j] > 0.01:  # Only apply if significant influence
                constraint_strength = influence_weights[i, j] * 0.1  # Reduce constraint strength
                target = target_values[i, j]
                # Blend the relaxed value with the target value
                new_value = new_value * (1 - constraint_strength) + target * constraint_strength
            
            # Clamp values for stability
            new_value = max(min_allowed, min(max_allowed, new_value))
            
            grid_z[i, j] = new_value
            
            # Track maximum change
            change = abs(new_value - old_value)
            if change > max_change:
                max_change = change
    
    return max_change


@jit(nopython=True, cache=True)
def _apply_boundary_conditions_numba(grid_z):
    """
    JIT-compiled function to apply boundary conditions (zero second derivative).
    
    Args:
        grid_z: Grid values (modified in-place)
    """
    ny, nx = grid_z.shape
    
    # Apply boundary conditions
    if ny >= 3:
        # Top and bottom boundaries (zero second derivative)
        for j in range(nx):
            grid_z[0, j] = 2*grid_z[1, j] - grid_z[2, j]
            grid_z[ny-1, j] = 2*grid_z[ny-2, j] - grid_z[ny-3, j]
    
    if nx >= 3:
        # Left and right boundaries (zero second derivative)
        for i in range(ny):
            grid_z[i, 0] = 2*grid_z[i, 1] - grid_z[i, 2]
            grid_z[i, nx-1] = 2*grid_z[i, nx-2] - grid_z[i, nx-3]


class GammaInterpolator(ScriptInterface):
    @property
    def name(self) -> str:
        return "Gamma Interpolator"

    @property
    def description(self) -> str:
        return "Interpolates gamma measurements to smooth grid visualizations using soft-constraint minimum curvature (eliminates flight point artifacts)"

    def get_parameters(self) -> Dict[str, Any]:
        return {
            'interpolation': {
                'method': {
                    'value': 'minimum_curvature',
                    'type': 'choice',
                    'choices': ['minimum_curvature', 'linear', 'cubic', 'nearest'],
                    'description': 'Interpolation method (minimum_curvature uses soft constraints to eliminate flight point artifacts)'
                },
                'grid_resolution': {
                    'value': 150,
                    'type': 'int',
                    'min': 20,
                    'max': 300,
                    'description': 'Grid resolution (number of points per axis)'
                },
                'max_iterations': {
                    'value': 1000,
                    'type': 'int',
                    'description': 'Maximum iterations for minimum curvature convergence'
                },
                'tolerance': {
                    'value': 1e-6,
                    'type': 'float',
                    'description': 'Convergence tolerance for minimum curvature iteration'
                },
                'relaxation_factor': {
                    'value': 1.8,
                    'type': 'float',
                    'description': 'Successive over-relaxation factor (1.0-2.0)'
                }
            },
            'boundary': {
                'enable_boundary_masking': {
                    'value': True,
                    'type': 'bool',
                    'description': 'Enable boundary masking to prevent extrapolation'
                },
                'boundary_method': {
                    'value': 'convex_hull',
                    'type': 'choice',
                    'choices': ['convex_hull', 'alpha_shape'],
                    'description': 'Method for determining data boundary'
                }
            },
            'measurements': {
                'countrate': {'value': True, 'type': 'bool', 'description': 'Interpolate Countrate'},
                'u238': {'value': True, 'type': 'bool', 'description': 'Interpolate U238'},
                'k40': {'value': True, 'type': 'bool', 'description': 'Interpolate K40'},
                'th232': {'value': True, 'type': 'bool', 'description': 'Interpolate Th232'},
                'cs137': {'value': True, 'type': 'bool', 'description': 'Interpolate Cs137'},
                'height': {'value': False, 'type': 'bool', 'description': 'Interpolate Height'},
                'press': {'value': False, 'type': 'bool', 'description': 'Interpolate Press'},
                'temp': {'value': False, 'type': 'bool', 'description': 'Interpolate Temp'},
                'hum': {'value': False, 'type': 'bool', 'description': 'Interpolate Hum'},
            },
            'visualization': {
                'generate_plot': {'value': True, 'type': 'bool', 'description': 'Generate summary plot'},
                'include_points': {'value': True, 'type': 'bool', 'description': 'Include point layers'},
                'use_global_color_range': {
                    'value': True, 
                    'type': 'bool', 
                    'description': 'Use same color range for all measurements (enables direct comparison of element prevalence)'
                },
            }
        }

    def validate_data(self, data: pd.DataFrame) -> bool:
        required = ['lat', 'lon']
        missing = [col for col in required if col not in data.columns]
        if missing:
            raise ProcessingError(f"Missing required columns: {missing}")
        if len(data) < 10:
            raise ProcessingError("Insufficient data points")
        return True

    def create_boundary_mask(self, x, y, grid_x, grid_y, method='convex_hull'):
        """
        Create a boundary mask to prevent interpolation outside data coverage.
        
        Args:
            x, y: Data point coordinates
            grid_x, grid_y: Grid meshgrid coordinates
            method: 'convex_hull' or 'alpha_shape'
        
        Returns:
            mask: Boolean array where True indicates points inside the data boundary
        """
        if not SCIPY_AVAILABLE:
            # Return all True mask if scipy is not available
            return np.ones(grid_x.shape, dtype=bool)
        
        try:
            # Get data points as array
            data_points = np.column_stack([x, y])
            
            # Get grid points as array
            grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
            
            if method == 'convex_hull':
                # Create convex hull
                hull = ConvexHull(data_points)
                
                # Check which grid points are inside the convex hull
                hull_path = MplPath(data_points[hull.vertices])
                mask_1d = hull_path.contains_points(grid_points)
                
            elif method == 'alpha_shape':
                # Simple distance-based approach as alpha shape approximation
                # For each grid point, check if it's within reasonable distance of data
                distances = cdist(grid_points, data_points)
                min_distances = np.min(distances, axis=1)
                
                # Use median nearest neighbor distance as threshold
                nn_distances = []
                for i in range(len(data_points)):
                    dists = cdist([data_points[i]], data_points)[0]
                    dists = dists[dists > 0]  # Remove self-distance
                    if len(dists) > 0:
                        nn_distances.append(np.min(dists))
                
                threshold = np.median(nn_distances) * 2.0 if nn_distances else 0.01
                mask_1d = min_distances <= threshold
            
            else:
                raise ValueError(f"Unknown boundary method: {method}")
            
            # Reshape to grid shape
            mask = mask_1d.reshape(grid_x.shape)
            
            return mask
            
        except Exception:
            # Fallback to no masking
            return np.ones(grid_x.shape, dtype=bool)

    def minimum_curvature_interpolation(self, x, y, z, grid_resolution=150, 
                                        max_iterations=1000, tolerance=1e-6, 
                                        omega=1.8, progress_callback=None):
        """
        Perform minimum curvature interpolation on irregular data points using JIT-compiled functions.
        
        Args:
            x, y, z: Data coordinates and values
            grid_resolution: Number of grid points per axis
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            omega: Relaxation factor for successive over-relaxation
            progress_callback: Optional callback for progress updates
        
        Returns:
            grid_x, grid_y, grid_z: Interpolated grid coordinates and values
        """
        if progress_callback:
            progress_callback(5, "Setting up interpolation grid...")
        
        # Create regular grid
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        
        # Add minimal padding to grid bounds
        x_range = x_max - x_min
        y_range = y_max - y_min
        padding = 0.01  # 1% padding
        
        x_min -= padding * x_range
        x_max += padding * x_range
        y_min -= padding * y_range
        y_max += padding * y_range
        
        grid_x = np.linspace(x_min, x_max, grid_resolution)
        grid_y = np.linspace(y_min, y_max, grid_resolution)
        grid_X, grid_Y = np.meshgrid(grid_x, grid_y)
        
        # Initialize grid with linear interpolation first
        if progress_callback:
            progress_callback(10, "Creating initial grid with linear interpolation...")
        
        # Create initial grid using linear interpolation to fill entire domain
        grid_z_initial = griddata((x, y), z, (grid_X, grid_Y), method='linear', fill_value=np.nan)
        
        if progress_callback:
            progress_callback(15, "Filling gaps with nearest neighbor interpolation...")
        
        # Fill any remaining NaN values with nearest neighbor interpolation
        nan_mask = np.isnan(grid_z_initial)
        if np.any(nan_mask):
            grid_z_nearest = griddata((x, y), z, (grid_X, grid_Y), method='nearest', fill_value=np.mean(z))
            grid_z_initial[nan_mask] = grid_z_nearest[nan_mask]
        
        grid_z = grid_z_initial.copy()
        
        if progress_callback:
            progress_callback(20, "Creating soft data constraints...")
        
        # Use soft constraints instead of hard constraints
        ny, nx = grid_z.shape
        dx = (x_max - x_min) / (nx - 1)
        dy = (y_max - y_min) / (ny - 1)
        
        # Calculate influence radius (adaptive based on data density)
        data_density = len(x) / (x_range * y_range)
        influence_radius = max(2 * dx, 2 * dy, 1.0 / np.sqrt(data_density))
        
        # Limit influence radius for performance with large grids
        max_reasonable_radius = min(x_range, y_range) * 0.1  # 10% of data range
        influence_radius = min(influence_radius, max_reasonable_radius)
        
        if progress_callback:
            progress_callback(22, f"Influence radius: {influence_radius:.4f} for {len(x)} data points on {ny}x{nx} grid")
        
        # Create influence weights and target values using JIT-compiled function
        if progress_callback:
            if NUMBA_AVAILABLE:
                progress_callback(25, f"🚀 Using JIT-compiled influence zones for {len(x)} data points...")
            else:
                progress_callback(25, f"Creating influence zones for {len(x)} data points...")
        
        # Convert to numpy arrays for JIT compilation
        x_array = np.asarray(x, dtype=np.float64)
        y_array = np.asarray(y, dtype=np.float64)
        z_array = np.asarray(z, dtype=np.float64)
        
        # Use JIT-compiled function for influence weight calculation
        influence_weights, target_values = _create_influence_weights_numba(
            x_array, y_array, z_array, grid_x, grid_y, influence_radius
        )
        
        # Normalize influence weights to [0, 1] range for stability
        max_weight = np.max(influence_weights)
        if max_weight > 0:
            influence_weights = influence_weights / max_weight
        
        if progress_callback:
            progress_callback(35, f"Influence zones created successfully (max weight: {max_weight:.3f})")
        
        # Value bounds for stability
        data_range = np.ptp(z)
        data_mean = np.mean(z)
        min_allowed = data_mean - 3 * data_range
        max_allowed = data_mean + 3 * data_range
        
        if progress_callback:
            progress_callback(40, "Starting minimum curvature iterations with soft constraints...")
        
        # Use stable omega value
        omega = 0.5  # Lower omega for stability
        
        if progress_callback:
            if NUMBA_AVAILABLE:
                progress_callback(45, f"🚀 Starting {max_iterations} JIT-compiled iterations with soft constraints...")
            else:
                progress_callback(45, f"Starting {max_iterations} iterations with soft constraints...")
        
        # Modified iterative minimum curvature with soft constraints using JIT
        for iteration in range(max_iterations):
            # Use JIT-compiled function for the iteration
            max_change = _minimum_curvature_iteration_numba(
                grid_z, influence_weights, target_values, omega, min_allowed, max_allowed
            )
            
            # Apply boundary conditions using JIT-compiled function
            _apply_boundary_conditions_numba(grid_z)
            
            # Early detection of instability
            if max_change > data_range:
                if progress_callback:
                    progress_callback(80, f"Large change detected at iteration {iteration+1}, stopping early")
                break
            
            # Progress update after each iteration
            if progress_callback:
                progress = 45 + 35 * ((iteration + 1) / max_iterations)
                convergence_info = f"max change: {max_change:.2e}"
                if max_change < tolerance * 10:  # Getting close to convergence
                    convergence_info += " (converging)"
                progress_callback(int(progress), f"Iteration {iteration+1}/{max_iterations}, {convergence_info}")
            
            # Check for convergence
            if max_change < tolerance:
                if progress_callback:
                    progress_callback(80, f"Converged after {iteration+1} iterations (change: {max_change:.2e})")
                break
        
        if progress_callback:
            if NUMBA_AVAILABLE:
                progress_callback(85, "🚀 JIT-compiled minimum curvature interpolation complete")
            else:
                progress_callback(85, "Minimum curvature interpolation complete")
        
        return grid_X, grid_Y, grid_z

    def execute(self, data: pd.DataFrame, params: Dict[str, Any], 
                progress_callback: Optional[Callable] = None, input_file_path: Optional[str] = None) -> ProcessingResult:
        
        result = ProcessingResult(success=True, processing_script=self.name)
        result.metadata['processor'] = 'gamma'

        if progress_callback:
            progress_callback(0, "Starting gamma interpolation...")

        # Log interpolation capabilities
        if NUMBA_AVAILABLE:
            logger.info("🚀 JIT-accelerated minimum curvature interpolation with soft constraints available")
        else:
            logger.info("📊 Using minimum curvature interpolation (install 'numba' for JIT acceleration)")

        if progress_callback:
            progress_callback(1, "Parsing interpolation parameters...")

        # Extract parameters
        interp_params = params.get('interpolation', {})
        boundary_params = params.get('boundary', {})
        meas_params = params.get('measurements', {})
        viz_params = params.get('visualization', {})

        method = interp_params.get('method', {}).get('value', 'minimum_curvature')
        grid_resolution = interp_params.get('grid_resolution', {}).get('value', 150)
        max_iterations = interp_params.get('max_iterations', {}).get('value', 1000)
        tolerance = interp_params.get('tolerance', {}).get('value', 1e-6)
        omega = interp_params.get('relaxation_factor', {}).get('value', 1.8)
        
        enable_masking = boundary_params.get('enable_boundary_masking', {}).get('value', True)
        boundary_method = boundary_params.get('boundary_method', {}).get('value', 'convex_hull')
        
        generate_plot = viz_params.get('generate_plot', {}).get('value', True)
        include_points = viz_params.get('include_points', {}).get('value', True)
        use_global_color_range = viz_params.get('use_global_color_range', {}).get('value', True)

        if progress_callback:
            progress_callback(2, "Identifying measurements to process...")

        # List of measurements to process
        measurements = []
        for m in ['Countrate', 'U238', 'K40', 'Th232', 'Cs137', 'Height', 'Press', 'Temp', 'Hum']:
            if m.lower() in meas_params and meas_params[m.lower()].get('value', False) and m in data.columns:
                measurements.append(m)

        if not measurements:
            raise ProcessingError("No measurements selected or available")
        
        if progress_callback:
            progress_callback(3, f"Found {len(measurements)} measurements to process: {', '.join(measurements)}")

        num_meas = len(measurements)
        progress_per_meas = 85 / num_meas if num_meas > 0 else 85  # Reserve 15% for setup and final steps

        if progress_callback:
            progress_callback(4, "Setting up color schemes and visualization...")

        figures = []
        gamma_colors = ["#004400", "#008800", "#00CC00", "#CCCC00", "#CC8800", "#CC0000"]
        gamma_cmap = mcolors.LinearSegmentedColormap.from_list('gamma', gamma_colors)

        gamma_color_ramps = {
            'Countrate': ['#000080', '#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF8000', '#FF0000'],
            'U238': ['#000044', '#000088', '#00FFFF', '#00FF00', '#FFFF00', '#FF8000', '#FF0000'],
            'K40': ['#004400', '#008800', '#00FF00', '#FFFF00', '#FF8000', '#FF0000', '#880000'],
            'Th232': ['#440000', '#880000', '#CC0000', '#FF0000', '#FF8000', '#FFFF00', '#FFFFFF'],
            'Cs137': ['#000080', '#0000FF', '#0044FF', '#0088FF', '#00CCFF', '#00FFFF', '#FFFFFF'],
            'Height': ['#FFFFFF', '#CCCCCC', '#999999', '#666666', '#333333', '#000000'],
            'Press': ['#FFCCCC', '#FF9999', '#FF6666', '#FF3333', '#FF0000', '#CC0000'],
            'Temp': ['#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF8000', '#FF0000'],
            'Hum': ['#00FFFF', '#00CCCC', '#009999', '#006666', '#003333', '#000000']
        }

        if progress_callback:
            progress_callback(5, "Validating measurement data...")

        # Pre-process measurements to find valid ones
        valid_measurements = []
        for measurement in measurements:
            valid_data = data.dropna(subset=['lat', 'lon', measurement])
            if len(valid_data) >= 3:
                valid_measurements.append(measurement)
        
        if not valid_measurements:
            raise ProcessingError("No valid data for any selected measurements")
        
        if progress_callback:
            progress_callback(6, f"Validated {len(valid_measurements)} measurements with sufficient data")
        
                # Calculate global color range if requested for consistent comparison
        global_vmin = global_vmax = None
        if use_global_color_range:
            if progress_callback:
                progress_callback(7, "Calculating global color range across all measurements...")
            
            all_values = []
            for measurement in valid_measurements:
                valid_data = data.dropna(subset=['lat', 'lon', measurement])
                values = valid_data[measurement].values
                all_values.extend(values)
            
            if progress_callback:
                progress_callback(8, f"Analyzing {len(all_values)} data points for color scaling...")
            
            # Calculate global statistics for consistent color scaling
            global_mean = np.mean(all_values)
            global_std = np.std(all_values)
            global_vmin = global_mean - 2 * global_std
            global_vmax = global_mean + 2 * global_std
            
            if progress_callback:
                progress_callback(9, f"Global color range: [{global_vmin:.2f}, {global_vmax:.2f}] across {len(valid_measurements)} measurements")
        else:
            if progress_callback:
                progress_callback(7, "Using individual color ranges for each measurement")

        # Update progress calculation for valid measurements
        measurements = valid_measurements
        num_meas = len(measurements)
        progress_per_meas = 85 / num_meas if num_meas > 0 else 85

        if progress_callback:
            progress_callback(10, f"Starting processing of {num_meas} measurements...")

        for i, measurement in enumerate(measurements):
            current_progress = 10 + i * progress_per_meas
            if progress_callback:
                progress_callback(int(current_progress), f"Processing {measurement} ({i+1}/{num_meas})...")

            valid_data = data.dropna(subset=['lat', 'lon', measurement])
            lats = valid_data['lat'].values
            lons = valid_data['lon'].values
            values = valid_data[measurement].values

            # Calculate color range based on user preference
            if use_global_color_range:
                # Use global color range for consistent comparison across measurements
                vmin = global_vmin
                vmax = global_vmax
            else:
                # Use individual color range for each measurement
                mean_val = np.mean(values)
                std_val = np.std(values)
                vmin = mean_val - 2 * std_val
                vmax = mean_val + 2 * std_val

            # Use minimum curvature interpolation for better results
            if progress_callback:
                if NUMBA_AVAILABLE:
                    progress_callback(int(current_progress + progress_per_meas * 0.1), f"🚀 Applying JIT-compiled minimum curvature interpolation for {measurement}...")
                else:
                    progress_callback(int(current_progress + progress_per_meas * 0.1), f"🔬 Applying minimum curvature interpolation for {measurement}...")
            
            if method == 'minimum_curvature':
                # Use advanced minimum curvature interpolation
                try:
                    if progress_callback:
                        if NUMBA_AVAILABLE:
                            progress_callback(int(current_progress + progress_per_meas * 0.12), f"🚀 Starting JIT-compiled minimum curvature with {grid_resolution}x{grid_resolution} grid...")
                        else:
                            progress_callback(int(current_progress + progress_per_meas * 0.12), f"🔬 Starting minimum curvature with {grid_resolution}x{grid_resolution} grid...")
                    
                    # Create minimum curvature interpolation - note coordinate order (lon, lat)
                    lon_mesh, lat_mesh, interpolated = self.minimum_curvature_interpolation(
                        lons, lats, values, 
                        grid_resolution=grid_resolution,
                        max_iterations=max_iterations,
                        tolerance=tolerance,
                        omega=omega,
                        progress_callback=lambda p, msg: progress_callback(int(current_progress + progress_per_meas * (0.12 + p * 0.68 / 100)), msg) if progress_callback else None
                    )
                    
                    if progress_callback:
                        if NUMBA_AVAILABLE:
                            progress_callback(int(current_progress + progress_per_meas * 0.82), "✅ JIT-compiled minimum curvature interpolation completed")
                        else:
                            progress_callback(int(current_progress + progress_per_meas * 0.82), "✅ Minimum curvature interpolation completed")
                    
                except Exception as mc_error:
                    if progress_callback:
                        progress_callback(int(current_progress + progress_per_meas * 0.4), 
                                        f"⚠️ Minimum curvature failed ({str(mc_error)[:50]}), using cubic interpolation...")
                    
                    # Fallback to scipy interpolation
                    points = np.column_stack((lats, lons))
                    lat_min, lat_max = float(lats.min()), float(lats.max())
                    lon_min, lon_max = float(lons.min()), float(lons.max())
                    
                    lat_grid = np.linspace(lat_min, lat_max, grid_resolution)
                    lon_grid = np.linspace(lon_min, lon_max, grid_resolution)
                    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
                    
                    grid_points = np.column_stack((lat_mesh.ravel(), lon_mesh.ravel()))
                    interpolated_values = griddata(
                        points, values, grid_points, 
                        method='cubic', fill_value=np.nan
                    )
                    interpolated = interpolated_values.reshape(lat_mesh.shape)
                    
            else:
                # Direct scipy interpolation for other methods
                if progress_callback:
                    progress_callback(int(current_progress + progress_per_meas * 0.4), f"📊 Using {method} interpolation...")
                
                points = np.column_stack((lats, lons))
                lat_min, lat_max = float(lats.min()), float(lats.max())
                lon_min, lon_max = float(lons.min()), float(lons.max())
                
                lat_grid = np.linspace(lat_min, lat_max, grid_resolution)
                lon_grid = np.linspace(lon_min, lon_max, grid_resolution)
                lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
                
                grid_points = np.column_stack((lat_mesh.ravel(), lon_mesh.ravel()))
                interpolated_values = griddata(
                    points, values, grid_points, 
                    method=method, fill_value=np.nan
                )
                interpolated = interpolated_values.reshape(lat_mesh.shape)

            # Apply boundary masking if enabled
            if enable_masking and method == 'minimum_curvature':
                if progress_callback:
                    progress_callback(int(current_progress + progress_per_meas * 0.84), f"🎯 Applying boundary masking ({boundary_method})...")
                
                try:
                    boundary_mask = self.create_boundary_mask(lons, lats, lon_mesh, lat_mesh, boundary_method)
                    interpolated = np.where(boundary_mask, interpolated, np.nan)
                    
                    if progress_callback:
                        progress_callback(int(current_progress + progress_per_meas * 0.86), "✅ Boundary masking applied")
                except Exception:
                    if progress_callback:
                        progress_callback(int(current_progress + progress_per_meas * 0.86), "⚠️ Boundary masking failed, skipping...")

            # Clip extreme values to avoid colormap issues  
            interpolated_clipped = np.clip(interpolated, vmin, vmax)

            # Add raster layer
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            base_name = Path(input_file_path).stem if input_file_path else 'data'
            layer_name = f'{base_name} - {measurement} Interpolated ({timestamp})'

            color_ramp = gamma_color_ramps.get(measurement, gamma_colors)

            # Calculate bounds from the actual grid coordinates
            lon_min = float(np.nanmin(lon_mesh))
            lon_max = float(np.nanmax(lon_mesh))
            lat_min = float(np.nanmin(lat_mesh))
            lat_max = float(np.nanmax(lat_mesh))

            raster_data = {
                'grid': interpolated_clipped,
                'bounds': [lon_min, lat_min, lon_max, lat_max],
                'field_name': measurement
            }

            result.add_layer_output(
                layer_type='raster',
                data=raster_data,
                style_info={
                    'use_graduated_colors': True,
                    'opacity': 0.7,
                    'color_ramp': color_ramp
                },
                metadata={
                    'layer_name': layer_name,
                    'description': f'Interpolated {measurement} field',
                    'data_type': f'gamma_{measurement.lower()}',
                    'grid_shape': interpolated_clipped.shape,
                    'vmin': float(vmin),
                    'vmax': float(vmax),
                    'global_range': use_global_color_range,  # Flag indicating color range type
                    'measurement_range': f'[{np.min(values):.2f}, {np.max(values):.2f}]'  # Individual range for reference
                }
            )

            # Add points layer if enabled
            if include_points:
                points_name = f'{base_name} - {measurement} Points ({timestamp})'
                result.add_layer_output(
                    layer_type='points',
                    data=valid_data[['lat', 'lon', measurement]],
                    style_info={
                        'color_field': measurement,
                        'use_graduated_colors': True,
                        'size': 4,
                        'opacity': 0.8,
                        'color_ramp': color_ramp
                    },
                    metadata={
                        'description': f'{measurement} data points',
                        'data_type': f'gamma_points_{measurement.lower()}',
                        'layer_name': points_name,
                        'coordinate_columns': {'latitude': 'lat', 'longitude': 'lon'},
                        'vmin': float(vmin),
                        'vmax': float(vmax),
                        'global_range': use_global_color_range,  # Flag indicating color range type
                        'measurement_range': f'[{np.min(values):.2f}, {np.max(values):.2f}]'  # Individual range for reference
                    }
                )

            # Collect for plot
            if generate_plot:
                figures.append((measurement, lon_mesh, lat_mesh, interpolated_clipped, lon_min, lat_min, lon_max, lat_max, vmin, vmax))

            if progress_callback:
                progress_callback(int(current_progress + progress_per_meas * 0.95), f"✅ {measurement} interpolation complete")

                # Generate summary plot if enabled
        if generate_plot and figures:
            if progress_callback:
                progress_callback(96, "Generating summary plot...")
            
            nrows = int(np.ceil(np.sqrt(len(figures))))
            ncols = int(np.ceil(len(figures) / nrows))
            fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)
            color_range_note = " (Global Color Range)" if use_global_color_range else " (Individual Color Ranges)"
            fig.suptitle(f'Gamma Interpolations{color_range_note}')

            for idx, (meas, lon_m, lat_m, grid, lmin, lamin, lmax, lamax, vm, vx) in enumerate(figures):
                row = idx // ncols
                col = idx % ncols
                ax = axs[row, col]
                im = ax.imshow(grid, extent=(lmin, lmax, lamin, lamax), origin='lower', cmap=mcolors.LinearSegmentedColormap.from_list(f'{meas}_cmap', gamma_color_ramps.get(meas, gamma_colors)), vmin=vm, vmax=vx, aspect='auto')
                ax.set_title(meas)
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')
                fig.colorbar(im, ax=ax, shrink=0.8)

            # Hide unused subplots
            for idx in range(len(figures), nrows * ncols):
                row = idx // ncols
                col = idx % ncols
                axs[row, col].axis('off')

            fig.tight_layout()
            result.figure = fig

        if progress_callback:
            progress_callback(100, "✅ Gamma interpolation complete!")

        return result

SCRIPT_CLASS = GammaInterpolator 