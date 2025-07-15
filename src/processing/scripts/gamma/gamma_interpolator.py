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


class GammaInterpolator(ScriptInterface):
    @property
    def name(self) -> str:
        return "Gamma Interpolator"

    @property
    def description(self) -> str:
        jit_status = "with JIT-accelerated minimum curvature" if NUMBA_AVAILABLE else "(install numba for JIT acceleration)"
        return f"Interpolates gamma measurements to grid visualizations and point layers {jit_status}"

    def get_parameters(self) -> Dict[str, Any]:
        return {
            'interpolation': {
                'method': {
                    'value': 'minimum_curvature',
                    'type': 'choice',
                    'choices': ['minimum_curvature', 'linear', 'cubic', 'nearest'],
                    'description': 'Interpolation method (minimum_curvature uses advanced JIT-compiled algorithm)'
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

    @staticmethod
    @jit(nopython=True, parallel=True, cache=True)
    def minimum_curvature_iteration_jit(grid_z, data_mask, data_values, omega, min_allowed, max_allowed):
        """
        JIT-compiled core iteration for minimum curvature interpolation.
        
        This function performs one iteration of the minimum curvature algorithm
        using a 5-point stencil and successive over-relaxation.
        
        Args:
            grid_z: Current grid values
            data_mask: Boolean mask indicating data constraint points
            data_values: Values at data constraint points
            omega: Relaxation factor
            min_allowed, max_allowed: Value clamps for stability
        
        Returns:
            max_change: Maximum change in grid values during this iteration
        """
        ny, nx = grid_z.shape
        max_change = 0.0
        
        # Create a copy of the grid for the new values
        new_grid_z = grid_z.copy()
        
        # Update interior points using 5-point stencil
        for i in range(1, ny-1):
            for j in range(1, nx-1):
                if not data_mask[i, j]:  # Don't modify data constraint points
                    old_value = grid_z[i, j]
                    
                    # 5-point stencil for Laplacian
                    neighbors = (grid_z[i+1, j] + grid_z[i-1, j] + 
                               grid_z[i, j+1] + grid_z[i, j-1]) / 4.0
                    
                    # Successive over-relaxation
                    new_value = old_value + omega * (neighbors - old_value)
                    
                    # Clamp values for stability
                    if new_value < min_allowed:
                        new_value = min_allowed
                    elif new_value > max_allowed:
                        new_value = max_allowed
                    
                    new_grid_z[i, j] = new_value
                    
                    # Track maximum change
                    change = abs(new_value - old_value)
                    if change > max_change:
                        max_change = change
        
        # Apply boundary conditions
        if ny >= 3:
            # Top and bottom boundaries (zero second derivative)
            for j in range(nx):
                new_grid_z[0, j] = 2*new_grid_z[1, j] - new_grid_z[2, j]
                new_grid_z[ny-1, j] = 2*new_grid_z[ny-2, j] - new_grid_z[ny-3, j]
        
        if nx >= 3:
            # Left and right boundaries (zero second derivative)
            for i in range(ny):
                new_grid_z[i, 0] = 2*new_grid_z[i, 1] - new_grid_z[i, 2]
                new_grid_z[i, nx-1] = 2*new_grid_z[i, nx-2] - new_grid_z[i, nx-3]
        
        # Re-apply data constraints
        for i in range(ny):
            for j in range(nx):
                if data_mask[i, j]:
                    new_grid_z[i, j] = data_values[i, j]
        
        # Copy results back
        for i in range(ny):
            for j in range(nx):
                grid_z[i, j] = new_grid_z[i, j]
        
        return max_change

    def _minimum_curvature_iteration_python(self, grid_z, data_mask, data_values, omega, min_allowed, max_allowed):
        """Python fallback using vectorized operations"""
        ny, nx = grid_z.shape
        grid_z_old = grid_z.copy()
        
        # Vectorized computation for interior points
        interior_mask = np.zeros_like(data_mask)
        interior_mask[1:-1, 1:-1] = True
        update_mask = interior_mask & ~data_mask
        
        if np.any(update_mask):
            # Vectorized 5-point stencil computation
            neighbors = np.zeros_like(grid_z)
            neighbors[1:-1, 1:-1] = (grid_z[2:, 1:-1] +     # i+1, j
                                   grid_z[:-2, 1:-1] +     # i-1, j  
                                   grid_z[1:-1, 2:] +      # i, j+1
                                   grid_z[1:-1, :-2]) / 4.0 # i, j-1
            
            # Vectorized relaxation update
            new_values = grid_z + omega * (neighbors - grid_z)
            new_values = np.clip(new_values, min_allowed, max_allowed)
            grid_z[update_mask] = new_values[update_mask]
        
        # Vectorized boundary conditions
        if ny >= 3:
            grid_z[0, :] = 2*grid_z[1, :] - grid_z[2, :]
            grid_z[-1, :] = 2*grid_z[-2, :] - grid_z[-3, :]
        if nx >= 3:
            grid_z[:, 0] = 2*grid_z[:, 1] - grid_z[:, 2]
            grid_z[:, -1] = 2*grid_z[:, -2] - grid_z[:, -3]
        
        # Re-apply data constraints
        grid_z[data_mask] = data_values[data_mask]
        
        # Compute maximum change
        changes = np.abs(grid_z - grid_z_old)
        max_change = np.max(changes[update_mask]) if np.any(update_mask) else 0.0
        
        return max_change

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
        Perform minimum curvature interpolation on irregular data points.
        
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
            progress_callback(0.1, "Setting up interpolation grid...")
        
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
            progress_callback(0.18, "Creating initial grid with linear interpolation...")
        
        # Create initial grid using linear interpolation to fill entire domain
        grid_z_initial = griddata((x, y), z, (grid_X, grid_Y), method='linear', fill_value=np.nan)
        
        if progress_callback:
            progress_callback(0.20, "Filling gaps with nearest neighbor interpolation...")
        
        # Fill any remaining NaN values with nearest neighbor interpolation
        nan_mask = np.isnan(grid_z_initial)
        if np.any(nan_mask):
            grid_z_nearest = griddata((x, y), z, (grid_X, grid_Y), method='nearest', fill_value=np.mean(z))
            grid_z_initial[nan_mask] = grid_z_nearest[nan_mask]
        
        grid_z = grid_z_initial.copy()
        
        if progress_callback:
            progress_callback(0.22, "Mapping data points to grid...")
        
        # Create data constraint mask and values
        ny, nx = grid_z.shape
        dx = (x_max - x_min) / (nx - 1)
        dy = (y_max - y_min) / (ny - 1)
        
        data_mask = np.zeros((ny, nx), dtype=bool)
        data_values = np.zeros((ny, nx))
        
        # Map data points to grid points with improved mapping
        for xi, yi, zi in zip(x, y, z):
            # Find closest grid point with bounds checking
            i = int(np.clip(round((yi - y_min) / dy), 0, ny-1))
            j = int(np.clip(round((xi - x_min) / dx), 0, nx-1))
            
            # Use distance-weighted averaging if multiple data points map to same grid point
            if data_mask[i, j]:
                # Average with existing value
                data_values[i, j] = (data_values[i, j] + zi) / 2
            else:
                data_mask[i, j] = True
                data_values[i, j] = zi
        
        # Apply data constraints to the initial grid
        grid_z[data_mask] = data_values[data_mask]
        
        # Value bounds for stability
        data_range = np.ptp(z)
        data_mean = np.mean(z)
        min_allowed = data_mean - 3 * data_range
        max_allowed = data_mean + 3 * data_range
        
        if progress_callback:
            progress_callback(0.3, "Starting minimum curvature iterations...")
        
        # Use stable omega value
        omega = 0.5  # Lower omega for stability
        
        if progress_callback:
            progress_callback(0.25, f"Fixed {np.sum(data_mask)} grid points to data values, starting {max_iterations} iterations...")
        
        # Iterative minimum curvature
        for iteration in range(max_iterations):
            if NUMBA_AVAILABLE:
                max_change = self.minimum_curvature_iteration_jit(
                    grid_z, data_mask, data_values, omega, min_allowed, max_allowed
                )
            else:
                max_change = self._minimum_curvature_iteration_python(
                    grid_z, data_mask, data_values, omega, min_allowed, max_allowed
                )
            
            # Early detection of instability
            if max_change > data_range:
                if progress_callback:
                    progress_callback(0.8, f"Large change detected at iteration {iteration+1}, stopping early")
                break
            
            # Progress update
            if progress_callback and iteration % 25 == 0:
                progress = 0.25 + 0.55 * (iteration / max_iterations)
                progress_callback(progress, f"Iteration {iteration+1}/{max_iterations}, max change: {max_change:.2e}")
            
            # Check for convergence
            if max_change < tolerance:
                if progress_callback:
                    progress_callback(0.8, f"Converged after {iteration+1} iterations")
                break
        
        if progress_callback:
            progress_callback(0.82, "Minimum curvature interpolation complete")
        
        return grid_X, grid_Y, grid_z

    def execute(self, data: pd.DataFrame, params: Dict[str, Any], 
                progress_callback: Optional[Callable] = None, input_file_path: Optional[str] = None) -> ProcessingResult:
        
        result = ProcessingResult(success=True, processing_script=self.name)
        result.metadata['processor'] = 'gamma'

        if progress_callback:
            progress_callback(0, "Starting gamma interpolation...")

        # Log interpolation capabilities
        if NUMBA_AVAILABLE:
            logger.info("🔬 JIT-accelerated minimum curvature interpolation available")
        else:
            logger.info("📊 Using scipy interpolation (install 'numba' for JIT acceleration)")

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

        # List of measurements to process
        measurements = []
        for m in ['Countrate', 'U238', 'K40', 'Th232', 'Cs137', 'Height', 'Press', 'Temp', 'Hum']:
            if m.lower() in meas_params and meas_params[m.lower()].get('value', False) and m in data.columns:
                measurements.append(m)

        if not measurements:
            raise ProcessingError("No measurements selected or available")

        num_meas = len(measurements)
        progress_per_meas = 0.9 / num_meas if num_meas > 0 else 0.9

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

        # Pre-process measurements to find valid ones
        valid_measurements = []
        for measurement in measurements:
            valid_data = data.dropna(subset=['lat', 'lon', measurement])
            if len(valid_data) >= 3:
                valid_measurements.append(measurement)
        
        if not valid_measurements:
            raise ProcessingError("No valid data for any selected measurements")
        
        # Calculate global color range if requested for consistent comparison
        global_vmin = global_vmax = None
        if use_global_color_range:
            if progress_callback:
                progress_callback(0.04, "Calculating global color range across all measurements...")
            
            all_values = []
            for measurement in valid_measurements:
                valid_data = data.dropna(subset=['lat', 'lon', measurement])
                values = valid_data[measurement].values
                all_values.extend(values)
            
            # Calculate global statistics for consistent color scaling
            global_mean = np.mean(all_values)
            global_std = np.std(all_values)
            global_vmin = global_mean - 2 * global_std
            global_vmax = global_mean + 2 * global_std
            
            if progress_callback:
                progress_callback(0.045, f"Global color range: [{global_vmin:.2f}, {global_vmax:.2f}] across {len(valid_measurements)} measurements")
        else:
            if progress_callback:
                progress_callback(0.04, f"Using individual color ranges for each measurement")

        # Update progress calculation for valid measurements
        measurements = valid_measurements
        num_meas = len(measurements)
        progress_per_meas = 0.9 / num_meas if num_meas > 0 else 0.9

        for i, measurement in enumerate(measurements):
            current_progress = 0.05 + i * progress_per_meas
            if progress_callback:
                progress_callback(current_progress, f"Processing {measurement}...")

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
                progress_callback(current_progress + progress_per_meas * 0.3, f"🔬 Applying minimum curvature interpolation for {measurement}...")
            
            if method == 'minimum_curvature':
                # Use advanced minimum curvature interpolation
                try:
                    if progress_callback:
                        progress_callback(current_progress + progress_per_meas * 0.35, f"🔬 Starting minimum curvature with {grid_resolution}x{grid_resolution} grid...")
                    
                    # Create minimum curvature interpolation - note coordinate order (lon, lat)
                    lon_mesh, lat_mesh, interpolated = self.minimum_curvature_interpolation(
                        lons, lats, values, 
                        grid_resolution=grid_resolution,
                        max_iterations=max_iterations,
                        tolerance=tolerance,
                        omega=omega,
                        progress_callback=lambda p, msg: progress_callback(current_progress + progress_per_meas * (0.35 + p * 0.4), msg) if progress_callback else None
                    )
                    
                    if progress_callback:
                        progress_callback(current_progress + progress_per_meas * 0.75, f"✅ Minimum curvature interpolation completed")
                    
                except Exception as mc_error:
                    if progress_callback:
                        progress_callback(current_progress + progress_per_meas * 0.4, 
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
                    progress_callback(current_progress + progress_per_meas * 0.4, f"📊 Using {method} interpolation...")
                
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
                    progress_callback(current_progress + progress_per_meas * 0.78, f"🎯 Applying boundary masking ({boundary_method})...")
                
                try:
                    boundary_mask = self.create_boundary_mask(lons, lats, lon_mesh, lat_mesh, boundary_method)
                    interpolated = np.where(boundary_mask, interpolated, np.nan)
                    
                    if progress_callback:
                        progress_callback(current_progress + progress_per_meas * 0.8, f"✅ Boundary masking applied")
                except Exception:
                    if progress_callback:
                        progress_callback(current_progress + progress_per_meas * 0.8, f"⚠️ Boundary masking failed, skipping...")

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
                progress_callback(current_progress + progress_per_meas * 0.8, f"{measurement} interpolation complete")

        # Generate summary plot if enabled
        if generate_plot and figures:
            if progress_callback:
                progress_callback(0.95, "Generating summary plot...")

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
            progress_callback(1.0, "Gamma interpolation complete")

        return result

SCRIPT_CLASS = GammaInterpolator 