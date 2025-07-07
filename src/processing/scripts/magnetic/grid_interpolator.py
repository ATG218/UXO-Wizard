"""
Grid Interpolation Script for UXO Wizard Framework
==================================================

Performs 2D grid interpolation of magnetic survey data using minimum curvature
interpolation method. Transforms sparse linear flight path data into continuous
2D grids suitable for visualization and analysis.

Features:
- Single file processing with automatic magnetic field column detection
- Minimum curvature interpolation with JIT compilation for performance
- Boundary masking to prevent unreliable extrapolation
- Raster layer generation for UXO Wizard map visualization
- Configurable grid resolution and interpolation parameters
- Comprehensive diagnostic output and quality control

Original: grid_interpolator.py from mag_import pipeline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Optional, Callable
import tempfile
import warnings
warnings.filterwarnings('ignore')

from src.processing.base import ScriptInterface, ProcessingResult, ProcessingError

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


class GridInterpolator(ScriptInterface):
    """
    Grid interpolation script for magnetic survey data processing
    """
    
    @property
    def name(self) -> str:
        return "Grid Interpolator"
    
    @property  
    def description(self) -> str:
        return "Perform 2D grid interpolation of magnetic survey data using minimum curvature method"
    
    def get_parameters(self) -> Dict[str, Any]:
        """Return parameter structure for grid interpolation"""
        return {
            'interpolation_parameters': {
                'grid_resolution': {
                    'value': 300,
                    'type': 'int',
                    'description': 'Number of grid points per axis (total grid = resolution²)'
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
            'boundary_parameters': {
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
                },
                'buffer_distance': {
                    'value': 0.0,
                    'type': 'float',
                    'description': 'Buffer distance for boundary expansion (coordinate units)'
                }
            },
            'output_parameters': {
                'generate_diagnostics': {
                    'value': True,
                    'type': 'bool',
                    'description': 'Generate diagnostic plots and statistics'
                },
                'save_grid_data': {
                    'value': True,
                    'type': 'bool',
                    'description': 'Save interpolated grid data as CSV'
                },
                'magnetic_field_column': {
                    'value': 'auto',
                    'type': 'choice',
                    'choices': ['auto', 'R1 [nT]', 'Btotal1 [nT]', 'Total [nT]', 'B_total [nT]'],
                    'description': 'Magnetic field column to interpolate'
                }
            }
        }
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data for grid interpolation"""
        required_columns = [
            'Latitude [Decimal Degrees]', 
            'Longitude [Decimal Degrees]'
        ]
        
        # Check for required coordinate columns
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            raise ProcessingError(f"Missing required columns: {missing_cols}")
        
        # Check for magnetic field columns
        magnetic_columns = ['R1 [nT]', 'Btotal1 [nT]', 'Total [nT]', 'B_total [nT]']
        has_magnetic_field = any(col in data.columns for col in magnetic_columns)
        
        if not has_magnetic_field:
            raise ProcessingError(f"No magnetic field column found. Looking for one of: {magnetic_columns}")
        
        # Check for minimum data points
        if len(data) < 10:
            raise ProcessingError(f"Insufficient data points: {len(data)}. Need at least 10 points.")
        
        # Check for valid coordinates
        lat_valid = data['Latitude [Decimal Degrees]'].between(-90, 90).all()
        lon_valid = data['Longitude [Decimal Degrees]'].between(-180, 180).all()
        
        if not lat_valid or not lon_valid:
            raise ProcessingError("Invalid latitude or longitude values detected")
        
        return True
    
    def get_magnetic_field_column(self, data: pd.DataFrame, target_field: str = 'auto') -> str:
        """Get the name of the magnetic field column in the DataFrame"""
        
        # If specific field is requested and exists, use it
        if target_field != "auto" and target_field in data.columns:
            return target_field
        
        # Otherwise, search for available magnetic field columns
        magnetic_columns = ['R1 [nT]', 'Btotal1 [nT]', 'Total [nT]', 'B_total [nT]']
        for col in magnetic_columns:
            if col in data.columns:
                return col
        
        raise ProcessingError(f"Target field '{target_field}' not found and no standard magnetic field columns available")
    
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
    
    def create_boundary_mask(self, x, y, grid_x, grid_y, method='convex_hull', buffer_distance=None):
        """
        Create a boundary mask to prevent interpolation outside data coverage.
        
        Args:
            x, y: Data point coordinates
            grid_x, grid_y: Grid meshgrid coordinates
            method: 'convex_hull' or 'alpha_shape'
            buffer_distance: Optional buffer distance for expanding the boundary
        
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
            
            # Apply buffer if specified
            if buffer_distance and buffer_distance > 0:
                # Simple buffer by expanding the mask
                distances = cdist(grid_points, data_points)
                min_distances = np.min(distances, axis=1)
                mask_1d = mask_1d | (min_distances <= buffer_distance)
            
            # Reshape to grid shape
            mask = mask_1d.reshape(grid_x.shape)
            
            return mask
            
        except Exception:
            # Fallback to no masking
            return np.ones(grid_x.shape, dtype=bool)
    
    def minimum_curvature_interpolation(self, x, y, z, grid_resolution=300, 
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
        
        # Add minimal padding to grid bounds (reduce from 5% to 1%)
        x_range = x_max - x_min
        y_range = y_max - y_min
        padding = 0.01  # 1% padding to reduce oversized raster
        
        x_min -= padding * x_range
        x_max += padding * x_range
        y_min -= padding * y_range
        y_max += padding * y_range
        
        grid_x = np.linspace(x_min, x_max, grid_resolution)
        grid_y = np.linspace(y_min, y_max, grid_resolution)
        grid_X, grid_Y = np.meshgrid(grid_x, grid_y)
        
        # Initialize grid with linear interpolation first
        from scipy.interpolate import griddata
        
        # Create initial grid using linear interpolation to fill entire domain
        grid_z_initial = griddata((x, y), z, (grid_X, grid_Y), method='linear', fill_value=np.nan)
        
        # Fill any remaining NaN values with nearest neighbor interpolation
        nan_mask = np.isnan(grid_z_initial)
        if np.any(nan_mask):
            grid_z_nearest = griddata((x, y), z, (grid_X, grid_Y), method='nearest', fill_value=np.mean(z))
            grid_z_initial[nan_mask] = grid_z_nearest[nan_mask]
        
        grid_z = grid_z_initial.copy()
        
        if progress_callback:
            progress_callback(0.2, "Mapping data points to grid...")
        
        # Create data constraint mask and values like the original script
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
        min_allowed = np.min(z) - 2 * np.std(z)
        max_allowed = np.max(z) + 2 * np.std(z)
        
        if progress_callback:
            progress_callback(0.3, "Starting minimum curvature iterations...")
        
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
            
            # Progress update
            if progress_callback and iteration % 50 == 0:
                progress = 0.3 + 0.6 * (iteration / max_iterations)
                progress_callback(progress, f"Iteration {iteration+1}/{max_iterations}, change: {max_change:.2e}")
            
            # Check for convergence
            if max_change < tolerance:
                if progress_callback:
                    progress_callback(0.9, f"Converged after {iteration+1} iterations")
                break
        
        if progress_callback:
            progress_callback(1.0, "Interpolation complete")
        
        return grid_X, grid_Y, grid_z
    
    def _minimum_curvature_iteration_python(self, grid_z, data_mask, data_values, omega, min_allowed, max_allowed):
        """Python fallback using vectorized operations like the original script"""
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
    
    def create_diagnostic_plot(self, x, y, z, grid_X, grid_Y, grid_z, field_name, output_dir):
        """Create diagnostic plots for interpolation quality assessment"""
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Grid Interpolation Diagnostics - {field_name}', fontsize=16)
        
        # Plot 1: Data distribution
        axes[0, 0].scatter(x, y, c=z, cmap='RdYlBu_r', s=2, alpha=0.7)
        axes[0, 0].set_title('Original Data Points')
        axes[0, 0].set_xlabel('Longitude')
        axes[0, 0].set_ylabel('Latitude')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Interpolated grid
        im = axes[0, 1].contourf(grid_X, grid_Y, grid_z, levels=50, cmap='RdYlBu_r')
        axes[0, 1].set_title('Interpolated Grid')
        axes[0, 1].set_xlabel('Longitude')
        axes[0, 1].set_ylabel('Latitude')
        plt.colorbar(im, ax=axes[0, 1])
        
        # Plot 3: Data histogram
        axes[1, 0].hist(z, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Data Value Distribution')
        axes[1, 0].set_xlabel(f'{field_name} Value')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Grid statistics
        grid_stats = {
            'Min': np.min(grid_z),
            'Max': np.max(grid_z),
            'Mean': np.mean(grid_z),
            'Std': np.std(grid_z),
            'Data Min': np.min(z),
            'Data Max': np.max(z),
            'Data Mean': np.mean(z),
            'Data Std': np.std(z)
        }
        
        stats_text = '\n'.join([f'{key}: {value:.2f}' for key, value in grid_stats.items()])
        axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[1, 1].set_title('Interpolation Statistics')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Save diagnostic plot
        diagnostic_path = output_dir / f'interpolation_diagnostics_{field_name}.png'
        plt.savefig(diagnostic_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return diagnostic_path
    
    def execute(self, data: pd.DataFrame, params: Dict[str, Any], 
                progress_callback: Optional[Callable] = None, input_file_path: Optional[str] = None) -> ProcessingResult:
        """
        Execute grid interpolation on magnetic survey data
        
        Args:
            data: Input DataFrame with magnetic survey data
            params: Processing parameters
            progress_callback: Optional callback for progress updates
            input_file_path: Optional path to the original input file
        
        Returns:
            ProcessingResult with interpolated grid layer and diagnostic outputs
        """
        
        # Create result object
        result = ProcessingResult(success=True)
        
        try:
            # Validate input data
            if progress_callback:
                progress_callback(0.05, "Validating input data...")
            
            self.validate_data(data)
            
            # Get processing parameters
            interp_params = params.get('interpolation_parameters', {})
            boundary_params = params.get('boundary_parameters', {})
            output_params = params.get('output_parameters', {})
            
            grid_resolution = interp_params.get('grid_resolution', {}).get('value', 300)
            max_iterations = interp_params.get('max_iterations', {}).get('value', 1000)
            tolerance = interp_params.get('tolerance', {}).get('value', 1e-6)
            omega = interp_params.get('relaxation_factor', {}).get('value', 1.8)
            
            enable_masking = boundary_params.get('enable_boundary_masking', {}).get('value', True)
            boundary_method = boundary_params.get('boundary_method', {}).get('value', 'convex_hull')
            buffer_distance = boundary_params.get('buffer_distance', {}).get('value', 0.0)
            
            target_field = output_params.get('magnetic_field_column', {}).get('value', 'auto')
            generate_diagnostics = output_params.get('generate_diagnostics', {}).get('value', True)
            save_grid_data = output_params.get('save_grid_data', {}).get('value', True)
            
            # Get magnetic field column
            field_column = self.get_magnetic_field_column(data, target_field)
            
            if progress_callback:
                progress_callback(0.1, f"Processing {len(data)} data points using field: {field_column}")
            
            # Extract coordinates and values
            x = data['Longitude [Decimal Degrees]'].values
            y = data['Latitude [Decimal Degrees]'].values
            z = data[field_column].values
            
            # Remove NaN values
            valid_mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
            x, y, z = x[valid_mask], y[valid_mask], z[valid_mask]
            
            if len(x) == 0:
                raise ProcessingError("No valid data points after removing NaN values")
            
            # Perform minimum curvature interpolation
            if progress_callback:
                progress_callback(0.15, "Starting minimum curvature interpolation...")
            
            grid_X, grid_Y, grid_z = self.minimum_curvature_interpolation(
                x, y, z, grid_resolution, max_iterations, tolerance, omega, progress_callback
            )
            
            # Apply boundary masking if enabled
            if enable_masking:
                if progress_callback:
                    progress_callback(0.92, "Applying boundary masking...")
                
                mask = self.create_boundary_mask(x, y, grid_X, grid_Y, boundary_method, buffer_distance)
                grid_z = np.where(mask, grid_z, np.nan)
            
            # Create temporary directory for outputs
            temp_dir = Path(tempfile.mkdtemp())
            
            # Generate diagnostic plot if requested
            if generate_diagnostics:
                if progress_callback:
                    progress_callback(0.95, "Generating diagnostic plots...")
                
                diagnostic_path = self.create_diagnostic_plot(
                    x, y, z, grid_X, grid_Y, grid_z, field_column, temp_dir
                )
                result.add_output_file(str(diagnostic_path), 'png', 'Interpolation diagnostic plots')
            
            # Save grid data if requested
            if save_grid_data:
                if progress_callback:
                    progress_callback(0.97, "Saving grid data...")
                
                # Create grid DataFrame
                grid_df = pd.DataFrame({
                    'longitude': grid_X.ravel(),
                    'latitude': grid_Y.ravel(),
                    'value': grid_z.ravel()
                })
                
                # Remove NaN values for CSV
                grid_df = grid_df.dropna()
                
                grid_csv_path = temp_dir / f'interpolated_grid_{field_column.replace(" ", "_").replace("[", "").replace("]", "")}.csv'
                grid_df.to_csv(grid_csv_path, index=False)
                result.add_output_file(str(grid_csv_path), 'csv', 'Interpolated grid data')
            
            # Create raster layer for visualization
            if progress_callback:
                progress_callback(0.99, "Creating raster layer...")
            
            # Calculate bounds for raster layer [min_x, min_y, max_x, max_y]
            bounds = [
                float(np.nanmin(grid_X)),  # west (min_x)
                float(np.nanmin(grid_Y)),  # south (min_y)
                float(np.nanmax(grid_X)),  # east (max_x)
                float(np.nanmax(grid_Y))   # north (max_y)
            ]
            
            # Create raster layer data - must have 'grid' key for UXO Wizard
            raster_data = {
                'grid': grid_z,
                'bounds': bounds,
                'field_name': field_column
            }
            
            # Add raster layer output
            result.add_layer_output(
                layer_type='raster',
                data=raster_data,
                style_info={},
                metadata={
                    'layer_name': f'Interpolated {field_column}',
                    'bounds': bounds  # Add bounds to metadata to help with layer creation
                }
            )
            
            # Add downsampled original data points as point layer
            point_data = data[['Latitude [Decimal Degrees]', 'Longitude [Decimal Degrees]', field_column]].copy()
            
            # Smart downsampling for map performance (preserves flight line structure)
            max_points = 5000
            if len(point_data) > max_points:
                # Use systematic sampling to preserve flight line structure
                step = len(point_data) // max_points
                if step > 1:
                    downsample_indices = np.arange(0, len(point_data), step)
                    point_data = point_data.iloc[downsample_indices].copy()
                    layer_name = f'Original Data Points (downsampled every {step} points)'
                else:
                    layer_name = 'Original Data Points'
            else:
                layer_name = 'Original Data Points'
            
            point_data = point_data.rename(columns={
                'Latitude [Decimal Degrees]': 'latitude',
                'Longitude [Decimal Degrees]': 'longitude',
                field_column: 'value'
            })
            
            result.add_layer_output(
                layer_type='point',
                data=point_data,
                style_info={},
                metadata={
                    'layer_name': layer_name,
                    'color_field': 'value',
                    'use_graduated_colors': True,
                    'point_size': 2,
                    'point_opacity': 0.8,
                    'enable_clustering': False
                }
            )
            
            if progress_callback:
                progress_callback(1.0, "Grid interpolation complete")
            
            # Add processing summary
            result.metadata.update({
                'processing_method': 'minimum_curvature',
                'grid_resolution': grid_resolution,
                'iterations_used': 'converged' if max_iterations > 100 else str(max_iterations),
                'field_processed': field_column,
                'data_points': len(data),
                'valid_points': len(x),
                'boundary_masking': enable_masking,
                'numba_acceleration': NUMBA_AVAILABLE
            })
            
            return result
            
        except Exception as e:
            raise ProcessingError(f"Grid interpolation failed: {str(e)}")


# Export the script class
SCRIPT_CLASS = GridInterpolator