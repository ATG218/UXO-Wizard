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
import datetime
import warnings

from src.processing.base import ScriptInterface, ProcessingResult, ProcessingError, NUMBA_AVAILABLE, jit, BaseProcessor

warnings.filterwarnings('ignore')


class _GridInterpolatorHelper(BaseProcessor):
    """Helper class to access BaseProcessor methods"""
    def validate_data(self, data):
        return True

class GridInterpolator(ScriptInterface):
    """
    Grid interpolation script for magnetic survey data processing
    """
    
    def __init__(self, project_manager=None):
        super().__init__(project_manager)
        # Initialize a helper processor instance for access to BaseProcessor methods
        self._base_processor = _GridInterpolatorHelper()
    
    @property
    def name(self) -> str:
        return "Grid Interpolator"
    
    @property  
    def description(self) -> str:
        return "Perform 2D grid interpolation of magnetic survey data using minimum curvature method with soft constraints to eliminate flight line artifacts"
    
    @property
    def handles_own_output(self) -> bool:
        """This script creates its own output files (grid CSV, diagnostic plots, analysis directory)"""
        return True
    
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
                },
                'constraint_mode': {
                    'value': 'soft',
                    'type': 'choice',
                    'choices': ['soft', 'hard'],
                    'description': 'Constraint mode: soft (eliminates flight line artifacts) or hard (traditional)'
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
                },
                'include_original_points': {
                    'value': False,
                    'type': 'bool',
                    'description': 'Include original data points as a map layer'
                },
                'generate_interactive_plot': {
                    'value': True,
                    'type': 'bool',
                    'description': 'Generate an interactive plot for the data viewer'
                },
                'plot_type': {
                    'value': '2D',
                    'type': 'choice',
                    'choices': ['2D'],
                    'description': 'Type of diagnostic plot to generate'
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
    
    def create_boundary_mask_with_buffer(self, x, y, grid_x, grid_y, method='convex_hull', buffer_distance=None):
        """
        Create a boundary mask with optional buffer - wraps the base class method
        
        Args:
            x, y: Data point coordinates
            grid_x, grid_y: Grid meshgrid coordinates
            method: 'convex_hull' or 'alpha_shape'
            buffer_distance: Optional buffer distance for expanding the boundary
        
        Returns:
            mask: Boolean array where True indicates points inside the data boundary
        """
        # Use the base processor method for core functionality
        mask = self._base_processor.create_boundary_mask(x, y, grid_x, grid_y, method)
        
        # Add buffer functionality if specified
        if buffer_distance and buffer_distance > 0:
            try:
                from scipy.spatial.distance import cdist
                # Get data points as array
                data_points = np.column_stack([x, y])
                # Get grid points as array
                grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
                
                # Simple buffer by expanding the mask
                distances = cdist(grid_points, data_points)
                min_distances = np.min(distances, axis=1)
                buffer_mask_1d = min_distances <= buffer_distance
                buffer_mask = buffer_mask_1d.reshape(grid_x.shape)
                
                # Combine with original mask
                mask = mask | buffer_mask
            except ImportError:
                # If scipy not available, just return original mask
                pass
        
        return mask
    
    def minimum_curvature_interpolation_enhanced(self, x, y, z, grid_resolution=300, 
                                                 max_iterations=1000, tolerance=1e-6, 
                                                 omega=1.8, constraint_mode='soft',
                                                 progress_callback=None):
        """
        Perform minimum curvature interpolation using the enhanced base class method.
        Supports both soft constraints (gamma style) and hard constraints (traditional style).
        
        Args:
            x, y, z: Data coordinates and values
            grid_resolution: Number of grid points per axis
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            omega: Relaxation factor for successive over-relaxation
            constraint_mode: 'soft' (eliminates flight line artifacts) or 'hard' (traditional)
            progress_callback: Optional callback for progress updates
        
        Returns:
            grid_x, grid_y, grid_z: Interpolated grid coordinates and values
        """
        # Use the base processor method with specified constraint mode
        return self._base_processor.minimum_curvature_interpolation(
            x, y, z, 
            grid_resolution=grid_resolution,
            max_iterations=max_iterations,
            tolerance=tolerance,
            omega=omega,
            constraint_mode=constraint_mode,
            progress_callback=progress_callback
        )
    
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
    
    def create_diagnostic_plot(self, x, y, z, grid_X, grid_Y, grid_z, field_name, output_dir, save_png=True):
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
        
        if save_png:
            # Save diagnostic plot
            diagnostic_path = output_dir / f'interpolation_diagnostics_{field_name}.png'
            plt.savefig(diagnostic_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            return diagnostic_path, None
        else:
            return None, fig # Return the figure object if not saving
    
    def create_contour_raster(self, grid_X, grid_Y, grid_z, bounds):
        """Create a contour lines raster overlay using matplotlib like the original script"""
        try:
            import matplotlib.pyplot as plt
            
            # Create contour levels
            field_min = np.nanmin(grid_z)
            field_max = np.nanmax(grid_z)
            
            if field_min >= field_max or np.isnan(field_min) or np.isnan(field_max):
                return None
                
            # Create contour levels
            num_contours = 10
            contour_levels = np.linspace(field_min, field_max, num_contours + 1)
            
            # Create matplotlib figure for contour calculation (not displayed)
            fig, ax = plt.subplots(figsize=(1, 1))
            ax.set_aspect('equal')
            
            # Generate contours
            cs = ax.contour(grid_X, grid_Y, grid_z, levels=contour_levels)
            
            # Create contour raster data matching the main field format
            contour_field = np.full_like(grid_z, np.nan)
            total_points_marked = 0
            
            # Handle different matplotlib versions exactly like the original script
            if hasattr(cs, 'collections'):
                collections = cs.collections
            else:
                # For newer matplotlib versions (QuadContourSet)
                collections = cs.allsegs if hasattr(cs, 'allsegs') else []
            
            if hasattr(cs, 'collections'):
                # Old matplotlib API
                for i, collection in enumerate(collections):
                    if i < len(contour_levels):
                        level = contour_levels[i]
                        
                        # Extract paths from collection
                        for path in collection.get_paths():
                            vertices = path.vertices
                            if len(vertices) > 0:
                                # Convert to grid indices
                                x_indices = np.round((vertices[:, 0] - bounds[0]) / (bounds[2] - bounds[0]) * (grid_z.shape[1] - 1)).astype(int)
                                y_indices = np.round((vertices[:, 1] - bounds[1]) / (bounds[3] - bounds[1]) * (grid_z.shape[0] - 1)).astype(int)
                                
                                # Clamp to valid range
                                x_indices = np.clip(x_indices, 0, grid_z.shape[1] - 1)
                                y_indices = np.clip(y_indices, 0, grid_z.shape[0] - 1)
                                
                                # Mark contour lines with their value
                                contour_field[y_indices, x_indices] = level
                                total_points_marked += len(x_indices)
            else:
                # New matplotlib API - use allsegs
                for i, level_segs in enumerate(collections):
                    if i >= len(contour_levels):
                        break
                    level = contour_levels[i]
                    
                    # Extract segments for this level
                    for seg in level_segs:
                        if len(seg) > 0:
                            # Convert to grid indices
                            x_indices = np.round((seg[:, 0] - bounds[0]) / (bounds[2] - bounds[0]) * (grid_z.shape[1] - 1)).astype(int)
                            y_indices = np.round((seg[:, 1] - bounds[1]) / (bounds[3] - bounds[1]) * (grid_z.shape[0] - 1)).astype(int)
                            
                            # Clamp to valid range
                            x_indices = np.clip(x_indices, 0, grid_z.shape[1] - 1)
                            y_indices = np.clip(y_indices, 0, grid_z.shape[0] - 1)
                            
                            # Mark contour lines with their value
                            contour_field[y_indices, x_indices] = level
                            total_points_marked += len(x_indices)
            
            plt.close(fig)  # Clean up matplotlib figure
            
            if total_points_marked == 0:
                return None
            
            return {
                'grid': contour_field,
                'bounds': bounds,
                'field_name': 'contours'
            }
            
        except Exception:
            return None
    
    def create_comprehensive_analysis(self, data, grid_X, grid_Y, grid_z, field_column, input_file_path, temp_dir):
        """Create comprehensive analysis like the original script"""
        try:
            # Create analysis directory in project/processed/magnetic/
            if input_file_path:
                input_path = Path(input_file_path)
                base_filename = input_path.stem
                
                # Find project root - look for working directory or use input file parent
                project_dir = input_path.parent
                while project_dir.parent != project_dir:  # Not at filesystem root
                    if (project_dir / "processed").exists() or len(list(project_dir.glob("*.uxo"))) > 0:
                        break
                    project_dir = project_dir.parent
                
                # Create project/processed/magnetic/filename_grid_analysis structure
                analysis_dir = project_dir / "processed" / "magnetic" / f"{base_filename}_grid_analysis"
            else:
                analysis_dir = temp_dir / 'grid_analysis'
            
            analysis_dir.mkdir(exist_ok=True)
            
            # Extract coordinates and values for analysis
            x = data['Longitude [Decimal Degrees]'].values
            y = data['Latitude [Decimal Degrees]'].values
            z = data[field_column].values
            
            # Remove NaN values
            valid_mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
            x_clean = x[valid_mask]
            y_clean = y[valid_mask]
            z_clean = z[valid_mask]
            
            # 1. Create field visualization plots
            self.create_field_plots(grid_X, grid_Y, grid_z, x_clean, y_clean, z_clean, field_column, analysis_dir)
            
            # 2. Create diagnostic plots  
            self.create_diagnostic_plots(grid_X, grid_Y, grid_z, x_clean, y_clean, z_clean, field_column, analysis_dir)
            
            # 3. Create data summary
            self.create_data_summary(data, grid_z, field_column, analysis_dir)
            
            # 4. Save processed grid as CSV
            self.save_processed_grid(grid_X, grid_Y, grid_z, field_column, analysis_dir)
            
            return analysis_dir
            
        except Exception:
            return None
    
    def create_field_plots(self, grid_X, grid_Y, grid_z, x_data, y_data, z_data, field_column, output_dir):
        """Create field visualization plots with contours"""
        try:
            # Create field plots directory
            field_plots_dir = output_dir / "field_visualizations"
            field_plots_dir.mkdir(exist_ok=True)
            
            # Global min/max for consistent scaling
            global_min = np.nanmin(grid_z)
            global_max = np.nanmax(grid_z)
            
            # Create comprehensive field plot
            _, axes = plt.subplots(1, 3, figsize=(20, 6))
            
            # Plot 1: Interpolated field with contours
            ax1 = axes[0]
            levels = np.linspace(global_min, global_max, 20)
            im1 = ax1.contourf(grid_X, grid_Y, grid_z, levels=levels, cmap='RdYlBu_r', extend='both')
            
            # Add contour lines
            contour_levels = np.linspace(global_min, global_max, 10)
            positive_levels = contour_levels[contour_levels >= 0]
            negative_levels = contour_levels[contour_levels < 0]
            
            if len(positive_levels) > 0:
                cs_pos = ax1.contour(grid_X, grid_Y, grid_z, levels=positive_levels, colors='red', linewidths=1.5, alpha=0.7)
                ax1.clabel(cs_pos, inline=True, fontsize=8, fmt='%d')
            
            if len(negative_levels) > 0:
                cs_neg = ax1.contour(grid_X, grid_Y, grid_z, levels=negative_levels, colors='blue', linewidths=1.5, alpha=0.7)
                ax1.clabel(cs_neg, inline=True, fontsize=8, fmt='%d')
            
            ax1.set_title(f'Interpolated {field_column}\nField with Contours')
            ax1.set_xlabel('Longitude')
            ax1.set_ylabel('Latitude')
            ax1.set_aspect('equal')
            plt.colorbar(im1, ax=ax1, shrink=0.8, label=field_column)
            
            # Plot 2: Original data points overlay
            ax2 = axes[1]
            ax2.contourf(grid_X, grid_Y, grid_z, levels=levels, cmap='RdYlBu_r', extend='both', alpha=0.7)
            scatter = ax2.scatter(x_data, y_data, c=z_data, cmap='RdYlBu_r', s=1, alpha=0.8, vmin=global_min, vmax=global_max)
            ax2.set_title(f'{field_column}\nField + Original Data Points')
            ax2.set_xlabel('Longitude')
            ax2.set_ylabel('Latitude')
            ax2.set_aspect('equal')
            plt.colorbar(scatter, ax=ax2, shrink=0.8, label=field_column)
            
            # Plot 3: Contour lines only
            ax3 = axes[2]
            ax3.set_facecolor('white')
            
            if len(positive_levels) > 0:
                cs_pos_clean = ax3.contour(grid_X, grid_Y, grid_z, levels=positive_levels, colors='red', linewidths=2)
                ax3.clabel(cs_pos_clean, inline=True, fontsize=10, fmt='%d nT')
            
            if len(negative_levels) > 0:
                cs_neg_clean = ax3.contour(grid_X, grid_Y, grid_z, levels=negative_levels, colors='blue', linewidths=2)
                ax3.clabel(cs_neg_clean, inline=True, fontsize=10, fmt='%d nT')
            
            # Zero contour
            zero_level = [0] if global_min <= 0 <= global_max else []
            if zero_level:
                cs_zero = ax3.contour(grid_X, grid_Y, grid_z, levels=zero_level, colors='black', linewidths=3)
                ax3.clabel(cs_zero, inline=True, fontsize=12, fmt='%d nT')
            
            ax3.set_title(f'{field_column}\nContour Lines Only')
            ax3.set_xlabel('Longitude')
            ax3.set_ylabel('Latitude')
            ax3.set_aspect('equal')
            ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(field_plots_dir / f'field_visualization_{field_column.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception:
            pass
    
    def create_diagnostic_plots(self, grid_X, grid_Y, grid_z, x_data, y_data, z_data, field_column, output_dir):
        """Create comprehensive diagnostic plots"""
        try:
            # Create diagnostic plots directory
            diagnostic_dir = output_dir / "diagnostic_plots"
            diagnostic_dir.mkdir(exist_ok=True)
            
            # Interpolate grid values at original data points for residual analysis
            from scipy.interpolate import griddata
            grid_at_data = griddata((grid_X.ravel(), grid_Y.ravel()), grid_z.ravel(), (x_data, y_data), method='linear')
            
            # Calculate residuals
            valid_interp_mask = ~np.isnan(grid_at_data)
            residuals = z_data[valid_interp_mask] - grid_at_data[valid_interp_mask]
            
            # Create diagnostic figure
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'Grid Interpolation Diagnostics - {field_column}', fontsize=16)
            
            # 1. Original data distribution
            axes[0, 0].scatter(x_data, y_data, c=z_data, cmap='RdYlBu_r', s=2, alpha=0.7)
            axes[0, 0].set_title('Original Data Points')
            axes[0, 0].set_xlabel('Longitude')
            axes[0, 0].set_ylabel('Latitude')
            axes[0, 0].set_aspect('equal')
            
            # 2. Interpolated grid
            im = axes[0, 1].contourf(grid_X, grid_Y, grid_z, levels=50, cmap='RdYlBu_r')
            axes[0, 1].set_title('Interpolated Grid')
            axes[0, 1].set_xlabel('Longitude')
            axes[0, 1].set_ylabel('Latitude')
            axes[0, 1].set_aspect('equal')
            plt.colorbar(im, ax=axes[0, 1])
            
            # 3. Residual histogram
            axes[0, 2].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
            axes[0, 2].set_title('Residual Distribution')
            axes[0, 2].set_xlabel('Residual (nT)')
            axes[0, 2].set_ylabel('Frequency')
            axes[0, 2].grid(True, alpha=0.3)
            
            # 4. Residual spatial distribution
            valid_x = x_data[valid_interp_mask]
            valid_y = y_data[valid_interp_mask]
            scatter = axes[1, 0].scatter(valid_x, valid_y, c=residuals, cmap='RdBu_r', s=2, alpha=0.7)
            axes[1, 0].set_title('Spatial Distribution of Residuals')
            axes[1, 0].set_xlabel('Longitude')
            axes[1, 0].set_ylabel('Latitude')
            axes[1, 0].set_aspect('equal')
            plt.colorbar(scatter, ax=axes[1, 0], label='Residual (nT)')
            
            # 5. Predicted vs Observed
            predicted = grid_at_data[valid_interp_mask]
            observed = z_data[valid_interp_mask]
            axes[1, 1].scatter(predicted, observed, alpha=0.5, s=1)
            min_val = min(np.min(predicted), np.min(observed))
            max_val = max(np.max(predicted), np.max(observed))
            axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 Line')
            axes[1, 1].set_xlabel('Predicted (nT)')
            axes[1, 1].set_ylabel('Observed (nT)')
            axes[1, 1].set_title('Predicted vs Observed')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            # 6. Statistics
            axes[1, 2].axis('off')
            rmse = np.sqrt(np.mean(residuals**2))
            mae = np.mean(np.abs(residuals))
            r_squared = np.corrcoef(predicted, observed)[0, 1]**2 if len(predicted) > 1 else 0
            
            stats_text = f"""
            Interpolation Statistics:
            
            RMSE: {rmse:.2f} nT
            MAE: {mae:.2f} nT
            R²: {r_squared:.3f}
            
            Mean Residual: {np.mean(residuals):.2f} nT
            Std Residual: {np.std(residuals):.2f} nT
            
            Valid Points: {len(residuals):,}
            Grid Size: {grid_z.shape[0]}×{grid_z.shape[1]}
            """
            
            axes[1, 2].text(0.1, 0.9, stats_text, transform=axes[1, 2].transAxes, 
                           fontsize=12, verticalalignment='top', 
                           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(diagnostic_dir / f'interpolation_diagnostics_{field_column.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception:
            pass
    
    def create_data_summary(self, data, grid_z, field_column, output_dir):
        """Create data summary report"""
        try:
            summary_text = f"""
Grid Interpolation Summary Report
================================

Input Data:
- Total points: {len(data):,}
- Valid points: {len(data.dropna()):,}
- Field processed: {field_column}

Data Range:
- Minimum: {data[field_column].min():.2f} nT
- Maximum: {data[field_column].max():.2f} nT
- Mean: {data[field_column].mean():.2f} nT
- Standard deviation: {data[field_column].std():.2f} nT

Grid Information:
- Grid size: {grid_z.shape[0]} × {grid_z.shape[1]}
- Valid grid points: {np.sum(~np.isnan(grid_z)):,}
- Grid coverage: {(np.sum(~np.isnan(grid_z)) / grid_z.size * 100):.1f}%

Interpolation Method:
- Algorithm: Minimum Curvature
- JIT acceleration: {"Yes" if NUMBA_AVAILABLE else "No"}

Processing Information:
- Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- Software: UXO Wizard Grid Interpolator
"""
            
            summary_path = output_dir / 'interpolation_summary.txt'
            with open(summary_path, 'w') as f:
                f.write(summary_text)
                
        except Exception:
            pass
    
    def save_processed_grid(self, grid_X, grid_Y, grid_z, field_column, output_dir):
        """Save processed grid as CSV"""
        try:
            # Create grid DataFrame
            grid_df = pd.DataFrame({
                'longitude': grid_X.ravel(),
                'latitude': grid_Y.ravel(),
                'value': grid_z.ravel()
            })
            
            # Remove NaN values
            grid_df = grid_df.dropna()
            
            # Save to CSV
            grid_csv_path = output_dir / f'grid_minimum_curvature_{field_column.replace(" ", "_").replace("[", "").replace("]", "")}.csv'
            grid_df.to_csv(grid_csv_path, index=False)
            
        except Exception:
            pass
    
    def create_interactive_plot(self, x, y, z, grid_X, grid_Y, grid_z, field_column, plot_type='2D'):
        """Create an interactive plot for the data viewer"""
        try:
            if plot_type == '3D':
                # Create 3D surface plot
                fig = plt.figure(figsize=(12, 9))
                ax = fig.add_subplot(111, projection='3d')
                
                # Create 3D surface plot
                surf = ax.plot_surface(grid_X, grid_Y, grid_z, cmap='RdYlBu_r', alpha=0.7)
                
                # Add original data points as scatter
                ax.scatter(x, y, z, c='red', s=1, alpha=0.6)
                
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')
                ax.set_zlabel(f'{field_column} [nT]')
                ax.set_title(f'3D Grid Interpolation - {field_column}')
                
                # Add colorbar
                fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
                
            else:
                # Create 2D contour plot
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                fig.suptitle(f'Grid Interpolation Results - {field_column}', fontsize=16)
                
                # Plot 1: Original data
                im1 = axes[0, 0].scatter(x, y, c=z, cmap='RdYlBu_r', s=1, alpha=0.7)
                axes[0, 0].set_title('Original Data Points')
                axes[0, 0].set_xlabel('Longitude')
                axes[0, 0].set_ylabel('Latitude')
                axes[0, 0].set_aspect('equal')
                plt.colorbar(im1, ax=axes[0, 0])
                
                # Plot 2: Interpolated field
                im2 = axes[0, 1].contourf(grid_X, grid_Y, grid_z, levels=50, cmap='RdYlBu_r')
                axes[0, 1].set_title('Interpolated Field')
                axes[0, 1].set_xlabel('Longitude')
                axes[0, 1].set_ylabel('Latitude')
                axes[0, 1].set_aspect('equal')
                plt.colorbar(im2, ax=axes[0, 1])
                
                # Plot 3: Contour lines
                axes[1, 0].contour(grid_X, grid_Y, grid_z, levels=20, colors='black', linewidths=0.5)
                axes[1, 0].set_title('Contour Lines')
                axes[1, 0].set_xlabel('Longitude')
                axes[1, 0].set_ylabel('Latitude')
                axes[1, 0].set_aspect('equal')
                
                # Plot 4: Combined view
                axes[1, 1].contourf(grid_X, grid_Y, grid_z, levels=50, cmap='RdYlBu_r', alpha=0.7)
                axes[1, 1].scatter(x, y, c=z, cmap='RdYlBu_r', s=0.5, alpha=0.8)
                axes[1, 1].set_title('Combined View')
                axes[1, 1].set_xlabel('Longitude')
                axes[1, 1].set_ylabel('Latitude')
                axes[1, 1].set_aspect('equal')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            raise ProcessingError(f"Error creating interactive plot: {e}")
    
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
        result = ProcessingResult(success=True, processing_script=self.name)
        
        try:
            # Validate input data
            if progress_callback:
                progress_callback(0.02, "Validating input data...")
            
            self.validate_data(data)
            
            if progress_callback:
                progress_callback(0.04, "Extracting processing parameters...")
            
            # Get processing parameters
            interp_params = params.get('interpolation_parameters', {})
            boundary_params = params.get('boundary_parameters', {})
            output_params = params.get('output_parameters', {})
            
            grid_resolution = interp_params.get('grid_resolution', {}).get('value', 300)
            max_iterations = interp_params.get('max_iterations', {}).get('value', 1000)
            tolerance = interp_params.get('tolerance', {}).get('value', 1e-6)
            omega = interp_params.get('relaxation_factor', {}).get('value', 1.8)
            constraint_mode = interp_params.get('constraint_mode', {}).get('value', 'soft')
            
            enable_masking = boundary_params.get('enable_boundary_masking', {}).get('value', True)
            boundary_method = boundary_params.get('boundary_method', {}).get('value', 'convex_hull')
            buffer_distance = boundary_params.get('buffer_distance', {}).get('value', 0.0)
            
            target_field = output_params.get('magnetic_field_column', {}).get('value', 'auto')
            generate_diagnostics = output_params.get('generate_diagnostics', {}).get('value', True)
            save_grid_data = output_params.get('save_grid_data', {}).get('value', True)
            include_original_points = output_params.get('include_original_points', {}).get('value', False)
            generate_interactive_plot = output_params.get('generate_interactive_plot', {}).get('value', True)
            plot_type = output_params.get('plot_type', {}).get('value', '2D')
            
            if progress_callback:
                progress_callback(0.06, "Detecting magnetic field column...")
            
            # Get magnetic field column
            field_column = self.get_magnetic_field_column(data, target_field)
            
            if progress_callback:
                progress_callback(0.08, f"Processing {len(data)} data points using field: {field_column}")
            
            # Extract coordinates and values
            if progress_callback:
                progress_callback(0.10, "Extracting coordinates and field values...")
            
            x = data['Longitude [Decimal Degrees]'].values
            y = data['Latitude [Decimal Degrees]'].values
            z = data[field_column].values
            
            if progress_callback:
                progress_callback(0.12, "Cleaning data (removing NaN values)...")
            
            # Remove NaN values
            valid_mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
            x, y, z = x[valid_mask], y[valid_mask], z[valid_mask]
            
            if len(x) == 0:
                raise ProcessingError("No valid data points after removing NaN values")
            
            if progress_callback:
                progress_callback(0.14, f"Data cleaned: {len(x)} valid points remaining")
            
            # Perform minimum curvature interpolation using enhanced base class method
            if progress_callback:
                progress_callback(0.16, f"Starting minimum curvature interpolation ({grid_resolution}×{grid_resolution} grid, {constraint_mode} constraints)...")
            
            grid_X, grid_Y, grid_z = self.minimum_curvature_interpolation_enhanced(
                x, y, z, grid_resolution, max_iterations, tolerance, omega, constraint_mode, progress_callback
            )
            
            # Apply boundary masking if enabled
            if enable_masking:
                if progress_callback:
                    progress_callback(0.85, f"Applying boundary masking ({boundary_method})...")
                
                mask = self.create_boundary_mask_with_buffer(x, y, grid_X, grid_Y, boundary_method, buffer_distance)
                grid_z = np.where(mask, grid_z, np.nan)
                
                if progress_callback:
                    progress_callback(0.87, "Boundary masking complete")
            
            # Create temporary directory for outputs
            if progress_callback:
                progress_callback(0.88, "Preparing output generation...")
            temp_dir = Path(tempfile.mkdtemp())
            
            # Generate diagnostic plot if requested
            if generate_diagnostics:
                if progress_callback:
                    progress_callback(0.90, "Generating diagnostic plots...")
                
                diagnostic_path = self.create_diagnostic_plot(
                    x, y, z, grid_X, grid_Y, grid_z, field_column, temp_dir
                )
                result.add_output_file(str(diagnostic_path), 'png', 'Interpolation diagnostic plots')
                
                if progress_callback:
                    progress_callback(0.92, "Diagnostic plots complete")
            
            # Generate interactive plot if requested
            if generate_interactive_plot:
                if progress_callback:
                    progress_callback(0.925, f"Generating interactive {plot_type} plot...")
                
                interactive_figure = self.create_interactive_plot(
                    x, y, z, grid_X, grid_Y, grid_z, field_column, plot_type
                )
                if interactive_figure:
                    result.figure = interactive_figure
                    if progress_callback:
                        progress_callback(0.93, f"Interactive {plot_type} plot generated")
            
            # Save grid data if requested
            if save_grid_data:
                if progress_callback:
                    progress_callback(0.93, "Saving grid data to CSV...")
                
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
                
                if progress_callback:
                    progress_callback(0.94, "Grid data saved to CSV")
            
            # Create raster layer for visualization
            if progress_callback:
                progress_callback(0.95, "Creating interpolated field raster layer...")
            
            # Calculate bounds for raster layer [min_x, min_y, max_x, max_y]
            bounds = [
                float(np.nanmin(grid_X)),  # west (min_x)
                float(np.nanmin(grid_Y)),  # south (min_y)
                float(np.nanmax(grid_X)),  # east (max_x)
                float(np.nanmax(grid_Y))   # north (max_y)
            ]
            
            # Create unique layer name based on input filename and timestamp
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            if input_file_path:
                input_filename = Path(input_file_path).stem
                raster_layer_name = f'{input_filename} - Interpolated {field_column} ({timestamp})'
            else:
                raster_layer_name = f'Interpolated {field_column} ({timestamp})'
            
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
                    'layer_name': raster_layer_name,
                    'bounds': bounds  # Add bounds to metadata to help with layer creation
                }
            )
            
            # Create contour layer as a second raster
            if progress_callback:
                progress_callback(0.96, "Creating contour raster layer...")
            
            contour_data = self.create_contour_raster(grid_X, grid_Y, grid_z, bounds)
            if contour_data is not None:
                # Create unique contour layer name based on input filename
                if input_file_path:
                    contour_layer_name = f'{input_filename} - Contours {field_column} ({timestamp})'
                else:
                    contour_layer_name = f'Contours {field_column} ({timestamp})'
                    
                result.add_layer_output(
                    layer_type='raster',
                    data=contour_data,
                    style_info={},
                    metadata={
                        'layer_name': contour_layer_name,
                        'bounds': bounds
                    }
                )
                if progress_callback:
                    progress_callback(0.97, "Contour layer created successfully")
            else:
                if progress_callback:
                    progress_callback(0.97, "Failed to create contour layer")
            
            # Generate comprehensive analysis in /grid_analysis directory
            if progress_callback:
                progress_callback(0.98, "Generating comprehensive analysis (field plots, diagnostics, summary)...")
            
            analysis_dir = self.create_comprehensive_analysis(
                data, grid_X, grid_Y, grid_z, field_column, input_file_path, temp_dir
            )
            
            if analysis_dir:
                # Add analysis directory to outputs
                result.add_output_file(str(analysis_dir), 'directory', 'Comprehensive grid analysis')
                if progress_callback:
                    progress_callback(0.985, "Comprehensive analysis complete")
            
            # Add downsampled original data points as point layer (if enabled)
            if include_original_points:
                if progress_callback:
                    progress_callback(0.99, "Creating original data points layer...")
                
                point_data = data[['Latitude [Decimal Degrees]', 'Longitude [Decimal Degrees]', field_column]].copy()
            
                # Smart downsampling for map performance (preserves flight line structure)
                max_points = 5000
                if len(point_data) > max_points:
                    # Use systematic sampling to preserve flight line structure
                    step = len(point_data) // max_points
                    if step > 1:
                        downsample_indices = np.arange(0, len(point_data), step)
                        point_data = point_data.iloc[downsample_indices].copy()
                        
                        # Create unique point layer name based on input filename
                        if input_file_path:
                            point_layer_name = f'{input_filename} - Original Data Points (downsampled every {step} points) ({timestamp})'
                        else:
                            point_layer_name = f'Original Data Points (downsampled every {step} points) ({timestamp})'
                            
                        if progress_callback:
                            progress_callback(0.992, f"Downsampled {len(data)} points to {len(point_data)} for map display")
                    else:
                        # Create unique point layer name based on input filename
                        if input_file_path:
                            point_layer_name = f'{input_filename} - Original Data Points ({timestamp})'
                        else:
                            point_layer_name = f'Original Data Points ({timestamp})'
                else:
                    # Create unique point layer name based on input filename
                    if input_file_path:
                        point_layer_name = f'{input_filename} - Original Data Points ({timestamp})'
                    else:
                        point_layer_name = f'Original Data Points ({timestamp})'
                
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
                        'layer_name': point_layer_name,
                        'color_field': 'value',
                        'use_graduated_colors': True,
                        'point_size': 2,
                        'point_opacity': 0.8,
                        'enable_clustering': False
                    }
                )
            
            if progress_callback:
                progress_callback(0.995, "Finalizing processing results...")
                
            if progress_callback:
                layers_created = 2 if contour_data is not None else 1
                progress_callback(1.0, f"Grid interpolation complete! Created {layers_created} raster layer(s) + 1 point layer")
            
            # Add processing summary
            result.metadata.update({
                'processor': 'magnetic',
                'processing_method': 'minimum_curvature',
                'constraint_mode': constraint_mode,
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