"""
Magnetic Anomaly Processor Script for UXO Wizard Framework
===========================================================

Applies a suite of advanced geophysical processing techniques to a 2D interpolated
magnetic grid. This script is designed to work on the output of the 
Grid Interpolator script.

Features:
- Takes a 2D grid as input.
- Applies one of several advanced analysis techniques:
  - Analytic Signal: Highlights anomaly locations.
  - Tilt Derivative: Delineates the edges of magnetic sources.
  - Horizontal Gradient: Also highlights edges.
  - Euler Deconvolution: Estimates source location and depth.
- Includes an 'all_methods' option to run all analyses simultaneously,
  generating a comprehensive 2x2 comparison plot and separate map layers for
  each result.
- Generates interactive plots and map layers for seamless integration into
  the UXO Wizard UI.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Optional, Callable
import datetime

# Import required components from the framework
from src.processing.base import ScriptInterface, ProcessingResult, ProcessingError

# Import scientific libraries for calculations
try:
    from scipy.ndimage import gaussian_filter
    from scipy.fft import fft2, ifft2, fftfreq
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class MagneticAnomalyProcessor(ScriptInterface):
    """
    Applies advanced filters and models to a 2D interpolated magnetic grid.
    """

    @property
    def name(self) -> str:
        return "Magnetic Anomaly Processor"

    @property
    def description(self) -> str:
        return "Apply advanced filters (Analytic Signal, Tilt Derivative, Euler) to a 2D interpolated magnetic grid. Includes 'all_methods' for a comprehensive comparison."

    def get_parameters(self) -> Dict[str, Any]:
        """Return parameter structure for the processing script."""
        return {
            'analysis_options': {
                'analysis_type': {
                    'value': 'all_methods',
                    'type': 'choice',
                    'choices': [
                        'all_methods',
                        'analytic_signal',
                        'tilt_derivative',
                        'euler_deconvolution',
                        'horizontal_gradient'
                    ],
                    'description': 'Select the processing method. "all_methods" runs all analyses and creates a comparison plot.'
                }
            },
            'input_data': {
                'grid_value_column': {
                    'value': 'value',
                    'type': 'choice',
                    'choices': ['value', 'R1 [nT]', 'R2 [nT]'],
                    'description': 'Column from the input grid data to process.'
                },
                'vertical_gradient_column': {
                    'value': 'none',
                    'type': 'choice',
                    'choices': ['none', 'VA'],
                    'description': "Optional: Specify a column with vertical gradient data (e.g., 'VA'). Choose 'none' to calculate it."
                }
            },
            'euler_parameters': {
                'structural_index': {
                    'value': 3,
                    'type': 'int',
                    'min': 0,
                    'max': 3,
                    'description': 'Structural Index (SI) for the source type. Use 3 for spheres/dipoles (UXO).'
                },
                'window_size': {
                    'value': 10,
                    'type': 'int',
                    'min': 3,
                    'max': 50,
                    'description': 'Size of the moving window (in grid points) for Euler calculations.'
                }
            },
            'advanced_debugging': {
                'enable_euler_debugging': {
                    'value': False,
                    'type': 'bool',
                    'description': 'Enable detailed console logging to debug the Euler Deconvolution solver.'
                }
            },
            'output_options': {
                'layer_name_suffix': {
                    'value': '',
                    'type': 'str',
                    'description': 'Optional suffix to add to the output layer names.'
                }
            }
        }

    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate that the input data is a grid."""
        if not SCIPY_AVAILABLE:
            raise ProcessingError("Scipy library not found. Please install it: pip install scipy")
            
        required_cols = ['longitude', 'latitude', 'value']
        if not all(col in data.columns for col in required_cols):
            raise ProcessingError("Input data does not appear to be a valid grid. Please run the Grid Interpolator first. Expected columns: 'longitude', 'latitude', 'value'.")
        if len(data) < 100:
            raise ProcessingError("Input grid data has fewer than 100 points. Please ensure the input is a valid grid.")
        return True

    def _calculate_derivatives(self, grid_data, dx, dy):
        """Helper to calculate spatial derivatives."""
        # Calculate horizontal derivatives using numpy's gradient
        dM_dy, dM_dx = np.gradient(grid_data, dy, dx)

        # Calculate vertical derivative using Fourier transform
        # This is a standard method in potential field geophysics
        ny, nx = grid_data.shape
        kx = 2 * np.pi * fftfreq(nx, d=dx)
        ky = 2 * np.pi * fftfreq(ny, d=dy)
        Ky, Kx = np.meshgrid(ky, kx, indexing='ij')
        
        # Wavenumber magnitude
        K = np.sqrt(Kx**2 + Ky**2)
        K[0, 0] = 1e-10 # Avoid division by zero

        # FFT of the grid
        M_fft = fft2(grid_data)
        
        # The vertical derivative in the Fourier domain is multiplication by |k|
        dM_dz_fft = K * M_fft
        
        # Inverse FFT to get the spatial domain derivative
        dM_dz = np.real(ifft2(dM_dz_fft))
        
        return dM_dx, dM_dy, dM_dz

    def _run_analytic_signal(self, dM_dx, dM_dy, dM_dz):
        """Calculates the analytic signal magnitude."""
        return np.sqrt(dM_dx**2 + dM_dy**2 + dM_dz**2)

    def _run_tilt_derivative(self, dM_dx, dM_dy, dM_dz):
        """Calculates the tilt derivative."""
        hg = np.sqrt(dM_dx**2 + dM_dy**2)
        # Add a small epsilon to avoid division by zero
        return np.arctan2(dM_dz, hg + 1e-10)

    def _run_horizontal_gradient(self, dM_dx, dM_dy):
        """Calculates the total horizontal gradient magnitude."""
        return np.sqrt(dM_dx**2 + dM_dy**2)
        
    def _run_euler_deconvolution(self, grid_data, dM_dx, dM_dy, dM_dz, grid_X, grid_Y, si, window_size, debug_mode=False):
        """Performs Euler Deconvolution with robust checks and debugging."""
        ny, nx = grid_data.shape
        solutions = []
        
        flat_tolerance = 1e-9

        for i in range(0, ny - window_size, 2):
            for j in range(0, nx - window_size, 2):
                # Extract window
                w_M = grid_data[i:i+window_size, j:j+window_size]
                w_dM_dx = dM_dx[i:i+window_size, j:j+window_size]
                w_dM_dy = dM_dy[i:i+window_size, j:j+window_size]
                w_dM_dz = dM_dz[i:i+window_size, j:j+window_size]
                w_X = grid_X[i:i+window_size, j:j+window_size]
                w_Y = grid_Y[i:i+window_size, j:j+window_size]

                # --- NEW, MORE ROBUST CHECKS ---
                # 1. Check for NaN or Inf values in the window. This is the most likely culprit.
                if not np.all(np.isfinite(w_M)) or \
                   not np.all(np.isfinite(w_dM_dx)) or \
                   not np.all(np.isfinite(w_dM_dy)):
                    if debug_mode:
                        print(f"DEBUG (Euler): Skipping window at (i={i}, j={j}) due to NaN/Inf values.")
                    continue

                # 2. Check if gradients are flat.
                if np.std(w_dM_dx) < flat_tolerance and np.std(w_dM_dy) < flat_tolerance:
                    if debug_mode:
                        print(f"DEBUG (Euler): Skipping window at (i={i}, j={j}) due to flat gradients.")
                    continue

                # Ravel the data *after* checks are complete
                w_M, w_dM_dx, w_dM_dy, w_dM_dz, w_X, w_Y = (
                    w_M.ravel(), w_dM_dx.ravel(), w_dM_dy.ravel(), 
                    w_dM_dz.ravel(), w_X.ravel(), w_Y.ravel()
                )

                # Set up the linear system A*m = b
                A = np.array([w_dM_dx, w_dM_dy, w_dM_dz, np.full_like(w_M, -si)]).T
                b = (w_X * w_dM_dx + w_Y * w_dM_dy - si * w_M)
                
                try:
                    m, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    
                    if np.isfinite(m).all() and 0 < m[2] < 100:
                        solutions.append({'longitude': m[0], 'latitude': m[1], 'depth': m[2]})
                except np.linalg.LinAlgError as e:
                    if debug_mode:
                        print(f"DEBUG (Euler): LinAlgError at (i={i}, j={j}). Skipping. Error: {e}")
                    continue

        return pd.DataFrame(solutions)

    def _plot_grid(self, ax, grid_X, grid_Y, output_grid, title, cbar_label, contour_grid=None):
        """Helper to plot a 2D grid."""
        im = ax.pcolormesh(grid_X, grid_Y, output_grid, cmap='RdYlBu_r', shading='auto')
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Longitude', fontsize=8)
        ax.set_ylabel('Latitude', fontsize=8)
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, label=cbar_label)

        if contour_grid is not None:
             # Highlight the zero contour for Tilt Derivative
             ax.contour(grid_X, grid_Y, contour_grid, levels=[0], colors='black', linewidths=1.5)

    def _plot_euler_solutions(self, ax, solutions_df, grid_X, grid_Y, title):
        """Helper to plot Euler solutions."""
        if not solutions_df.empty:
            sc = ax.scatter(solutions_df['longitude'], solutions_df['latitude'], c=solutions_df['depth'], cmap='viridis_r', s=10, alpha=0.7)
            plt.colorbar(sc, ax=ax, label='Estimated Depth (m)')
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Longitude', fontsize=8)
        ax.set_ylabel('Latitude', fontsize=8)
        ax.set_xlim(grid_X.min(), grid_X.max())
        ax.set_ylim(grid_Y.min(), grid_Y.max())
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.5)

    def execute(self, data: pd.DataFrame, params: Dict[str, Any],
                progress_callback: Optional[Callable] = None, input_file_path: Optional[str] = None) -> ProcessingResult:
        
        result = ProcessingResult(success=True)
        debug_mode = params.get('advanced_debugging', {}).get('enable_euler_debugging', {}).get('value', False)
        try:
            # --- 1. Initialization and Parameter Extraction ---
            if progress_callback: progress_callback(0, "Initializing Anomaly Processor...")
            
            # Safe parameter extraction
            analysis_type = params.get('analysis_options', {}).get('analysis_type', {}).get('value', 'all_methods')
            grid_value_col = params.get('input_data', {}).get('grid_value_column', {}).get('value', 'value')
            si = params.get('euler_parameters', {}).get('structural_index', {}).get('value', 3)
            window_size = params.get('euler_parameters', {}).get('window_size', {}).get('value', 10)
            suffix = params.get('output_options', {}).get('layer_name_suffix', {}).get('value', '')
            
            df = data.copy()

            # --- 2. Data Preparation: Reshape to 2D Grid ---
            if progress_callback: progress_callback(0.1, "Reshaping data to 2D grid...")
            
            try:
                grid_pivot = df.pivot_table(index='latitude', columns='longitude', values=grid_value_col)
            except Exception as e:
                raise ProcessingError(f"Failed to pivot data into a grid. Ensure the input is from the Grid Interpolator. Error: {e}")

            grid_data = grid_pivot.values
            lat_coords = grid_pivot.index.values
            lon_coords = grid_pivot.columns.values
            grid_X, grid_Y = np.meshgrid(lon_coords, lat_coords)

            # --- 3. Derivative Calculation ---
            if progress_callback: progress_callback(0.2, "Calculating spatial derivatives...")
            dx = np.abs(lon_coords[1] - lon_coords[0])
            dy = np.abs(lat_coords[1] - lat_coords[0])
            dM_dx, dM_dy, dM_dz = self._calculate_derivatives(grid_data, dx, dy)
            
            # --- 4. Analysis Execution ---
            single_run = analysis_type != 'all_methods'
            fig = None

            if single_run:
                fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            else: # 'all_methods'
                fig, axes = plt.subplots(2, 2, figsize=(15, 15))
                fig.suptitle('Comprehensive Anomaly Analysis', fontsize=16)

            # --- Function to create map layers ---
            def create_map_layer(layer_type, data, name, style_info={}):
                timestamp = datetime.datetime.now().strftime("%H%M%S")
                base_name = Path(input_file_path).stem if input_file_path else "analysis"
                layer_name = f"{base_name} - {name}{' ' + suffix if suffix else ''} ({timestamp})"
                
                metadata = {
                    'layer_name': layer_name,
                    'bounds': [lon_coords.min(), lat_coords.min(), lon_coords.max(), lat_coords.max()]
                }
                result.add_layer_output(layer_type=layer_type, data=data, style_info=style_info, metadata=metadata)

            # --- Run Analyses ---
            if analysis_type in ['analytic_signal', 'all_methods']:
                if progress_callback: progress_callback(0.3, "Running Analytic Signal...")
                output_grid = self._run_analytic_signal(dM_dx, dM_dy, dM_dz)
                ax = axes[0, 0] if not single_run else ax
                self._plot_grid(ax, grid_X, grid_Y, output_grid, "Analytic Signal", "Amplitude (nT/m)")
                create_map_layer('raster', {'grid': output_grid, 'bounds': [lon_coords.min(), lat_coords.min(), lon_coords.max(), lat_coords.max()]}, 'Analytic Signal')

            if analysis_type in ['tilt_derivative', 'all_methods']:
                if progress_callback: progress_callback(0.45, "Running Tilt Derivative...")
                output_grid = self._run_tilt_derivative(dM_dx, dM_dy, dM_dz)
                ax = axes[0, 1] if not single_run else ax
                self._plot_grid(ax, grid_X, grid_Y, np.degrees(output_grid), "Tilt Derivative", "Angle (degrees)", contour_grid=output_grid)
                create_map_layer('raster', {'grid': output_grid, 'bounds': [lon_coords.min(), lat_coords.min(), lon_coords.max(), lat_coords.max()]}, 'Tilt Derivative')
                
            if analysis_type in ['horizontal_gradient', 'all_methods']:
                if progress_callback: progress_callback(0.6, "Running Horizontal Gradient...")
                output_grid = self._run_horizontal_gradient(dM_dx, dM_dy)
                ax = axes[1, 0] if not single_run else ax
                self._plot_grid(ax, grid_X, grid_Y, output_grid, "Horizontal Gradient", "Amplitude (nT/m)")
                create_map_layer('raster', {'grid': output_grid, 'bounds': [lon_coords.min(), lat_coords.min(), lon_coords.max(), lat_coords.max()]}, 'Horizontal Gradient')
                
            if analysis_type in ['euler_deconvolution', 'all_methods']:
                if progress_callback: progress_callback(0.75, "Running Euler Deconvolution...")
                solutions_df = self._run_euler_deconvolution(grid_data, dM_dx, dM_dy, dM_dz, grid_X, grid_Y, si, window_size, debug_mode=debug_mode)
                ax = axes[1, 1] if not single_run else ax
                self._plot_euler_solutions(ax, solutions_df, grid_X, grid_Y, f"Euler Solutions (SI={si})")
                if not solutions_df.empty:
                    create_map_layer('point', solutions_df, f'Euler SI{si} Solutions')

            # --- 5. Finalize Result ---
            if progress_callback: progress_callback(0.95, "Finalizing results...")
            
            fig.tight_layout(rect=[0, 0.03, 1, 0.95] if not single_run else None)
            result.figure = fig
            
            result.metadata.update({
                'processor': 'magnetic', # CRITICAL for file organization
                'analysis_method': analysis_type,
                'grid_shape': f"{grid_data.shape[0]}x{grid_data.shape[1]}",
                'euler_si_used': si if 'euler' in analysis_type else 'N/A'
            })
            
            if progress_callback: progress_callback(1.0, "Processing complete!")

            return result

        except Exception as e:
            raise ProcessingError(f"Magnetic Anomaly Processing failed: {str(e)}")


# IMPORTANT: Export the class for discovery by the framework
SCRIPT_CLASS = MagneticAnomalyProcessor