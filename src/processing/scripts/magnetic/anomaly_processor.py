"""
Magnetic Anomaly Processor Script for UXO Wizard Framework
===========================================================

Applies a suite of advanced geophysical processing techniques to a 2D interpolated
magnetic grid. This script is designed to work on the output of the 
Grid Interpolator script.

V10 - DEFINITIVE VERSION based on user-provided framework example.

Features:
- Adheres strictly to the UXO Wizard framework for all file output.
- Creates a dedicated output folder for each run.
- Saves all plots as separate, high-quality PNG files.
- Includes a comprehensive statistics report text file.
- All outputs are co-located in the same analysis folder.
- Applies intelligent interpolation and smoothing to prevent numerical errors.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Optional, Callable
import datetime
import tempfile

from src.processing.base import ScriptInterface, ProcessingResult, ProcessingError
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.fft import fft2, ifft2, fftfreq
from scipy.interpolate import griddata


class MagneticAnomalyProcessor(ScriptInterface):
    def __init__(self, project_manager=None):
        super().__init__(project_manager)
    
    @property
    def name(self) -> str: return "Magnetic Anomaly Processor"
    @property
    def description(self) -> str: return "Apply advanced filters (Analytic Signal, Tilt, Euler, etc.) to a 2D grid. Saves all plots and reports to a dedicated folder."

    def get_parameters(self) -> Dict[str, Any]:
        return {
            'analysis_options': {'analysis_type': {'value': 'all_methods', 'type': 'choice', 'choices': ['all_methods','analytic_signal','tilt_derivative','euler_deconvolution','horizontal_gradient','blakely_peaks'], 'description': 'Select the processing method. "all_methods" runs all analyses.'}},
            'input_data': {
                'grid_value_column': { 'value': 'value', 'type': 'choice', 'choices': ['value', 'R1 [nT]', 'R2 [nT]'], 'description': 'Primary magnetic field column to process.'},
                'vertical_gradient_column': {'value': 'none', 'type': 'choice', 'choices': ['none', 'VA'], 'description': "Use a measured vertical gradient column ('VA') for higher accuracy."},
                'gaussian_sigma': {'value': 1.0, 'type': 'float', 'min': 0.0, 'max': 5.0, 'description': 'Sigma for Gaussian smoothing filter. Helps stabilize derivatives.'}
            },
            'euler_parameters': {
                'structural_index': {'value': 3, 'type': 'int', 'min': 0, 'max': 3, 'description': 'Structural Index (SI) for the source type. Use 3 for UXO.'},
                'window_size': {'value': 10, 'type': 'int', 'min': 3, 'max': 50, 'description': 'Size of the moving window (in grid points) for Euler calculations.'}
            },
            'output_options': {
                'generate_statistics_report': {'value': True, 'type': 'bool', 'description': 'Generate a text file with detailed statistics for all calculated grids.'},
                'generate_plots': {'value': True, 'type': 'bool', 'description': 'Generate and save a separate PNG image for each analysis.'},
                'layer_name_suffix': {'value': '', 'type': 'str', 'description': 'Optional suffix to add to the output layer names.'}
            }
        }

    def validate_data(self, data: pd.DataFrame) -> bool:
        # Check for coordinate columns
        has_utm = 'UTM_Easting' in data.columns and 'UTM_Northing' in data.columns
        has_latlon = 'longitude' in data.columns and 'latitude' in data.columns
        
        if not has_utm and not has_latlon:
            raise ProcessingError("Input data must have either UTM coordinates (UTM_Easting, UTM_Northing) or geographic coordinates (longitude, latitude).")
        
        if 'value' not in data.columns:
            raise ProcessingError("Input data must have a 'value' column. Please run the Grid Interpolator first.")
        
        return True

    def _create_output_directory(self, input_file_path: Optional[str], working_directory: Optional[str] = None) -> Path:
        """Create output directory in project/processed/magnetic/ following framework structure"""
        
        print(f"DEBUG: _create_output_directory called with:")
        print(f"  input_file_path: {input_file_path}")
        print(f"  working_directory: {working_directory}")
        
        # Always prefer working directory if provided
        if working_directory and working_directory.strip():
            project_dir = Path(working_directory)
            base_filename = Path(input_file_path).stem if input_file_path else "anomaly_analysis"
            output_dir = project_dir / "processed" / "magnetic" / f"{base_filename}_analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            print(f"DEBUG: Using working directory - final output_dir: {output_dir}")
        elif input_file_path:
            # Fallback method: try to find project root from input file path
            input_path = Path(input_file_path)
            base_filename = input_path.stem
            
            # Find project root directory
            project_dir = input_path.parent
            while project_dir != project_dir.parent:
                if (project_dir / "processed").exists() or len(list(project_dir.glob("*.uxo"))) > 0:
                    break
                project_dir = project_dir.parent
            
            # Create project/processed/magnetic/filename_analysis_timestamp structure
            output_dir = project_dir / "processed" / "magnetic" / f"{base_filename}_analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            # Last resort: temporary directory
            temp_dir = tempfile.mkdtemp(prefix="anomaly_analysis_")
            output_dir = Path(temp_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _calculate_derivatives(self, grid_data_no_nans, dx, dy, nan_mask, va_grid_no_nans=None):
        dM_dy, dM_dx = np.gradient(grid_data_no_nans, dy, dx)
        if va_grid_no_nans is not None: 
            dM_dz = va_grid_no_nans
        else:
            ny, nx = grid_data_no_nans.shape
            kx, ky = 2 * np.pi * fftfreq(nx, d=dx), 2 * np.pi * fftfreq(ny, d=dy)
            Ky, Kx = np.meshgrid(ky, kx, indexing='ij')
            K = np.sqrt(Kx**2 + Ky**2); K[0, 0] = 1e-10
            dM_dz = np.real(ifft2(K * fft2(grid_data_no_nans)))
        dM_dx[nan_mask], dM_dy[nan_mask], dM_dz[nan_mask] = np.nan, np.nan, np.nan
        return dM_dx, dM_dy, dM_dz

    def _run_analytic_signal(self, dM_dx, dM_dy, dM_dz): return np.sqrt(dM_dx**2 + dM_dy**2 + dM_dz**2)
    def _run_tilt_derivative(self, dM_dx, dM_dy, dM_dz): return np.arctan2(dM_dz, np.sqrt(dM_dx**2 + dM_dy**2) + 1e-10)
    def _run_horizontal_gradient(self, dM_dx, dM_dy): return np.sqrt(dM_dx**2 + dM_dy**2)
    def _run_blakely_peaks(self, hg, grid_X, grid_Y, coord_type='LatLon'):
        local_max = maximum_filter(hg, size=3) == hg
        peak_y, peak_x = np.where(local_max & (hg > np.nanmean(hg)))
        
        return pd.DataFrame({'longitude': grid_X[peak_y, peak_x], 'latitude': grid_Y[peak_y, peak_x]})

    def _run_euler_deconvolution(self, grid_data, dM_dx, dM_dy, dM_dz, grid_X, grid_Y, si, window_size, coord_type='LatLon'):
        ny, nx = grid_data.shape; solutions = []
        for i in range(0, ny - window_size, 2):
            for j in range(0, nx - window_size, 2):
                w_M = grid_data[i:i+window_size, j:j+window_size]
                if not np.all(np.isfinite(w_M)): continue
                w_dM_dx, w_dM_dy, w_dM_dz = dM_dx[i:i+window_size, j:j+window_size], dM_dy[i:i+window_size, j:j+window_size], dM_dz[i:i+window_size, j:j+window_size]
                w_X_flat, w_Y_flat = grid_X[i:i+window_size, j:j+window_size].ravel(), grid_Y[i:i+window_size, j:j+window_size].ravel()
                A = np.array([w_dM_dx.ravel(), w_dM_dy.ravel(), w_dM_dz.ravel(), np.full_like(w_M.ravel(), -si)]).T
                b = (w_X_flat * w_dM_dx.ravel() + w_Y_flat * w_dM_dy.ravel() - si * w_M.ravel())
                if not np.all(np.isfinite(A)) or not np.all(np.isfinite(b)): continue
                try:
                    m, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    if np.isfinite(m).all() and 0 < m[2] < 100: 
                        solutions.append({'longitude': m[0], 'latitude': m[1], 'depth': m[2]})
                except np.linalg.LinAlgError: continue
        return pd.DataFrame(solutions)
    
    def _create_visualizations(self, output_dir, all_results, grid_X, grid_Y, euler_si):
        """Create individual plot files for each analysis result"""
        saved_plots = []
        for name, data in all_results.items():
            fig, ax = plt.subplots(figsize=(10, 8))
            if isinstance(data, pd.DataFrame): # Point data
                if 'depth' in data.columns:
                    # Use UTM coordinates if available, otherwise lat/lon
                    if 'UTM_Easting' in data.columns and 'UTM_Northing' in data.columns:
                        sc = ax.scatter(data['UTM_Easting'], data['UTM_Northing'], c=data['depth'], cmap='viridis_r', s=25, alpha=0.8, vmin=0)
                    else:
                        sc = ax.scatter(data['longitude'], data['latitude'], c=data['depth'], cmap='viridis_r', s=25, alpha=0.8, vmin=0)
                    plt.colorbar(sc, ax=ax, label="Depth (m)", shrink=0.8)
                    ax.set_title(f"Euler Solutions (SI={euler_si})")
                else:
                    if 'UTM_Easting' in data.columns and 'UTM_Northing' in data.columns:
                        ax.plot(data['UTM_Easting'], data['UTM_Northing'], 'k.', markersize=5, alpha=0.7)
                    elif 'longitude' in data.columns and 'latitude' in data.columns:
                        ax.plot(data['longitude'], data['latitude'], 'k.', markersize=5, alpha=0.7)
                    else:
                        # Skip plotting if no coordinate columns found
                        ax.text(0.5, 0.5, 'No coordinate data available', transform=ax.transAxes, ha='center', va='center')
                    ax.set_title(name)
            else: # Grid data
                title = name.replace("_", " ").title()
                cbar_label = "nT/m" if "gradient" in name or "signal" in name else "Radians"
                is_tilt = "tilt" in name
                im = ax.pcolormesh(grid_X, grid_Y, data, cmap='RdYlBu_r', shading='auto')
                plt.colorbar(im, ax=ax, label=cbar_label, shrink=0.8)
                if is_tilt: ax.contour(grid_X, grid_Y, data, levels=[0], colors='black', linewidths=2.5, linestyles='--')
                ax.set_title(title)

            ax.set_aspect('equal'); ax.grid(True, linestyle='--', alpha=0.5)
            # Use appropriate axis labels based on coordinate system and data type
            if isinstance(data, pd.DataFrame):
                if 'UTM_Easting' in data.columns:
                    ax.set_xlabel('UTM Easting (m)'); ax.set_ylabel('UTM Northing (m)')
                elif 'longitude' in data.columns:
                    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
                else:
                    ax.set_xlabel('X'); ax.set_ylabel('Y')
            else:
                # Grid data - always using UTM coordinates now
                ax.set_xlabel('UTM Easting (m)'); ax.set_ylabel('UTM Northing (m)')
            
            plot_path = output_dir / f"{name}.png"
            plt.savefig(plot_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
            saved_plots.append((plot_path, f"Plot of {name}"))
        return saved_plots

    def _create_comprehensive_plot(self, all_results, grid_X, grid_Y, euler_si):
        """Create a comprehensive plot showing all analysis results"""
        # Count the number of results to determine grid layout
        num_results = len(all_results)
        if num_results == 0:
            return None
        
        # Create subplot grid
        if num_results <= 4:
            rows, cols = 2, 2
        elif num_results <= 6:
            rows, cols = 2, 3
        else:
            rows, cols = 3, 3
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 12))
        if num_results == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        fig.suptitle("Magnetic Anomaly Analysis Results", fontsize=16, fontweight='bold')
        
        for i, (name, data) in enumerate(all_results.items()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            if isinstance(data, pd.DataFrame): # Point data
                if 'depth' in data.columns:
                    # Use UTM coordinates if available, otherwise lat/lon
                    if 'UTM_Easting' in data.columns and 'UTM_Northing' in data.columns:
                        sc = ax.scatter(data['UTM_Easting'], data['UTM_Northing'], c=data['depth'], 
                                       cmap='viridis_r', s=15, alpha=0.8, vmin=0)
                    else:
                        sc = ax.scatter(data['longitude'], data['latitude'], c=data['depth'], 
                                       cmap='viridis_r', s=15, alpha=0.8, vmin=0)
                    plt.colorbar(sc, ax=ax, label="Depth (m)", shrink=0.6)
                    ax.set_title(f"Euler Solutions (SI={euler_si})")
                else:
                    if 'UTM_Easting' in data.columns and 'UTM_Northing' in data.columns:
                        ax.plot(data['UTM_Easting'], data['UTM_Northing'], 'k.', markersize=3, alpha=0.7)
                    elif 'longitude' in data.columns and 'latitude' in data.columns:
                        ax.plot(data['longitude'], data['latitude'], 'k.', markersize=3, alpha=0.7)
                    else:
                        # Skip plotting if no coordinate columns found
                        ax.text(0.5, 0.5, 'No coordinate data available', transform=ax.transAxes, ha='center', va='center')
                    ax.set_title(name.replace("_", " ").title())
            else: # Grid data
                title = name.replace("_", " ").title()
                cbar_label = "nT/m" if "gradient" in name or "signal" in name else "Radians"
                is_tilt = "tilt" in name
                im = ax.pcolormesh(grid_X, grid_Y, data, cmap='RdYlBu_r', shading='auto')
                plt.colorbar(im, ax=ax, label=cbar_label, shrink=0.6)
                if is_tilt: ax.contour(grid_X, grid_Y, data, levels=[0], colors='black', linewidths=1.5, linestyles='--')
                ax.set_title(title)

            ax.set_aspect('equal')
            ax.grid(True, linestyle='--', alpha=0.3)
            # Use appropriate axis labels - grid data uses coordinate system from grid_X, grid_Y
            ax.set_xlabel('UTM Easting (m)')
            ax.set_ylabel('UTM Northing (m)')
        
        # Hide unused subplots
        for i in range(len(all_results), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        return fig

    def _create_statistics_report(self, output_dir, all_grids, input_file_name):
        report_path = output_dir / "statistics_report.txt"
        stats_str = f"--- Grid Statistics Report ---\nGenerated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nInput File: {input_file_name}\n\n"
        for name, grid in all_grids.items():
            stats_str += f"--- {name.replace('_', ' ').title()} ---\n"
            if grid is not None and np.any(np.isfinite(grid)):
                stats_str += f"  Min: {np.nanmin(grid):.3f}\n  Max: {np.nanmax(grid):.3f}\n  Mean: {np.nanmean(grid):.3f}\n  Std Dev: {np.nanstd(grid):.3f}\n  Median: {np.nanmedian(grid):.3f}\n\n"
            else: stats_str += "  (No valid data)\n\n"
        with open(report_path, 'w') as f: f.write(stats_str)
        return report_path

    def execute(self, data: pd.DataFrame, params: Dict[str, Any], progress_callback: Optional[Callable] = None, input_file_path: Optional[str] = None) -> ProcessingResult:
        processing_result = ProcessingResult(success=True)
        try:
            if progress_callback: progress_callback(0, "Initializing...")
            
            # Set up traceability metadata
            input_files = [input_file_path] if input_file_path else []
            self.set_current_files(input_files)
            
            analysis_type = params.get('analysis_options', {}).get('analysis_type', {}).get('value', 'all_methods')
            grid_value_col = params.get('input_data', {}).get('grid_value_column', {}).get('value', 'value')
            va_col = params.get('input_data', {}).get('vertical_gradient_column', {}).get('value', 'none')
            gaussian_sigma = params.get('input_data', {}).get('gaussian_sigma', {}).get('value', 1.0)
            si = params.get('euler_parameters', {}).get('structural_index', {}).get('value', 3)
            window_size = params.get('euler_parameters', {}).get('window_size', {}).get('value', 10)
            gen_stats = params.get('output_options', {}).get('generate_statistics_report', {}).get('value', True)
            gen_plots = params.get('output_options', {}).get('generate_plots', {}).get('value', True)
            suffix = params.get('output_options', {}).get('layer_name_suffix', {}).get('value', '')
            
            # Work with lat/lon grid
            grid_pivot = data.pivot_table(index='latitude', columns='longitude', values=grid_value_col)
            y_coords, x_coords = grid_pivot.index.values, grid_pivot.columns.values
            grid_X, grid_Y = np.meshgrid(x_coords, y_coords)
            grid_data_raw = grid_pivot.values
            
            coord_type = 'LatLon'
            
            if progress_callback: progress_callback(0.1, "Interpolating grid edges...")
            # Work with lat/lon grid for all calculations
            points_all = np.array([grid_X.ravel(), grid_Y.ravel()]).T
            nan_mask_flat = np.isnan(grid_data_raw.ravel())
            points_known = points_all[~nan_mask_flat]
            values_known = grid_data_raw.ravel()[~nan_mask_flat]
            grid_data_filled = griddata(points_known, values_known, points_all, method='linear', fill_value=np.nanmean(values_known))
            if np.any(np.isnan(grid_data_filled)): grid_data_filled = griddata(points_known, values_known, points_all, method='nearest')
            grid_data_filled = grid_data_filled.reshape(grid_data_raw.shape)
            grid_data_clean = gaussian_filter(grid_data_filled, sigma=gaussian_sigma) if gaussian_sigma > 0 else grid_data_filled
            nan_mask = np.isnan(grid_data_raw)
            
            va_grid_clean = None
            if va_col != 'none' and va_col in data.columns:
                va_pivot = data.pivot_table(index='latitude', columns='longitude', values=va_col)
                if va_pivot.shape == grid_pivot.shape:
                    va_grid_filled = griddata(points_known, va_pivot.values.ravel()[~nan_mask_flat], points_all, method='linear').reshape(va_pivot.shape)
                    if np.any(np.isnan(va_grid_filled)): va_grid_filled = griddata(points_known, va_pivot.values.ravel()[~nan_mask_flat], points_all, method='nearest').reshape(va_pivot.shape)
                    va_grid_clean = gaussian_filter(va_grid_filled, sigma=gaussian_sigma) if gaussian_sigma > 0 else va_grid_filled

            if progress_callback: progress_callback(0.2, "Calculating derivatives on lat/lon grid...")
            
            # Calculate derivatives on lat/lon grid (like before)
            dx = abs(x_coords[1] - x_coords[0]) if len(x_coords) > 1 else 1
            dy = abs(y_coords[1] - y_coords[0]) if len(y_coords) > 1 else 1
            
            dM_dx, dM_dy, dM_dz = self._calculate_derivatives(grid_data_clean, dx, dy, nan_mask, va_grid_clean)
            
            analyses_to_run = [analysis_type] if analysis_type != 'all_methods' else ['analytic_signal', 'tilt_derivative', 'horizontal_gradient', 'blakely_peaks', 'euler_deconvolution']
            all_results = {}

            for i, current_analysis in enumerate(analyses_to_run):
                prog = 0.3 + (i * 0.1)
                if progress_callback: progress_callback(prog, f"Running {current_analysis.replace('_', ' ')}...")
                if current_analysis == 'analytic_signal': 
                    result = self._run_analytic_signal(dM_dx, dM_dy, dM_dz)
                    all_results['analytic_signal'] = result
                elif current_analysis == 'tilt_derivative': 
                    result = self._run_tilt_derivative(dM_dx, dM_dy, dM_dz)
                    all_results['tilt_derivative'] = result
                elif current_analysis == 'horizontal_gradient': 
                    result = self._run_horizontal_gradient(dM_dx, dM_dy)
                    all_results['horizontal_gradient'] = result
                elif current_analysis == 'blakely_peaks': 
                    hg = all_results.get('horizontal_gradient', self._run_horizontal_gradient(dM_dx, dM_dy))
                    result = self._run_blakely_peaks(hg, grid_X, grid_Y, 'LatLon')
                    all_results['blakely_peaks'] = result
                elif current_analysis == 'euler_deconvolution':
                    grid_data_with_nans = grid_data_clean.copy(); grid_data_with_nans[nan_mask] = np.nan
                    result = self._run_euler_deconvolution(grid_data_with_nans, dM_dx, dM_dy, dM_dz, grid_X, grid_Y, si, window_size, 'LatLon')
                    all_results['euler_deconvolution'] = result
            
            if progress_callback: progress_callback(0.8, "Creating output layers...")
            
            if progress_callback: progress_callback(0.75, "Preparing layer outputs...")
            
            self._add_layer_outputs(processing_result, all_results, grid_X, grid_Y, input_file_path, coord_type)

            output_dir = self._create_output_directory(input_file_path, self.get_project_working_directory())
            if gen_plots:
                if progress_callback: progress_callback(0.85, "Saving plots...")
                viz_files = self._create_visualizations(output_dir, all_results, grid_X, grid_Y, si)
                for file_path, desc in viz_files: processing_result.add_output_file(str(file_path), 'png', desc)
            
            if gen_stats:
                if progress_callback: progress_callback(0.9, "Saving statistics report...")
                grid_for_stats = grid_data_clean.copy(); grid_for_stats[nan_mask] = np.nan
                all_results['input_data_smoothed'] = grid_for_stats
                report_path = self._create_statistics_report(output_dir, {k:v for k,v in all_results.items() if not isinstance(v, pd.DataFrame)}, Path(input_file_path).name if input_file_path else "N/A")
                processing_result.add_output_file(str(report_path), 'txt', "Grid statistics report")

            # Create comprehensive plot for the data viewer
            if progress_callback: progress_callback(0.95, "Creating comprehensive plot...")
            comprehensive_fig = self._create_comprehensive_plot(all_results, grid_X, grid_Y, si)
            processing_result.figure = comprehensive_fig
            
            # Set essential metadata
            processing_result.metadata.update({
                'processor': 'magnetic',
                'analysis_method': analysis_type,
                'gaussian_sigma': gaussian_sigma,
                'structural_index': si,
                'window_size': window_size,
                'total_analyses': len(all_results),
                'grid_shape': grid_data_clean.shape,
                'coordinate_system': coord_type,
                'grid_spacing_deg': {'dx': dx, 'dy': dy}
            })
            
            if progress_callback: progress_callback(1.0, "Processing complete!")
            return processing_result
        except Exception as e:
            raise ProcessingError(f"Magnetic Anomaly Processing failed: {str(e)}")

    def _add_layer_outputs(self, result: ProcessingResult, all_results: dict, grid_X: np.ndarray, grid_Y: np.ndarray, input_file_path: Optional[str] = None, coord_type: str = 'LatLon'):
        """Add layer outputs for map visualization integration"""
        
        # Create unique layer names with timestamp
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        if input_file_path:
            input_filename = Path(input_file_path).stem
        else:
            input_filename = "anomaly_analysis"
        
        # Process each analysis result
        for name, data in all_results.items():
            if isinstance(data, pd.DataFrame):
                # Point data (Euler solutions, Blakely peaks)
                if not data.empty:
                    layer_name = f'{input_filename} - {name.replace("_", " ").title()} ({timestamp})'
                    
                    # Data should already be in lat/lon coordinates
                    
                    # Determine style based on data type
                    if 'depth' in data.columns:
                        # Euler solutions with depth information
                        style_info = {
                            'color_field': 'depth',
                            'use_graduated_colors': True,
                            'color_scheme': 'viridis_r',
                            'size': 6,
                            'opacity': 0.9,
                            'show_labels': True,
                            'label_field': 'depth'
                        }
                    else:
                        # Blakely peaks or other point data
                        style_info = {
                            'color': '#FF0000',
                            'size': 8,
                            'opacity': 1.0,
                            'show_labels': False
                        }
                    
                    result.add_layer_output(
                        layer_type='processed',
                        data=data,
                        style_info=style_info,
                        metadata={
                            'description': f'{name.replace("_", " ").title()} analysis results',
                            'layer_name': layer_name,
                            'total_points': len(data),
                            'data_type': name,
                            'coordinate_system': coord_type
                        }
                    )
            else:
                # Grid data (analytic signal, tilt derivative, etc.) - keep as raster
                if data is not None and np.any(np.isfinite(data)):
                    layer_name = f'{input_filename} - {name.replace("_", " ").title()} ({timestamp})'
                    
                    # Determine color scheme based on analysis type
                    if 'tilt' in name:
                        color_scheme = 'RdYlBu_r'
                    elif 'signal' in name or 'gradient' in name:
                        color_scheme = 'magnetic'
                    else:
                        color_scheme = 'viridis'
                    
                    # Grid is already in lat/lon coordinates
                    bounds = [
                        float(np.nanmin(grid_X)),  # west (min_x) - longitude
                        float(np.nanmin(grid_Y)),  # south (min_y) - latitude
                        float(np.nanmax(grid_X)),  # east (max_x) - longitude
                        float(np.nanmax(grid_Y))   # north (max_y) - latitude
                    ]
                    
                    # Create raster data structure like grid interpolator
                    raster_data = {
                        'grid': data,
                        'bounds': bounds,
                        'field_name': name
                    }
                    
                    result.add_layer_output(
                        layer_type='raster',
                        data=raster_data,
                        style_info={
                            'use_graduated_colors': True,
                            'color_scheme': color_scheme,
                            'opacity': 0.7
                        },
                        metadata={
                            'description': f'{name.replace("_", " ").title()} grid analysis',
                            'layer_name': layer_name,
                            'data_type': name,
                            'bounds': bounds
                        }
                    )

SCRIPT_CLASS = MagneticAnomalyProcessor