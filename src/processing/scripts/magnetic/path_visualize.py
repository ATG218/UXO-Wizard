"""
Path Visualize Script for UXO-Wizard Framework

This script provides a quick visualization of unprocessed magnetic survey data
to display the drone's flight path, altitude, and other flight characteristics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Callable
from pathlib import Path
from datetime import datetime

from src.processing.base import ScriptInterface, ProcessingResult, ProcessingError

class PathVisualize(ScriptInterface):
    """
    A script to quickly visualize the flight path and sensor data from
    unprocessed magnetic survey files.
    """

    @property
    def name(self) -> str:
        return "Path Visualizer"

    @property
    def description(self) -> str:
        return "Quickly visualizes drone flight path and sensor data, and saves statistics to a text file."

    def get_parameters(self) -> Dict[str, Any]:
        """Returns the parameters for the script."""
        return {
            'visualization_options': {
                'generate_plots': {
                    'value': True,
                    'type': 'bool',
                    'description': 'Generate all visualization plots.'
                },
                 'plot_title': {
                    'value': 'Flight Path and Sensor Data Visualization',
                    'type': 'str',
                    'description': 'Main title for the combined plot.'
                }
            },
            'output_options': {
                'generate_stats_file': {
                    'value': True,
                    'type': 'bool',
                    'description': 'Generate a separate .txt file for data statistics.'
                }
            }
        }

    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validates that the necessary columns are present in the DataFrame."""
        if data.empty:
            raise ProcessingError("Input data cannot be empty.")

        required_cols = ['Latitude [Decimal Degrees]', 'Longitude [Decimal Degrees]']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ProcessingError(f"Missing required columns for path visualization: {', '.join(missing_cols)}")

        return True

    def execute(self, data: pd.DataFrame, params: Dict[str, Any],
                progress_callback: Optional[Callable] = None,
                input_file_path: Optional[str] = None) -> ProcessingResult:
        """Executes the visualization script."""
        if progress_callback:
            progress_callback(0, "Starting path visualization...")

        df = data.copy()
        result = ProcessingResult(success=True, processing_script=self.name)

        # --- Data Cleaning and Preparation ---
        cols_to_convert = [
            'Latitude [Decimal Degrees]', 'Longitude [Decimal Degrees]', 'Altitude [m]',
            'B1x [nT]', 'B1y [nT]', 'B1z [nT]', 'B2x [nT]', 'B2y [nT]', 'B2z [nT]',
            'AccX [g]', 'AccY [g]', 'AccZ [g]', 'Temp [Deg]'
        ]
        for col in cols_to_convert:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop rows where essential coordinates are missing or zero for visualization
        vis_df = df.dropna(subset=['Latitude [Decimal Degrees]', 'Longitude [Decimal Degrees]']).copy()
        vis_df = vis_df[
            (vis_df['Latitude [Decimal Degrees]'] != 0.0) & 
            (vis_df['Longitude [Decimal Degrees]'] != 0.0)
        ].copy()

        if vis_df.empty:
            return ProcessingResult(success=False, error_message="No valid coordinate data found for visualization after cleaning.")

        if progress_callback:
            progress_callback(20, "Data prepared. Generating outputs...")

        # --- Generate Statistics File ---
        if params.get('output_options', {}).get('generate_stats_file', {}).get('value', True):
            stats = df[cols_to_convert].describe().transpose()
            stats_subset = stats[['mean', 'std', 'min', 'max', 'count']]
            stats_text = stats_subset.to_string()
            
            # Create a unique filename for the stats file and write to disk
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            input_stem = Path(input_file_path).stem if input_file_path else "visualization"
            stats_filename = f"{input_stem}_stats_{timestamp}.txt"
            
            # Write stats file to the processed directory
            try:
                # Get the processed directory path
                if hasattr(self, 'project_manager') and self.project_manager:
                    working_dir = self.project_manager.get_current_working_directory()
                    if working_dir:
                        processed_dir = Path(working_dir) / "processed" / "magnetic"
                        processed_dir.mkdir(parents=True, exist_ok=True)
                        stats_file_path = processed_dir / stats_filename
                        
                        # Write stats to file
                        with open(stats_file_path, 'w') as f:
                            f.write(f"Statistical Summary of Sensor Data\n")
                            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                            f.write(f"Input file: {Path(input_file_path).name if input_file_path else 'N/A'}\n\n")
                            f.write(stats_text)
                        
                        result.add_output_file(
                            file_path=str(stats_file_path),
                            file_type="txt",
                            description="Statistical summary of the sensor data."
                        )
                        result.metadata['statistics_file'] = str(stats_file_path)
                    else:
                        # Fallback to metadata storage
                        result.add_output_file(
                            file_path=stats_filename,
                            file_type="txt", 
                            description="Statistical summary of the sensor data.",
                            metadata={'content': stats_text}
                        )
                else:
                    # Fallback to metadata storage
                    result.add_output_file(
                        file_path=stats_filename,
                        file_type="txt",
                        description="Statistical summary of the sensor data.",
                        metadata={'content': stats_text}
                    )
            except Exception as e:
                # Fallback to metadata storage
                result.add_output_file(
                    file_path=stats_filename,
                    file_type="txt",
                    description="Statistical summary of the sensor data.",
                    metadata={'content': stats_text, 'write_error': str(e)}
                )


        # --- Plotting ---
        if params.get('visualization_options', {}).get('generate_plots', {}).get('value', True):
            fig, axs = plt.subplots(2, 2, figsize=(20, 16))
            fig.suptitle(params.get('visualization_options', {}).get('plot_title', {}).get('value'), fontsize=16, y=0.95)

            # 1. Flight Path (Lat vs Lon)
            ax = axs[0, 0]
            ax.plot(vis_df['Longitude [Decimal Degrees]'], vis_df['Latitude [Decimal Degrees]'], marker='.', linestyle='-', markersize=2)
            ax.set_title('Flight Path')
            ax.set_xlabel('Longitude [Decimal Degrees]')
            ax.set_ylabel('Latitude [Decimal Degrees]')
            ax.grid(True)
            # Remove equal aspect ratio to prevent squishing
            ax.set_aspect('auto')

            # 2. Altitude Profile
            ax = axs[0, 1]
            if 'Altitude [m]' in vis_df.columns and not vis_df['Altitude [m]'].isnull().all():
                ax.plot(vis_df.index, vis_df['Altitude [m]'])
                ax.set_title('Altitude Profile')
                ax.set_xlabel('Sample Index')
                ax.set_ylabel('Altitude [m]')
                ax.grid(True)
            else:
                ax.text(0.5, 0.5, 'Altitude data not available', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Altitude Profile')

            if progress_callback:
                progress_callback(50, "Plotting sensor data...")

            # 3. Magnetic Field Sensors
            ax = axs[1, 0]
            mag_cols1 = ['B1x [nT]', 'B1y [nT]', 'B1z [nT]']
            mag_cols2 = ['B2x [nT]', 'B2y [nT]', 'B2z [nT]']
            for col in mag_cols1:
                if col in vis_df.columns and not vis_df[col].isnull().all():
                    ax.plot(vis_df.index, vis_df[col], label=f"S1 {col.split(' ')[0]}")
            for col in mag_cols2:
                 if col in vis_df.columns and not vis_df[col].isnull().all():
                    ax.plot(vis_df.index, vis_df[col], label=f"S2 {col.split(' ')[0]}", linestyle='--')
            ax.set_title('Magnetic Field Sensors')
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Magnetic Field [nT]')
            ax.legend()
            ax.grid(True)

            # 4. Accelerometer Data
            ax = axs[1, 1]
            accel_cols = ['AccX [g]', 'AccY [g]', 'AccZ [g]']
            for col in accel_cols:
                if col in vis_df.columns and not vis_df[col].isnull().all():
                    ax.plot(vis_df.index, vis_df[col], label=col)
            ax.set_title('Accelerometer Data')
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Acceleration [g]')
            ax.legend()
            ax.grid(True)

            plt.tight_layout(rect=[0, 0.03, 1, 0.92])
            plt.subplots_adjust(hspace=0.3, wspace=0.3)
            result.figure = fig

        if progress_callback:
            progress_callback(90, "Finalizing result...")

        # --- Layer Generation ---
        # Create proper vector flight path similar to flight_path_segmenter
        # Sort by index to ensure chronological order and filter zeros
        df_clean = vis_df[
            (vis_df['Latitude [Decimal Degrees]'] != 0.0) & 
            (vis_df['Longitude [Decimal Degrees]'] != 0.0)
        ].sort_index().copy()
        
        if df_clean.empty:
            return ProcessingResult(success=False, error_message="No valid GPS coordinates found for flight path visualization")
        
        # Downsample for better performance but maintain flight path continuity
        downsample_factor = max(1, len(df_clean) // 1000)
        layer_df = df_clean[['Latitude [Decimal Degrees]', 'Longitude [Decimal Degrees]']].iloc[::downsample_factor].copy()
        
        # Create vector structure like flight_path_segmenter
        layer_df = layer_df.rename(columns={
            'Latitude [Decimal Degrees]': 'latitude',
            'Longitude [Decimal Degrees]': 'longitude'
        })
        
        # Add required fields for vector line rendering (like flight_path_segmenter does)
        layer_df['line_id'] = 0  # Single continuous flight path
        layer_df['segment_name'] = 'Complete Flight Path'
        
        # Reset index to ensure proper line continuity
        layer_df = layer_df.reset_index(drop=True)
        
        # Create a unique layer name with timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        input_stem = Path(input_file_path).stem if input_file_path else "data"
        layer_name = f'Flight Path - {input_stem} ({timestamp})'

        # Use flight_lines layer type to trigger LINE geometry detection
        result.add_layer_output(
            layer_type='flight_lines',
            data=layer_df,
            style_info={
                'line_color': '#007BFF',
                'line_width': 2.5,
                'line_opacity': 0.9
            },
            metadata={
                'layer_name': layer_name,
                'description': f'Drone flight path vector ({len(layer_df)} points)',
                'data_type': 'flight_path',
                'coordinate_columns': {
                    'latitude': 'latitude',
                    'longitude': 'longitude'
                }
            }
        )
        
        # --- Final Metadata ---
        result.metadata.update({
            'processor': 'magnetic',
            
            'script': self.name,
            'total_points': len(df),
            'visualized_points': len(vis_df),
            'processing_timestamp': datetime.now().isoformat(),
        })

        if progress_callback:
            progress_callback(100, "Visualization complete.")

        return result

SCRIPT_CLASS = PathVisualize