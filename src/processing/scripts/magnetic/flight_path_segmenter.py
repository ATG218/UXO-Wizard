"""
Flight Path Segmentation Script for UXO Wizard Framework
========================================================

Processes magnetic survey data to automatically segment flight paths into 
directional segments based on heading consistency. Adapted from standalone 
flight_path_segmenter.py to integrate with UXO Wizard's script framework.

Features:
- Automatic detection of straight-line flight segments
- Removal of start/end portions and turning segments  
- Separation by heading direction for grid pattern surveys
- Comprehensive visualization of flight path and segmentation
- UTM coordinate plotting for quality control
- Integration with UXO Wizard's processing pipeline

Author: Adapted for UXO Wizard Framework
Original: flight_path_segmenter.py from mag_import pipeline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import datetime
from typing import Dict, Any, Optional, Callable, List, Tuple
from scipy.ndimage import uniform_filter1d
import tempfile
import os

from src.processing.base import ScriptInterface, ProcessingResult, ProcessingError


class FlightPathSegmenter(ScriptInterface):
    """
    Flight path segmentation script for magnetic survey data processing
    """
    
    @property
    def name(self) -> str:
        return "Flight Path Segmenter"
    
    @property  
    def description(self) -> str:
        return "Automatically segment flight paths into directional segments based on heading consistency for grid pattern surveys"
    
    def get_parameters(self) -> Dict[str, Any]:
        """Return parameter structure for flight path segmentation"""
        return {
            'segmentation_parameters': {
                'heading_window_size': {
                    'value': 51,
                    'type': 'int',
                    'min': 5,
                    'max': 201,
                    'description': 'Window size for heading smoothing (odd number recommended)'
                },
                'min_segment_length': {
                    'value': 100,
                    'type': 'int', 
                    'min': 10,
                    'max': 1000,
                    'description': 'Minimum points per segment'
                },
                'heading_tolerance': {
                    'value': 5.0,
                    'type': 'float',
                    'min': 1.0,
                    'max': 30.0,
                    'description': 'Degrees tolerance for consistent heading'
                },
                'stability_window': {
                    'value': 200,
                    'type': 'int',
                    'min': 50,
                    'max': 500,
                    'description': 'Points to analyze for path stability'
                },
                'min_stable_duration': {
                    'value': 5000,
                    'type': 'int',
                    'min': 100,
                    'max': 20000,
                    'description': 'Minimum points for a stable segment'
                },
                'heading_change_threshold': {
                    'value': 10.0,
                    'type': 'float',
                    'min': 1.0,
                    'max': 45.0,
                    'description': 'Max heading change rate for stability (deg/point)'
                },
                'min_points_for_valid_line': {
                    'value': 50,
                    'type': 'int',
                    'min': 10,
                    'max': 500,
                    'description': 'Minimum points required for a valid flight line'
                }
            },
            'quality_control': {
                'min_speed_ms': {
                    'value': 0.5,
                    'type': 'float',
                    'min': 0.1,
                    'max': 10.0,
                    'description': 'Minimum speed to consider valid movement (m/s)'
                },
                'max_heading_change_rate': {
                    'value': 5.0,
                    'type': 'float',
                    'min': 1.0,
                    'max': 20.0,
                    'description': 'Max heading change per second (degrees/s)'
                }
            },
            'heading_combination': {
                'combine_opposite_headings': {
                    'value': True,
                    'type': 'bool',
                    'description': 'Combine opposite headings (N/S and E/W)'
                },
                'opposite_heading_tolerance': {
                    'value': 30.0,
                    'type': 'float',
                    'min': 5.0,
                    'max': 60.0,
                    'description': 'Tolerance for considering headings as opposite (degrees)'
                }
            },
            'visualization_settings': {
                'figure_size_width': {
                    'value': 15,
                    'type': 'int',
                    'min': 8,
                    'max': 25,
                    'description': 'Plot figure width in inches'
                },
                'figure_size_height': {
                    'value': 10,
                    'type': 'int',
                    'min': 6,
                    'max': 20,
                    'description': 'Plot figure height in inches'
                },
                'dpi': {
                    'value': 300,
                    'type': 'int',
                    'min': 72,
                    'max': 600,
                    'description': 'Plot resolution (DPI)'
                },
                'create_visualizations': {
                    'value': True,
                    'type': 'bool',
                    'description': 'Generate visualization plots and maps'
                }
            }
        }
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate that data has required columns for flight path segmentation"""
        if data.empty:
            raise ProcessingError("Input data cannot be empty")
        
        # Required columns for flight path segmentation
        required_columns = [
            'UTM_Easting', 'UTM_Northing', 'Timestamp [ms]',
            'Latitude [Decimal Degrees]', 'Longitude [Decimal Degrees]', 
            'Altitude [m]'
        ]
        
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            raise ProcessingError(f"Missing required columns for flight path segmentation: {missing_cols}")
        
        # Check for minimum data points
        if len(data) < 100:
            raise ProcessingError("Need at least 100 data points for meaningful flight path segmentation")
        
        return True
    
    def execute(self, data: pd.DataFrame, params: Dict[str, Any], 
                progress_callback: Optional[Callable] = None, 
                input_file_path: Optional[str] = None) -> ProcessingResult:
        """Execute flight path segmentation processing"""
        
        try:
            if progress_callback:
                progress_callback(0, "Starting flight path segmentation...")
            
            # Extract parameters
            seg_params = params.get('segmentation_parameters', {})
            qc_params = params.get('quality_control', {})
            heading_params = params.get('heading_combination', {})
            viz_params = params.get('visualization_settings', {})
            
            # Create working copy of data
            df = data.copy()
            
            if progress_callback:
                progress_callback(10, "Calculating flight metrics...")
            
            # Calculate headings and speeds
            headings = self._calculate_heading(df)
            speeds = self._calculate_speed(df)
            
            if progress_callback:
                progress_callback(20, "Smoothing heading data...")
            
            # Smooth headings
            window_size = seg_params.get('heading_window_size', {}).get('value', 51)
            smoothed_headings = self._smooth_heading(headings, window_size)
            
            if progress_callback:
                progress_callback(30, "Detecting stabilized segments...")
            
            # Detect stabilized segments
            segments = self._detect_stabilized_segments(
                df, smoothed_headings, speeds, seg_params, qc_params
            )
            
            if not segments:
                raise ProcessingError("No stabilized flight segments detected. Check parameters.")
            
            if progress_callback:
                progress_callback(50, "Grouping segments by heading...")
            
            # Group by heading direction
            direction_segments = self._group_segments_by_heading(
                segments, df, seg_params
            )
            
            if progress_callback:
                progress_callback(60, "Processing heading combinations...")
            
            # Combine opposite headings if configured
            final_segments = self._combine_opposite_headings(
                direction_segments, heading_params
            )
            
            if progress_callback:
                progress_callback(70, "Generating output files...")
            
            # Create output directory
            output_dir = self._create_output_directory(input_file_path)
            base_filename = self._get_base_filename(input_file_path)
            
            # Save segmented data files
            saved_files = self._save_segments(final_segments, output_dir, base_filename)
            
            # Create processing result
            result = ProcessingResult(
                success=True,
                data=df,  # Return original data as primary result
                processing_script=self.name,
                metadata={
                    'processor': 'magnetic',
                    'total_segments': len(final_segments),
                    'segment_details': self._get_segment_statistics(final_segments),
                    'original_points': len(df),
                    'segmented_points': sum(len(seg) for seg_list in final_segments.values() for seg in seg_list),
                    'parameters': params
                }
            )
            
            # Add segmented CSV files as outputs
            for file_path in saved_files:
                result.add_output_file(
                    file_path=str(file_path),
                    file_type="csv",
                    description=f"Segmented flight data: {file_path.stem}"
                )
            
            if progress_callback:
                progress_callback(80, "Creating visualizations...")
            
            # Create visualizations if enabled
            if viz_params.get('create_visualizations', {}).get('value', True):
                viz_files = self._create_visualizations(
                    df, final_segments, output_dir, viz_params, headings, smoothed_headings, speeds
                )
                
                # Add visualization files as outputs
                for viz_file, description in viz_files:
                    result.add_output_file(
                        file_path=str(viz_file),
                        file_type="png", 
                        description=description
                    )
            
            if progress_callback:
                progress_callback(90, "Creating summary report...")
            
            # Create summary report
            summary_file = self._create_summary_report(df, final_segments, output_dir, params)
            result.add_output_file(
                file_path=str(summary_file),
                file_type="txt",
                description="Flight path segmentation summary report"
            )
            
            if progress_callback:
                progress_callback(95, "Adding layer outputs...")
            
            # Add layer outputs for map visualization
            self._add_layer_outputs(result, df, final_segments, input_file_path)
            
            if progress_callback:
                progress_callback(100, "Flight path segmentation complete!")
            
            return result
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                error_message=f"Flight path segmentation failed: {str(e)}"
            )
    
    def _calculate_heading(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate heading (bearing) from UTM coordinates"""
        # Calculate differences
        dx = np.diff(df['UTM_Easting'].values)
        dy = np.diff(df['UTM_Northing'].values)
        
        # Calculate bearing in radians
        bearing_rad = np.arctan2(dx, dy)
        
        # Convert to degrees (0-360)
        bearing_deg = np.degrees(bearing_rad) % 360
        
        # Add first value (assume same as second)
        if len(bearing_deg) > 0:
            bearing_deg = np.insert(bearing_deg, 0, bearing_deg[0])
        else:
            bearing_deg = np.array([0])
        
        return bearing_deg
    
    def _calculate_speed(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate speed from position data"""
        # Calculate distances
        dx = np.diff(df['UTM_Easting'].values)
        dy = np.diff(df['UTM_Northing'].values)
        distances = np.sqrt(dx**2 + dy**2)
        
        # Calculate time differences (convert from ms to seconds)
        dt = np.diff(df['Timestamp [ms]'].values) / 1000.0
        
        # Avoid division by zero
        dt[dt == 0] = 1e-6
        
        # Calculate speed
        speeds = distances / dt
        
        # Add first value
        if len(speeds) > 0:
            speeds = np.insert(speeds, 0, speeds[0])
        else:
            speeds = np.array([0])
        
        return speeds
    
    def _smooth_heading(self, headings: np.ndarray, window_size: int) -> np.ndarray:
        """Smooth heading data accounting for circular nature (0-360 degrees)"""
        if len(headings) < window_size:
            return headings
        
        # Convert to complex numbers for circular smoothing
        complex_headings = np.exp(1j * np.radians(headings))
        
        # Apply uniform filter
        smoothed_complex = uniform_filter1d(complex_headings.real, window_size) + \
                          1j * uniform_filter1d(complex_headings.imag, window_size)
        
        # Convert back to degrees
        smoothed_headings = np.degrees(np.angle(smoothed_complex)) % 360
        
        return smoothed_headings
    
    def _detect_stabilized_segments(self, df: pd.DataFrame, smoothed_headings: np.ndarray, 
                                  speeds: np.ndarray, seg_params: Dict, qc_params: Dict) -> List[Tuple[int, int, float]]:
        """Detect stabilized flight segments by analyzing heading consistency"""
        
        # Extract parameters
        stability_window = seg_params.get('stability_window', {}).get('value', 200)
        min_stable_duration = seg_params.get('min_stable_duration', {}).get('value', 5000)
        heading_change_threshold = seg_params.get('heading_change_threshold', {}).get('value', 10.0)
        min_segment_length = seg_params.get('min_segment_length', {}).get('value', 100)
        heading_tolerance = seg_params.get('heading_tolerance', {}).get('value', 5.0)
        min_speed_ms = qc_params.get('min_speed_ms', {}).get('value', 0.5)
        
        # Calculate heading change rate (stability metric)
        heading_changes = np.abs(np.diff(smoothed_headings))
        # Handle wrap-around (359° to 1°)
        heading_changes = np.minimum(heading_changes, 360 - heading_changes)
        # Pad to match original length
        heading_changes = np.append(heading_changes, heading_changes[-1])
        
        # Calculate rolling stability metrics
        stability_metric = uniform_filter1d(heading_changes, stability_window, mode='constant', cval=999)
        
        # Identify stable regions (low heading change rate + adequate speed)
        is_stable = (stability_metric < heading_change_threshold) & (speeds > min_speed_ms)
        
        # Find continuous stable segments
        segments = []
        in_segment = False
        start_idx = 0
        
        for i in range(len(is_stable)):
            if is_stable[i] and not in_segment:
                # Start of new stable segment
                start_idx = i
                in_segment = True
            elif not is_stable[i] and in_segment:
                # End of stable segment
                if i - start_idx >= min_stable_duration:
                    # Calculate average heading for this stable segment
                    segment_headings = smoothed_headings[start_idx:i]
                    # Use circular mean for headings
                    x = np.mean(np.cos(np.radians(segment_headings)))
                    y = np.mean(np.sin(np.radians(segment_headings)))
                    avg_heading = np.degrees(np.arctan2(y, x)) % 360
                    
                    segments.append((start_idx, i-1, avg_heading))
                in_segment = False
        
        # Handle case where we end in a stable segment
        if in_segment and len(df) - start_idx >= min_stable_duration:
            segment_headings = smoothed_headings[start_idx:]
            x = np.mean(np.cos(np.radians(segment_headings)))
            y = np.mean(np.sin(np.radians(segment_headings)))
            avg_heading = np.degrees(np.arctan2(y, x)) % 360
            segments.append((start_idx, len(df)-1, avg_heading))
        
        # Further refine segments by grouping nearby segments with similar headings
        refined_segments = []
        if segments:
            current_start, current_end, current_heading = segments[0]
            
            for i in range(1, len(segments)):
                next_start, next_end, next_heading = segments[i]
                
                # Check if headings are similar and segments are close
                heading_diff = abs(current_heading - next_heading)
                heading_diff = min(heading_diff, 360 - heading_diff)  # Handle wrap-around
                
                gap_size = next_start - current_end
                
                if heading_diff <= heading_tolerance and gap_size <= min_segment_length:
                    # Merge segments
                    current_end = next_end
                    # Recalculate average heading
                    merged_headings = smoothed_headings[current_start:current_end+1]
                    x = np.mean(np.cos(np.radians(merged_headings)))
                    y = np.mean(np.sin(np.radians(merged_headings)))
                    current_heading = np.degrees(np.arctan2(y, x)) % 360
                else:
                    # Save current segment and start new one
                    if current_end - current_start >= min_segment_length:
                        refined_segments.append((current_start, current_end, current_heading))
                    current_start, current_end, current_heading = next_start, next_end, next_heading
            
            # Add final segment
            if current_end - current_start >= min_segment_length:
                refined_segments.append((current_start, current_end, current_heading))
        
        return refined_segments
    
    def _group_segments_by_heading(self, segments: List[Tuple[int, int, float]], 
                                 df: pd.DataFrame, seg_params: Dict) -> Dict[str, List[pd.DataFrame]]:
        """Group segments by specific heading degree, keeping them as separate parallel lines"""
        
        if not segments:
            return {}
        
        # Extract parameters
        heading_tolerance = seg_params.get('heading_tolerance', {}).get('value', 5.0)
        min_points_for_valid_line = seg_params.get('min_points_for_valid_line', {}).get('value', 50)
        
        # Extract headings and cluster them
        headings = [seg[2] for seg in segments]
        
        # Find unique heading directions (allowing for some tolerance)
        heading_groups = []
        used_segments = set()
        
        for i, heading in enumerate(headings):
            if i in used_segments:
                continue
                
            # Find all segments with similar heading
            group_segments = []
            for j, other_heading in enumerate(headings):
                if j in used_segments:
                    continue
                    
                # Calculate angular difference
                diff = abs(heading - other_heading)
                diff = min(diff, 360 - diff)  # Handle wrap-around
                
                if diff <= heading_tolerance * 2:  # Allow more tolerance for grouping
                    group_segments.append(j)
                    used_segments.add(j)
            
            if group_segments:
                heading_groups.append(group_segments)
        
        # Create separate DataFrames for each heading group
        direction_segments = {}
        
        for segment_indices in heading_groups:
            # Calculate representative heading for this group
            group_headings = [headings[i] for i in segment_indices]
            
            # Handle circular mean for headings
            x = np.mean(np.cos(np.radians(group_headings)))
            y = np.mean(np.sin(np.radians(group_headings)))
            avg_heading = np.degrees(np.arctan2(y, x)) % 360
            
            # Use the exact heading degree for naming (rounded to nearest degree)
            direction = f"Heading_{int(round(avg_heading))}"
            
            # Keep each segment as a separate DataFrame (independent parallel lines)
            direction_lines = []
            total_points = 0
            
            for seg_idx in segment_indices:
                start, end, _ = segments[seg_idx]
                segment_df = df.iloc[start:end+1].copy()
                # Add a condition to check for minimum points per line
                if len(segment_df) >= min_points_for_valid_line:
                    direction_lines.append(segment_df)
                    total_points += len(segment_df)
            
            if direction_lines:
                direction_segments[direction] = direction_lines
        
        return direction_segments
    
    def _combine_opposite_headings(self, direction_segments: Dict[str, List[pd.DataFrame]], 
                                 heading_params: Dict) -> Dict[str, List[pd.DataFrame]]:
        """Combine opposite headings (e.g., North/South and East/West) based on configuration"""
        
        combine_opposite = heading_params.get('combine_opposite_headings', {}).get('value', True)
        if not combine_opposite:
            return direction_segments
        
        tolerance = heading_params.get('opposite_heading_tolerance', {}).get('value', 30.0)
        
        # Extract heading values from direction names
        heading_data = {}
        for direction, seg_lines in direction_segments.items():
            try:
                heading_value = int(direction.split('_')[1])
                heading_data[heading_value] = (direction, seg_lines)
            except (IndexError, ValueError):
                continue
        
        # Find opposite heading pairs
        combined_segments = {}
        used_headings = set()
        
        for heading1, (dir1, lines1) in heading_data.items():
            if heading1 in used_headings:
                continue
                
            # Find the opposite heading (±180 degrees)
            opposite_heading = (heading1 + 180) % 360
            best_match = None
            best_diff = float('inf')
            
            for heading2, (dir2, lines2) in heading_data.items():
                if heading2 in used_headings or heading2 == heading1:
                    continue
                    
                # Check if it's close to opposite
                diff1 = abs(heading2 - opposite_heading)
                diff2 = abs(heading2 - opposite_heading + 360)
                diff3 = abs(heading2 - opposite_heading - 360)
                diff = min(diff1, diff2, diff3)
                
                if diff <= tolerance and diff < best_diff:
                    best_match = (heading2, dir2, lines2)
                    best_diff = diff
            
            if best_match:
                heading2, dir2, lines2 = best_match
                
                # Combine the segments
                combined_lines = lines1 + lines2
                
                # Create a descriptive name
                h1_cardinal = self._get_cardinal_direction(heading1)
                h2_cardinal = self._get_cardinal_direction(heading2)
                combined_name = f"{h1_cardinal}_{h2_cardinal}_Combined_{heading1}_{heading2}"
                
                combined_segments[combined_name] = combined_lines
                used_headings.add(heading1)
                used_headings.add(heading2)
            else:
                # No opposite found, keep original
                combined_segments[dir1] = lines1
                used_headings.add(heading1)
        
        return combined_segments
    
    def _get_cardinal_direction(self, heading: float) -> str:
        """Convert heading to cardinal direction name"""
        if 337.5 <= heading or heading < 22.5:
            return "N"
        elif 22.5 <= heading < 67.5:
            return "NE"
        elif 67.5 <= heading < 112.5:
            return "E"
        elif 112.5 <= heading < 157.5:
            return "SE"
        elif 157.5 <= heading < 202.5:
            return "S"
        elif 202.5 <= heading < 247.5:
            return "SW"
        elif 247.5 <= heading < 292.5:
            return "W"
        elif 292.5 <= heading < 337.5:
            return "NW"
        else:
            return f"H{int(heading)}"
    
    def _create_output_directory(self, input_file_path: Optional[str]) -> Path:
        """Create output directory for segmentation results in project/processed/magnetic/"""
        if input_file_path:
            input_path = Path(input_file_path)
            base_filename = input_path.stem
            
            # Find project root - look for working directory or use input file parent
            project_dir = input_path.parent
            while project_dir.parent != project_dir:  # Not at filesystem root
                if (project_dir / "processed").exists() or len(list(project_dir.glob("*.uxo"))) > 0:
                    break
                project_dir = project_dir.parent
            
            # Create project/processed/magnetic/filename_segmented structure
            output_dir = project_dir / "processed" / "magnetic" / f"{base_filename}_segmented"
        else:
            # Use temporary directory if no input file path
            temp_dir = tempfile.mkdtemp(prefix="flight_segmentation_")
            output_dir = Path(temp_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    def _get_base_filename(self, input_file_path: Optional[str]) -> str:
        """Get base filename for output files"""
        if input_file_path:
            return Path(input_file_path).stem
        else:
            return "flight_data"
    
    def _save_segments(self, segments: Dict[str, List[pd.DataFrame]], 
                      output_dir: Path, base_filename: str) -> List[Path]:
        """Save segmented data to CSV files"""
        saved_files = []
        
        # Sort by number of lines and keep only the top 2 headings
        sorted_segments = sorted(segments.items(), 
                               key=lambda item: sum(len(seg) for seg in item[1]), 
                               reverse=True)
        top_segments = dict(sorted_segments[:2])  # Get the top 2 headings
        
        for direction, seg_lines in top_segments.items():
            if not seg_lines:
                continue
            
            # Combine all lines for the current heading direction into one DataFrame
            combined_df = pd.concat(seg_lines, ignore_index=True)
            
            # Save to a single CSV file with the exact heading degree
            filename = f"{base_filename}_{direction}.csv"
            filepath = output_dir / filename
            
            # Save CSV
            try:
                combined_df.to_csv(filepath, index=False)
                saved_files.append(filepath)
            except Exception as e:
                raise ProcessingError(f"Error saving segment '{direction}': {e}")
        
        return saved_files
    
    def _create_visualizations(self, df: pd.DataFrame, segments: Dict[str, List[pd.DataFrame]], 
                             output_dir: Path, viz_params: Dict, headings: np.ndarray, 
                             smoothed_headings: np.ndarray, speeds: np.ndarray) -> List[Tuple[Path, str]]:
        """Create visualization plots and return list of (filepath, description) tuples"""
        
        viz_files = []
        
        # Extract visualization parameters
        fig_width = viz_params.get('figure_size_width', {}).get('value', 15)
        fig_height = viz_params.get('figure_size_height', {}).get('value', 10) 
        dpi = viz_params.get('dpi', {}).get('value', 300)
        figure_size = (fig_width, fig_height)
        
        # 1. Create overview plots
        overview_file = self._create_overview_plots(df, output_dir, figure_size, dpi, headings, smoothed_headings, speeds)
        viz_files.append((overview_file, "Flight path overview with altitude, heading, and speed analysis"))
        
        # 2. Create segmentation analysis plots
        seg_analysis_file = self._create_segmentation_analysis_plots(df, segments, output_dir, figure_size, dpi)
        viz_files.append((seg_analysis_file, "Flight path segmentation analysis with segment boundaries and statistics"))
        
        # 3. Create UTM verification plot
        utm_file = self._create_utm_verification_plot(segments, output_dir, figure_size, dpi) 
        viz_files.append((utm_file, "UTM coordinate verification plot showing final segmented flight paths"))
        
        return viz_files
    
    def _create_overview_plots(self, df: pd.DataFrame, output_dir: Path, 
                             figure_size: Tuple[int, int], dpi: int, 
                             headings: np.ndarray, smoothed_headings: np.ndarray, 
                             speeds: np.ndarray) -> Path:
        """Create overview plots of the flight path"""
        
        fig, axes = plt.subplots(2, 2, figsize=figure_size)
        fig.suptitle("Flight Path Overview", fontsize=16, fontweight='bold')
        
        # 1. Latitude vs Longitude
        axes[0, 0].plot(df['Longitude [Decimal Degrees]'], df['Latitude [Decimal Degrees]'], 
                       'b-', linewidth=1, alpha=0.7)
        axes[0, 0].scatter(df['Longitude [Decimal Degrees]'].iloc[0], df['Latitude [Decimal Degrees]'].iloc[0], 
                          color='green', s=100, marker='o', label='Start', zorder=5)
        axes[0, 0].scatter(df['Longitude [Decimal Degrees]'].iloc[-1], df['Latitude [Decimal Degrees]'].iloc[-1], 
                          color='red', s=100, marker='s', label='End', zorder=5)
        axes[0, 0].set_xlabel('Longitude (°)')
        axes[0, 0].set_ylabel('Latitude (°)')
        axes[0, 0].set_title('Flight Path (Lat/Lon)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Altitude vs Time
        time_hours = (df['Timestamp [ms]'] - df['Timestamp [ms]'].iloc[0]) / (1000 * 3600)
        axes[0, 1].plot(time_hours, df['Altitude [m]'], 'g-', linewidth=1)
        axes[0, 1].set_xlabel('Time (hours)')
        axes[0, 1].set_ylabel('Altitude (m)')
        axes[0, 1].set_title('Altitude vs Time')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Heading vs Time
        axes[1, 0].plot(time_hours, headings, 'r-', alpha=0.3, linewidth=0.5, label='Raw')
        axes[1, 0].plot(time_hours, smoothed_headings, 'r-', linewidth=2, label='Smoothed')
        axes[1, 0].set_xlabel('Time (hours)')
        axes[1, 0].set_ylabel('Heading (°)')
        axes[1, 0].set_title('Heading vs Time')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim(0, 360)
        
        # 4. Speed vs Time
        axes[1, 1].plot(time_hours, speeds, 'm-', linewidth=1)
        axes[1, 1].set_xlabel('Time (hours)')
        axes[1, 1].set_ylabel('Speed (m/s)')
        axes[1, 1].set_title('Speed vs Time')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        overview_path = output_dir / "01_flight_overview.png"
        plt.savefig(overview_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        return overview_path
    
    def _create_segmentation_analysis_plots(self, df: pd.DataFrame, segments: Dict[str, List[pd.DataFrame]], 
                                          output_dir: Path, figure_size: Tuple[int, int], dpi: int) -> Path:
        """Create segmentation analysis plots"""
        
        fig, axes = plt.subplots(2, 2, figsize=figure_size)
        fig.suptitle("Flight Path Segmentation Analysis", fontsize=16, fontweight='bold')
        
        # Calculate time for plotting
        time_hours = (df['Timestamp [ms]'] - df['Timestamp [ms]'].iloc[0]) / (1000 * 3600)
        
        # 1. UTM Path with segments (plot each parallel line separately)
        colors = plt.cm.Set1(np.linspace(0, 1, len(segments)))
        axes[0, 0].plot(df['UTM_Easting'], df['UTM_Northing'], 'lightgray', linewidth=1, alpha=0.5, label='Full path')
        
        for i, (seg_name, seg_lines) in enumerate(segments.items()):
            for j, seg_data in enumerate(seg_lines):
                if not seg_data.empty:
                    label = f'{seg_name}' if j == 0 else None  # Only label first line of each direction
                    axes[0, 0].plot(seg_data['UTM_Easting'], seg_data['UTM_Northing'], 
                                   color=colors[i % len(colors)], linewidth=2, label=label, alpha=0.8)
        
        axes[0, 0].set_xlabel('UTM Easting (m)')
        axes[0, 0].set_ylabel('UTM Northing (m)')
        axes[0, 0].set_title('UTM Flight Path - Segmented (Parallel Lines)')
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axis('equal')
        
        # 2. Heading with segment boundaries
        headings = self._calculate_heading(df)
        smoothed_headings = self._smooth_heading(headings, 51)
        
        axes[0, 1].plot(time_hours, smoothed_headings, 'k-', linewidth=1, alpha=0.7, label='Heading')
        
        # Mark segment boundaries for each parallel line
        for i, (seg_name, seg_lines) in enumerate(segments.items()):
            for j, seg_data in enumerate(seg_lines):
                if not seg_data.empty:
                    start_time = (seg_data['Timestamp [ms]'].iloc[0] - df['Timestamp [ms]'].iloc[0]) / (1000 * 3600)
                    end_time = (seg_data['Timestamp [ms]'].iloc[-1] - df['Timestamp [ms]'].iloc[0]) / (1000 * 3600)
                    label = f'{seg_name}' if j == 0 else None  # Only label first line of each direction
                    axes[0, 1].axvspan(start_time, end_time, alpha=0.3, 
                                     color=colors[i % len(colors)], label=label)
        
        axes[0, 1].set_xlabel('Time (hours)')
        axes[0, 1].set_ylabel('Heading (°)')
        axes[0, 1].set_title('Heading with Segment Boundaries')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(0, 360)
        
        # 3. Segment statistics (count parallel lines)
        seg_stats = []
        for seg_name, seg_lines in segments.items():
            total_points = 0
            all_headings = []
            
            for seg_data in seg_lines:
                if not seg_data.empty:
                    seg_headings = self._calculate_heading(seg_data)
                    all_headings.extend(seg_headings)
                    total_points += len(seg_data)
            
            if all_headings:
                # Use circular statistics for headings
                x = np.mean(np.cos(np.radians(all_headings)))
                y = np.mean(np.sin(np.radians(all_headings)))
                avg_heading = np.degrees(np.arctan2(y, x)) % 360
                heading_std = np.std(all_headings)
                
                seg_stats.append({
                    'Segment': f'{seg_name}\n({len(seg_lines)} lines)',
                    'Points': total_points,
                    'Avg Heading': avg_heading,
                    'Heading Std': heading_std
                })
        
        if seg_stats:
            stats_df = pd.DataFrame(seg_stats)
            axes[1, 0].bar(stats_df['Segment'], stats_df['Points'], color=colors[:len(stats_df)])
            axes[1, 0].set_xlabel('Segment')
            axes[1, 0].set_ylabel('Number of Points')
            axes[1, 0].set_title('Total Points per Direction')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Heading distribution
            axes[1, 1].bar(stats_df['Segment'], stats_df['Avg Heading'], 
                          yerr=stats_df['Heading Std'], color=colors[:len(stats_df)], alpha=0.7)
            axes[1, 1].set_xlabel('Segment')
            axes[1, 1].set_ylabel('Average Heading (°)')
            axes[1, 1].set_title('Average Heading per Direction')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_ylim(0, 360)
        
        plt.tight_layout()
        segmentation_path = output_dir / "02_segmentation_analysis.png"
        plt.savefig(segmentation_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        return segmentation_path
    
    def _create_utm_verification_plot(self, segments: Dict[str, List[pd.DataFrame]], 
                                    output_dir: Path, figure_size: Tuple[int, int], dpi: int) -> Path:
        """Create UTM verification plot for final segments showing parallel lines"""
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(segments)))
        
        total_lines = 0
        for i, (seg_name, seg_lines) in enumerate(segments.items()):
            total_points = sum(len(seg_data) for seg_data in seg_lines if not seg_data.empty)
            
            for j, seg_data in enumerate(seg_lines):
                if not seg_data.empty:
                    # Only label the first line of each direction
                    label = f'{seg_name} ({len(seg_lines)} lines, {total_points} pts)' if j == 0 else None
                    
                    ax.plot(seg_data['UTM_Easting'], seg_data['UTM_Northing'], 
                           color=colors[i % len(colors)], linewidth=2, 
                           marker='o', markersize=1, label=label, alpha=0.8)
                    
                    # Mark start and end of each parallel line
                    ax.scatter(seg_data['UTM_Easting'].iloc[0], seg_data['UTM_Northing'].iloc[0], 
                              color=colors[i % len(colors)], s=50, marker='o', edgecolor='black', 
                              linewidth=1, zorder=5, alpha=0.8)
                    ax.scatter(seg_data['UTM_Easting'].iloc[-1], seg_data['UTM_Northing'].iloc[-1], 
                              color=colors[i % len(colors)], s=50, marker='s', edgecolor='black', 
                              linewidth=1, zorder=5, alpha=0.8)
                    
                    total_lines += 1
        
        ax.set_xlabel('UTM Easting (m)', fontsize=12)
        ax.set_ylabel('UTM Northing (m)', fontsize=12)
        ax.set_title(f'Final Segmented Flight Paths - {total_lines} Parallel Lines (UTM Coordinates)', 
                    fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        # Add grid reference and explanation
        info_text = ('Grid Reference: UTM Zone 33N\n'
                    '○ = Line start, □ = Line end\n'
                    'Each direction contains multiple parallel lines')
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
               fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
               verticalalignment='top')
        
        plt.tight_layout()
        utm_path = output_dir / "03_utm_verification.png"
        plt.savefig(utm_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        return utm_path
    
    def _create_summary_report(self, original_df: pd.DataFrame, segments: Dict[str, List[pd.DataFrame]], 
                             output_dir: Path, params: Dict[str, Any]) -> Path:
        """Create a summary report of the segmentation"""
        
        report_path = output_dir / "segmentation_summary.txt"
        
        with open(report_path, 'w') as f:
            f.write("Flight Path Segmentation Summary\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Configuration
            seg_params = params.get('segmentation_parameters', {})
            f.write("Configuration:\n")
            f.write(f"  Heading window size: {seg_params.get('heading_window_size', {}).get('value', 'N/A')}\n")
            f.write(f"  Minimum segment length: {seg_params.get('min_segment_length', {}).get('value', 'N/A')}\n")
            f.write(f"  Heading tolerance: {seg_params.get('heading_tolerance', {}).get('value', 'N/A')}°\n")
            f.write(f"  Stability window: {seg_params.get('stability_window', {}).get('value', 'N/A')}\n")
            f.write(f"  Min stable duration: {seg_params.get('min_stable_duration', {}).get('value', 'N/A')}\n")
            f.write(f"  Heading change threshold: {seg_params.get('heading_change_threshold', {}).get('value', 'N/A')}°\n\n")
            
            f.write("Original Data:\n")
            f.write(f"  Total points: {len(original_df)}\n")
            f.write(f"  Duration: {(original_df['Timestamp [ms]'].iloc[-1] - original_df['Timestamp [ms]'].iloc[0])/60000:.1f} minutes\n\n")
            
            f.write("Segmented Data:\n")
            total_segmented_points = sum(len(seg) for seg_list in segments.values() for seg in seg_list)
            f.write(f"  Number of directions: {len(segments)}\n")
            f.write(f"  Total segmented points: {total_segmented_points}\n")
            f.write(f"  Data retention: {total_segmented_points/len(original_df)*100:.1f}%\n\n")
            
            for direction, seg_lines in segments.items():
                total_direction_points = sum(len(seg) for seg in seg_lines)
                f.write(f"  Direction '{direction}' ({len(seg_lines)} parallel lines):\n")
                f.write(f"    Total points: {total_direction_points}\n")
                
                for i, segment_df in enumerate(seg_lines):
                    if not segment_df.empty:
                        headings = self._calculate_heading(segment_df)
                        avg_heading = np.mean(headings)
                        heading_std = np.std(headings)
                        duration = (segment_df['Timestamp [ms]'].iloc[-1] - segment_df['Timestamp [ms]'].iloc[0]) / 60000
                        
                        f.write(f"    Line {i+1}:\n")
                        f.write(f"      Points: {len(segment_df)}\n")
                        f.write(f"      Duration: {duration:.1f} minutes\n")
                        f.write(f"      Average heading: {avg_heading:.1f}° ± {heading_std:.1f}°\n")
                        f.write(f"      UTM extent: E {segment_df['UTM_Easting'].min():.0f}-{segment_df['UTM_Easting'].max():.0f}, "
                               f"N {segment_df['UTM_Northing'].min():.0f}-{segment_df['UTM_Northing'].max():.0f}\n")
                f.write("\n")
        
        return report_path
    
    def _get_segment_statistics(self, segments: Dict[str, List[pd.DataFrame]]) -> Dict[str, Any]:
        """Get comprehensive statistics about the segments"""
        stats = {}
        for direction, seg_lines in segments.items():
            total_points = sum(len(seg) for seg in seg_lines)
            all_headings = []
            for seg in seg_lines:
                if not seg.empty:
                    seg_headings = self._calculate_heading(seg)
                    all_headings.extend(seg_headings)
            
            if all_headings:
                x = np.mean(np.cos(np.radians(all_headings)))
                y = np.mean(np.sin(np.radians(all_headings)))
                avg_heading = np.degrees(np.arctan2(y, x)) % 360
                
                stats[direction] = {
                    'parallel_lines': len(seg_lines),
                    'total_points': total_points,
                    'average_heading': avg_heading,
                    'heading_std': np.std(all_headings)
                }
        return stats
    
    def _add_layer_outputs(self, result: ProcessingResult, df: pd.DataFrame, 
                         segments: Dict[str, List[pd.DataFrame]], input_file_path: Optional[str] = None) -> None:
        """Add layer outputs for map visualization integration"""
        
        from loguru import logger
        import datetime
        
        # 1. Add original flight path as a vector layer
        logger.info("Creating original flight path vector layer")
        # Sort by timestamp to ensure proper chronological order and avoid jitter
        df_sorted = df.sort_values('Timestamp [ms]').copy()
        original_flight_df = df_sorted[['Latitude [Decimal Degrees]', 'Longitude [Decimal Degrees]']].copy()
        original_flight_df = original_flight_df.rename(columns={
            'Latitude [Decimal Degrees]': 'latitude',
            'Longitude [Decimal Degrees]': 'longitude'
        })
        
        # Add required line_id column for vector rendering
        original_flight_df['line_id'] = 0  # Single line for the complete flight path
        original_flight_df['segment_name'] = 'Original Flight'
        
        # Reset index to ensure clean line rendering
        original_flight_df = original_flight_df.reset_index(drop=True)
        
        # Create unique layer names based on input filename and timestamp
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        if input_file_path:
            input_filename = Path(input_file_path).stem
            original_layer_name = f'{input_filename} - Original Flight Path ({timestamp})'
        else:
            original_layer_name = f'Original Flight Path ({timestamp})'
        
        result.add_layer_output(
            layer_type="flight_path",
            data=original_flight_df,
            style_info={
                'line_color': '#666666',
                'line_width': 1,
                'line_opacity': 0.7,
                'line_style': 'solid'
            },
            metadata={
                'description': 'Complete original flight path',
                'layer_name': original_layer_name,
                'total_points': len(original_flight_df)
            }
        )
        
        # 2. Add each heading direction as separate vector layers
        logger.info(f"Creating {len(segments)} heading direction layers")
        for direction, seg_lines in segments.items():
            if not seg_lines:
                continue
                
            # Combine all parallel lines for this heading direction
            combined_lines = []
            total_points = 0
            
            for i, segment_df in enumerate(seg_lines):
                if segment_df.empty:
                    continue
                
                # Sort the segment by timestamp to ensure proper chronological order
                segment_df_sorted = segment_df.sort_values('Timestamp [ms]').copy()
                    
                # Create line data for this segment
                line_df = segment_df_sorted[['Latitude [Decimal Degrees]', 'Longitude [Decimal Degrees]']].copy()
                line_df = line_df.rename(columns={
                    'Latitude [Decimal Degrees]': 'latitude',
                    'Longitude [Decimal Degrees]': 'longitude'
                })
                
                # Add line identifier for multi-line rendering (each parallel line gets unique ID)
                line_df['line_id'] = i
                line_df['segment_name'] = direction
                
                # Reset index to ensure clean line rendering
                line_df = line_df.reset_index(drop=True)
                
                combined_lines.append(line_df)
                total_points += len(line_df)
            
            if combined_lines:
                # Combine all lines for this direction
                direction_df = pd.concat(combined_lines, ignore_index=True)
                
                # Create distinct colors for each direction
                colors = ['#FF6600', '#0066CC', '#00CC66', '#CC0066', '#CCCC00', '#CC6600']
                direction_index = list(segments.keys()).index(direction)
                color = colors[direction_index % len(colors)]
                
                # Create unique layer name for each heading direction
                if input_file_path:
                    segment_layer_name = f'{input_filename} - Flight Lines - {direction} ({timestamp})'
                else:
                    segment_layer_name = f'Flight Lines - {direction} ({timestamp})'
                
                result.add_layer_output(
                    layer_type="flight_lines",
                    data=direction_df,
                    style_info={
                        'line_color': color,
                        'line_width': 2,
                        'line_opacity': 0.9,
                        'line_style': 'solid',
                        'show_labels': True,
                        'label_field': 'segment_name'
                    },
                    metadata={
                        'description': f'Flight lines for {direction}',
                        'layer_name': segment_layer_name,
                        'parallel_lines': len(seg_lines),
                        'total_points': total_points,
                        'direction': direction
                    }
                )
        
        logger.info(f"Created {len(segments) + 1} vector layers for flight path visualization")


# Export the script class for discovery
SCRIPT_CLASS = FlightPathSegmenter