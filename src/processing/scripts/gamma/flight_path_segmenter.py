"""
Gamma Flight Path Segmentation Script for UXO Wizard Framework
============================================================

Processes gamma survey data to automatically segment flight paths into 
directional segments based on heading consistency. Adapted from magnetic 
flight_path_segmenter.py to handle gamma-specific data structure.

Features:
- Automatic detection of straight-line flight segments
- Removal of start/end portions and turning segments  
- Separation by heading direction for grid pattern surveys
- Comprehensive visualization of flight path and segmentation
- Gamma-specific data filtering and quality control
- Integration with UXO Wizard's processing pipeline

Author: Adapted for UXO Wizard Framework - Gamma Channel
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
import utm

from ...base import ScriptInterface, ProcessingResult, ProcessingError


class GammaFlightPathSegmenter(ScriptInterface):
    """
    Flight path segmentation script for gamma survey data processing
    """
    
    @property
    def name(self) -> str:
        return "Gamma Flight Path Segmenter"
    
    @property  
    def description(self) -> str:
        return "Automatically segment gamma flight paths into directional segments based on heading consistency for grid pattern surveys"
    
    def get_parameters(self) -> Dict[str, Any]:
        """Return parameter structure for gamma flight path segmentation"""
        return {
            'segmentation_parameters': {
                'heading_window_size': {
                    'value': 31,
                    'type': 'int',
                    'min': 5,
                    'max': 201,
                    'description': 'Window size for heading smoothing (odd number recommended)'
                },
                'min_segment_length': {
                    'value': 50,
                    'type': 'int', 
                    'min': 10,
                    'max': 1000,
                    'description': 'Minimum points per segment'
                },
                'heading_tolerance': {
                    'value': 15.0,
                    'type': 'float',
                    'min': 1.0,
                    'max': 45.0,
                    'description': 'Degrees tolerance for consistent heading'
                },
                'stability_window': {
                    'value': 100,
                    'type': 'int',
                    'min': 20,
                    'max': 500,
                    'description': 'Points to analyze for path stability'
                },
                'min_stable_duration': {
                    'value': 200,
                    'type': 'int',
                    'min': 50,
                    'max': 5000,
                    'description': 'Minimum points for a stable segment'
                },
                'heading_change_threshold': {
                    'value': 20.0,
                    'type': 'float',
                    'min': 1.0,
                    'max': 45.0,
                    'description': 'Max heading change rate for stability (deg/point)'
                },
                'min_points_for_valid_line': {
                    'value': 30,
                    'type': 'int',
                    'min': 10,
                    'max': 500,
                    'description': 'Minimum points required for a valid flight line'
                }
            },
            'quality_control': {
                'remove_start_points': {
                    'value': 20,
                    'type': 'int',
                    'min': 0,
                    'max': 1000,
                    'description': 'Points to remove from start of each segment'
                },
                'remove_end_points': {
                    'value': 20,
                    'type': 'int',
                    'min': 0,
                    'max': 1000,
                    'description': 'Points to remove from end of each segment'
                },
                'min_speed_kmh': {
                    'value': 1.0,
                    'type': 'float',
                    'min': 0.0,
                    'max': 100.0,
                    'description': 'Minimum speed for valid data points (km/h)'
                },
                'max_speed_kmh': {
                    'value': 300.0,
                    'type': 'float',
                    'min': 10.0,
                    'max': 500.0,
                    'description': 'Maximum speed for valid data points (km/h)'
                },
                'gamma_quality_filters': {
                    'value': False,
                    'type': 'bool',
                    'description': 'Apply gamma-specific quality filters'
                },
                'min_total_counts': {
                    'value': 0,
                    'type': 'int',
                    'min': 0,
                    'max': 1000,
                    'description': 'Minimum total counts for valid gamma reading'
                },
                'max_total_counts': {
                    'value': 50000,
                    'type': 'int',
                    'min': 100,
                    'max': 100000,
                    'description': 'Maximum total counts for valid gamma reading'
                }
            },
            'heading_combination': {
                'combine_opposite_headings': {
                    'value': False,
                    'type': 'bool',
                    'description': 'Combine segments with opposite headings (e.g., N+S, E+W)'
                },
                'heading_difference_threshold': {
                    'value': 160.0,
                    'type': 'float',
                    'min': 120.0,
                    'max': 180.0,
                    'description': 'Minimum degrees difference to consider headings opposite'
                }
            },
            'visualization_settings': {
                'figure_size_width': {
                    'value': 15,
                    'type': 'int',
                    'min': 8,
                    'max': 30,
                    'description': 'Figure width in inches'
                },
                'figure_size_height': {
                    'value': 10,
                    'type': 'int',
                    'min': 6,
                    'max': 20,
                    'description': 'Figure height in inches'
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
        """Validate that data has required columns for gamma flight path segmentation"""
        if data.empty:
            raise ProcessingError("Input data cannot be empty")
        
        # Required columns for gamma flight path segmentation
        required_columns = [
            'timestamp', 'lat', 'lon', 'Height'
        ]
        
        # Optional but recommended gamma columns
        gamma_columns = ['Total', 'Countrate', 'U238', 'K40', 'Th232', 'Cs137']
        
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            raise ProcessingError(f"Missing required columns for gamma flight path segmentation: {missing_cols}")
        
        # Check for gamma columns
        available_gamma_cols = [col for col in gamma_columns if col in data.columns]
        if not available_gamma_cols:
            raise ProcessingError("No gamma measurement columns found. Expected at least one of: " + 
                                ", ".join(gamma_columns))
        
        # Check for minimum data points
        if len(data) < 100:
            raise ProcessingError("Need at least 100 data points for meaningful flight path segmentation")
        
        return True
    
    def execute(self, data: pd.DataFrame, params: Dict[str, Any], 
                progress_callback: Optional[Callable] = None, 
                input_file_path: Optional[str] = None) -> ProcessingResult:
        """Execute gamma flight path segmentation processing"""
        
        try:
            if progress_callback:
                progress_callback(0, "Starting gamma flight path segmentation...")
            
            # Extract parameters
            seg_params = params.get('segmentation_parameters', {})
            qc_params = params.get('quality_control', {})
            heading_params = params.get('heading_combination', {})
            viz_params = params.get('visualization_settings', {})
            
            # Create working copy of data
            df = data.copy()
            
            # Sort data by timestamp to ensure proper order for speed/heading calculations
            if progress_callback:
                progress_callback(5, "Sorting data by timestamp...")
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Convert lat/lon to UTM if needed
            if 'UTM_Easting' not in df.columns or 'UTM_Northing' not in df.columns:
                if progress_callback:
                    progress_callback(8, "Converting coordinates to UTM...")
                df = self._convert_to_utm(df)
            
            # Apply gamma-specific quality filters
            if qc_params.get('gamma_quality_filters', {}).get('value', True):
                if progress_callback:
                    progress_callback(10, "Applying gamma quality filters...")
                df = self._apply_gamma_quality_filters(df, qc_params)
            
            if progress_callback:
                progress_callback(15, "Calculating flight metrics...")
            
            # Calculate headings and speeds
            headings = self._calculate_heading(df)
            speeds = self._calculate_speed(df)
            
            # Debug output for speed calculation
            print(f"Speed calculation results:")
            print(f"  Min speed: {speeds.min():.2f} km/h")
            print(f"  Max speed: {speeds.max():.2f} km/h")
            print(f"  Mean speed: {speeds.mean():.2f} km/h")
            print(f"  Median speed: {np.median(speeds):.2f} km/h")
            
            # Check for potential data issues
            very_fast = np.sum(speeds > 200)
            very_slow = np.sum(speeds < 1)
            print(f"  Points > 200 km/h: {very_fast}")
            print(f"  Points < 1 km/h: {very_slow}")
            
            # Time span analysis
            time_span = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]) / 1000.0 / 60.0  # minutes
            print(f"  Total time span: {time_span:.1f} minutes")
            print(f"  Data points: {len(df)}")
            print(f"  Average sampling rate: {len(df) / time_span:.1f} points/minute")
            
            if progress_callback:
                progress_callback(25, "Smoothing heading data...")
            
            # Smooth headings
            window_size = seg_params.get('heading_window_size', {}).get('value', 51)
            smoothed_headings = self._smooth_heading(headings, window_size)
            
            if progress_callback:
                progress_callback(35, "Detecting stabilized segments...")
            
            # Detect stabilized segments
            segments = self._detect_stabilized_segments(
                df, smoothed_headings, speeds, seg_params, qc_params
            )
            
            if not segments:
                # Provide more detailed error information
                error_msg = f"No stabilized flight segments detected. Data summary:\n"
                error_msg += f"- Total points: {len(df)}\n"
                error_msg += f"- Speed range: {speeds.min():.1f} - {speeds.max():.1f} km/h\n"
                error_msg += f"- Heading range: {smoothed_headings.min():.1f} - {smoothed_headings.max():.1f}°\n"
                error_msg += f"- Current parameters:\n"
                error_msg += f"  - Min stable duration: {seg_params.get('min_stable_duration', {}).get('value', 'N/A')}\n"
                error_msg += f"  - Heading tolerance: {seg_params.get('heading_tolerance', {}).get('value', 'N/A')}°\n"
                error_msg += f"  - Speed range: {qc_params.get('min_speed_kmh', {}).get('value', 'N/A')}-{qc_params.get('max_speed_kmh', {}).get('value', 'N/A')} km/h\n"
                error_msg += "Try reducing min_stable_duration or increasing heading_tolerance."
                raise ProcessingError(error_msg)
            
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
                data=df,  # Return processed data as primary result
                processing_script=self.name,
                metadata={
                    'processor': 'gamma',
                    'total_segments': len(final_segments),
                    'segment_details': self._get_segment_statistics(final_segments),
                    'original_points': len(data),
                    'processed_points': len(df),
                    'segmented_points': sum(len(seg) for seg_list in final_segments.values() for seg in seg_list),
                    'parameters': params,
                    'gamma_channels': [col for col in ['Total', 'Countrate', 'U238', 'K40', 'Th232', 'Cs137'] 
                                     if col in df.columns]
                }
            )
            
            # Add segmented CSV files as outputs
            for file_path in saved_files:
                result.add_output_file(
                    file_path=str(file_path),
                    file_type="csv",
                    description=f"Segmented gamma flight data: {file_path.stem}"
                )
            
            if progress_callback:
                progress_callback(80, "Creating visualizations...")
            
            # Create visualizations if enabled
            if viz_params.get('create_visualizations', {}).get('value', True):
                viz_files = self._create_visualizations(
                    df, final_segments, output_dir, viz_params, 
                    headings, smoothed_headings, speeds
                )
                
                # Add visualization files to result
                for file_path, description in viz_files:
                    result.add_output_file(
                        file_path=str(file_path),
                        file_type="png",
                        description=description
                    )
            
            if progress_callback:
                progress_callback(90, "Creating summary report...")
            
            # Generate summary report
            summary_path = self._create_summary_report(data, final_segments, output_dir, params)
            result.add_output_file(
                file_path=str(summary_path),
                file_type="txt",
                description="Flight path segmentation summary report"
            )
            
            # Add layer outputs for future layer system
            self._add_layer_outputs(result, df, final_segments, input_file_path)
            
            if progress_callback:
                progress_callback(100, "Gamma flight path segmentation completed!")
            
            return result
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                error_message=f"Gamma flight path segmentation failed: {str(e)}"
            )
    
    def _convert_to_utm(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert lat/lon coordinates to UTM"""
        utm_eastings = []
        utm_northings = []
        
        for _, row in df.iterrows():
            try:
                # Convert to UTM (automatically determines zone)
                easting, northing, zone_num, zone_letter = utm.from_latlon(row['lat'], row['lon'])
                utm_eastings.append(easting)
                utm_northings.append(northing)
            except Exception:
                # If conversion fails, use NaN
                utm_eastings.append(np.nan)
                utm_northings.append(np.nan)
        
        df['UTM_Easting'] = utm_eastings
        df['UTM_Northing'] = utm_northings
        
        # Remove any rows with NaN UTM coordinates
        df = df.dropna(subset=['UTM_Easting', 'UTM_Northing'])
        
        return df
    
    def _apply_gamma_quality_filters(self, df: pd.DataFrame, qc_params: Dict) -> pd.DataFrame:
        """Apply gamma-specific quality filters"""
        original_len = len(df)
        
        # Filter by total counts if available
        if 'Total' in df.columns:
            min_total = qc_params.get('min_total_counts', {}).get('value', 5)
            max_total = qc_params.get('max_total_counts', {}).get('value', 10000)
            df = df[(df['Total'] >= min_total) & (df['Total'] <= max_total)]
        
        # Remove rows with invalid gamma measurements (negative values except for backgrounds)
        gamma_cols = ['U238', 'K40', 'Th232', 'Cs137']
        for col in gamma_cols:
            if col in df.columns:
                # Allow reasonable negative values for background subtraction
                df = df[df[col] > -1000]  # Very loose filter to remove obvious errors
        
        # Remove rows with invalid environmental data
        if 'Height' in df.columns:
            df = df[df['Height'] > 0]  # Height should be positive
        
        filtered_len = len(df)
        if filtered_len < original_len * 0.5:  # If we removed more than 50% of data
            raise ProcessingError(f"Gamma quality filters removed too much data ({original_len} → {filtered_len} points)")
        
        return df
    
    def _calculate_heading(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate heading from consecutive UTM coordinates"""
        eastings = df['UTM_Easting'].values
        northings = df['UTM_Northing'].values
        
        # Calculate differences
        d_east = np.diff(eastings)
        d_north = np.diff(northings)
        
        # Calculate headings (in degrees, 0 = North)
        headings = np.arctan2(d_east, d_north) * 180 / np.pi
        headings = (headings + 360) % 360  # Ensure 0-360 range
        
        # Extend to match original length (duplicate last heading)
        headings = np.append(headings, headings[-1])
        
        return headings
    
    def _calculate_speed(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate speed from consecutive positions and timestamps"""
        eastings = df['UTM_Easting'].values
        northings = df['UTM_Northing'].values
        timestamps = df['timestamp'].values
        
        # Calculate distances
        d_east = np.diff(eastings)
        d_north = np.diff(northings)
        distances = np.sqrt(d_east**2 + d_north**2)
        
        # Calculate time differences (convert ms to seconds)
        d_time = np.diff(timestamps) / 1000.0
        
        # Filter out unrealistic time differences
        # Minimum time step should be > 0.01 seconds (100ms)
        # Maximum time step should be < 60 seconds
        d_time = np.where(d_time < 0.01, 0.01, d_time)  # Minimum 10ms
        d_time = np.where(d_time > 60.0, 60.0, d_time)   # Maximum 60s
        
        # Calculate speeds (m/s) and convert to km/h
        speeds = (distances / d_time) * 3.6
        
        # Apply reasonable speed limits to catch calculation errors
        # Cap speeds at 500 km/h (very generous for aircraft)
        speeds = np.where(speeds > 500.0, 500.0, speeds)
        speeds = np.where(speeds < 0, 0, speeds)
        
        # Smooth speeds to reduce noise
        if len(speeds) > 5:
            from scipy.ndimage import uniform_filter1d
            speeds = uniform_filter1d(speeds, size=min(5, len(speeds)))
        
        # Extend to match original length
        speeds = np.append(speeds, speeds[-1])
        
        return speeds
    
    def _smooth_heading(self, headings: np.ndarray, window_size: int) -> np.ndarray:
        """Smooth headings using circular mean"""
        if window_size <= 1:
            return headings
        
        # Convert to radians for circular statistics
        headings_rad = headings * np.pi / 180
        
        # Convert to complex numbers for circular averaging
        complex_headings = np.exp(1j * headings_rad)
        
        # Apply uniform filter for smoothing
        smoothed_complex = uniform_filter1d(complex_headings.real, size=window_size) + \
                          1j * uniform_filter1d(complex_headings.imag, size=window_size)
        
        # Convert back to angles
        smoothed_rad = np.angle(smoothed_complex)
        smoothed_headings = (smoothed_rad * 180 / np.pi + 360) % 360
        
        return smoothed_headings
    
    def _detect_stabilized_segments(self, df: pd.DataFrame, smoothed_headings: np.ndarray, 
                                  speeds: np.ndarray, seg_params: Dict, qc_params: Dict) -> List[Tuple[int, int, float]]:
        """Detect segments with stable heading and appropriate speed"""
        
        # Extract parameters
        min_segment_length = seg_params.get('min_segment_length', {}).get('value', 50)
        heading_tolerance = seg_params.get('heading_tolerance', {}).get('value', 15.0)
        stability_window = seg_params.get('stability_window', {}).get('value', 100)
        min_stable_duration = seg_params.get('min_stable_duration', {}).get('value', 200)
        heading_change_threshold = seg_params.get('heading_change_threshold', {}).get('value', 20.0)
        
        min_speed = qc_params.get('min_speed_kmh', {}).get('value', 1.0)
        max_speed = qc_params.get('max_speed_kmh', {}).get('value', 300.0)
        
        segments = []
        i = 0
        potential_segments = 0
        speed_filtered = 0
        heading_filtered = 0
        length_filtered = 0
        
        while i < len(smoothed_headings) - min_segment_length:
            # Check if current section has stable heading
            window_end = min(i + stability_window, len(smoothed_headings))
            window_headings = smoothed_headings[i:window_end]
            window_speeds = speeds[i:window_end]
            
            # More lenient speed criteria - check if any speeds are reasonable
            valid_speeds = (window_speeds >= min_speed) & (window_speeds <= max_speed)
            if np.sum(valid_speeds) < len(window_speeds) * 0.5:  # Reduced to 50% of points must have valid speed
                speed_filtered += 1
                i += 1
                continue
            
            # More lenient heading stability check
            heading_std = np.std(window_headings)
            if heading_std > heading_tolerance:
                heading_filtered += 1
                i += 1
                continue
            
            potential_segments += 1
            
            # Found start of stable segment - extend it
            segment_start = i
            current_heading = np.mean(window_headings)
            
            # Extend segment as long as heading remains reasonably stable
            j = window_end
            consecutive_bad_points = 0
            max_consecutive_bad = 10  # Allow some noise in the data
            
            while j < len(smoothed_headings) and consecutive_bad_points < max_consecutive_bad:
                heading_diff = abs(smoothed_headings[j] - current_heading)
                # Check for heading wrap-around (e.g., 359° to 1°)
                heading_diff = min(heading_diff, 360 - heading_diff)
                
                if heading_diff > heading_tolerance:
                    consecutive_bad_points += 1
                else:
                    consecutive_bad_points = 0  # Reset counter on good point
                
                # More lenient speed check - allow temporary speed anomalies
                if not (min_speed <= speeds[j] <= max_speed):
                    consecutive_bad_points += 1
                else:
                    consecutive_bad_points = max(0, consecutive_bad_points - 1)  # Reduce counter on good speed
                
                j += 1
            
            # Back up to exclude the bad points at the end
            segment_end = j - consecutive_bad_points
            segment_length = segment_end - segment_start
            
            # Check if segment meets minimum requirements
            if segment_length >= min_stable_duration:
                segments.append((segment_start, segment_end, current_heading))
            else:
                length_filtered += 1
            
            i = max(segment_end, i + 1)  # Ensure we always advance
        
        print(f"Segment detection summary:")
        print(f"  Potential segments found: {potential_segments}")
        print(f"  Filtered by speed: {speed_filtered}")
        print(f"  Filtered by heading stability: {heading_filtered}")
        print(f"  Filtered by length: {length_filtered}")
        print(f"  Final segments: {len(segments)}")
        
        return segments
    
    def _group_segments_by_heading(self, segments: List[Tuple[int, int, float]], 
                                 df: pd.DataFrame, seg_params: Dict) -> Dict[str, List[pd.DataFrame]]:
        """Group segments by cardinal direction"""
        
        min_points = seg_params.get('min_points_for_valid_line', {}).get('value', 50)
        remove_start = seg_params.get('quality_control', {}).get('remove_start_points', {}).get('value', 100)
        remove_end = seg_params.get('quality_control', {}).get('remove_end_points', {}).get('value', 100)
        
        direction_segments = {}
        
        for start_idx, end_idx, heading in segments:
            # Apply start/end point removal
            trimmed_start = start_idx + remove_start
            trimmed_end = end_idx - remove_end
            
            # Check if segment still has enough points after trimming
            if trimmed_end - trimmed_start < min_points:
                continue
            
            # Extract segment data
            segment_df = df.iloc[trimmed_start:trimmed_end].copy()
            
            # Determine cardinal direction
            direction = self._get_cardinal_direction(heading)
            
            # Add to appropriate direction group
            if direction not in direction_segments:
                direction_segments[direction] = []
            direction_segments[direction].append(segment_df)
        
        return direction_segments
    
    def _combine_opposite_headings(self, direction_segments: Dict[str, List[pd.DataFrame]], 
                                 heading_params: Dict) -> Dict[str, List[pd.DataFrame]]:
        """Combine segments with opposite headings if enabled"""
        
        if not heading_params.get('combine_opposite_headings', {}).get('value', False):
            return direction_segments
        
        combined_segments = {}
        
        # Define opposite direction pairs
        opposite_pairs = [
            (['North', 'South'], 'North-South'),
            (['East', 'West'], 'East-West'),
            (['Northeast', 'Southwest'], 'NE-SW'),
            (['Northwest', 'Southeast'], 'NW-SE')
        ]
        
        processed_directions = set()
        
        for directions, combined_name in opposite_pairs:
            combined_list = []
            
            for direction in directions:
                if direction in direction_segments and direction not in processed_directions:
                    combined_list.extend(direction_segments[direction])
                    processed_directions.add(direction)
            
            if combined_list:
                combined_segments[combined_name] = combined_list
        
        # Add any remaining uncombined directions
        for direction, segments in direction_segments.items():
            if direction not in processed_directions:
                combined_segments[direction] = segments
        
        return combined_segments
    
    def _get_cardinal_direction(self, heading: float) -> str:
        """Convert heading to cardinal direction"""
        directions = [
            (0, 22.5, 'North'),
            (22.5, 67.5, 'Northeast'),
            (67.5, 112.5, 'East'),
            (112.5, 157.5, 'Southeast'),
            (157.5, 202.5, 'South'),
            (202.5, 247.5, 'Southwest'),
            (247.5, 292.5, 'West'),
            (292.5, 337.5, 'Northwest'),
            (337.5, 360, 'North')
        ]
        
        for min_deg, max_deg, direction in directions:
            if min_deg <= heading < max_deg:
                return direction
        
        return 'North'  # Default fallback
    
    def _create_output_directory(self, input_file_path: Optional[str]) -> Path:
        """Create output directory for segmented files"""
        if input_file_path:
            input_path = Path(input_file_path)
            output_dir = input_path.parent / f"gamma_segmented_{input_path.stem}"
        else:
            output_dir = Path("gamma_segmented_output")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    def _get_base_filename(self, input_file_path: Optional[str]) -> str:
        """Get base filename for output files"""
        if input_file_path:
            return Path(input_file_path).stem
        else:
            return "gamma_data"
    
    def _save_segments(self, segments: Dict[str, List[pd.DataFrame]], 
                      output_dir: Path, base_filename: str) -> List[Path]:
        """Save segmented data to CSV files"""
        saved_files = []
        
        for direction, segment_list in segments.items():
            for i, segment_df in enumerate(segment_list):
                filename = f"{base_filename}_gamma_{direction}_{i+1:03d}.csv"
                file_path = output_dir / filename
                
                # Save to CSV
                segment_df.to_csv(file_path, index=False)
                saved_files.append(file_path)
        
        return saved_files
    
    def _create_visualizations(self, df: pd.DataFrame, segments: Dict[str, List[pd.DataFrame]], 
                             output_dir: Path, viz_params: Dict, headings: np.ndarray, 
                             smoothed_headings: np.ndarray, speeds: np.ndarray) -> List[Tuple[Path, str]]:
        """Create visualization plots"""
        
        figure_size = (
            viz_params.get('figure_size_width', {}).get('value', 15),
            viz_params.get('figure_size_height', {}).get('value', 10)
        )
        dpi = viz_params.get('dpi', {}).get('value', 300)
        
        viz_files = []
        
        # Create overview plots
        overview_path = self._create_overview_plots(
            df, output_dir, figure_size, dpi, headings, smoothed_headings, speeds
        )
        viz_files.append((overview_path, "Gamma flight path overview"))
        
        # Create segmentation analysis plots
        segmentation_path = self._create_segmentation_analysis_plots(
            df, segments, output_dir, figure_size, dpi
        )
        viz_files.append((segmentation_path, "Gamma flight path segmentation analysis"))
        
        return viz_files
    
    def _create_overview_plots(self, df: pd.DataFrame, output_dir: Path, 
                             figure_size: Tuple[int, int], dpi: int, 
                             headings: np.ndarray, smoothed_headings: np.ndarray, 
                             speeds: np.ndarray) -> Path:
        """Create overview plots of flight path and gamma data"""
        
        fig, axes = plt.subplots(2, 2, figsize=figure_size, dpi=dpi)
        
        # Plot 1: Flight path with gamma intensity
        ax1 = axes[0, 0]
        if 'Total' in df.columns:
            scatter = ax1.scatter(df['UTM_Easting'], df['UTM_Northing'], 
                                c=df['Total'], cmap='viridis', s=1, alpha=0.6)
            plt.colorbar(scatter, ax=ax1, label='Total Gamma Counts')
        else:
            ax1.scatter(df['UTM_Easting'], df['UTM_Northing'], s=1, alpha=0.6)
        ax1.set_title('Gamma Flight Path')
        ax1.set_xlabel('UTM Easting (m)')
        ax1.set_ylabel('UTM Northing (m)')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Heading analysis
        ax2 = axes[0, 1]
        ax2.plot(headings, alpha=0.3, label='Raw Heading')
        ax2.plot(smoothed_headings, label='Smoothed Heading')
        ax2.set_title('Heading Analysis')
        ax2.set_xlabel('Data Point')
        ax2.set_ylabel('Heading (degrees)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Speed analysis
        ax3 = axes[1, 0]
        ax3.plot(speeds, alpha=0.7)
        ax3.set_title('Speed Analysis')
        ax3.set_xlabel('Data Point')
        ax3.set_ylabel('Speed (km/h)')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Gamma channel overview
        ax4 = axes[1, 1]
        gamma_cols = ['U238', 'K40', 'Th232', 'Cs137']
        available_cols = [col for col in gamma_cols if col in df.columns]
        
        if available_cols:
            for col in available_cols:
                ax4.plot(df[col], label=col, alpha=0.7)
            ax4.set_title('Gamma Channels')
            ax4.set_xlabel('Data Point')
            ax4.set_ylabel('Concentration')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No gamma channels available', 
                    transform=ax4.transAxes, ha='center', va='center')
        
        plt.tight_layout()
        
        output_path = output_dir / 'gamma_flight_overview.png'
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _create_segmentation_analysis_plots(self, df: pd.DataFrame, segments: Dict[str, List[pd.DataFrame]], 
                                          output_dir: Path, figure_size: Tuple[int, int], dpi: int) -> Path:
        """Create segmentation analysis plots"""
        
        fig, axes = plt.subplots(2, 2, figsize=figure_size, dpi=dpi)
        
        # Define colors for different directions
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        # Plot 1: Segmented flight paths
        ax1 = axes[0, 0]
        color_idx = 0
        
        for direction, segment_list in segments.items():
            color = colors[color_idx % len(colors)]
            for segment in segment_list:
                ax1.plot(segment['UTM_Easting'], segment['UTM_Northing'], 
                        color=color, linewidth=2, alpha=0.7, label=direction if segment is segment_list[0] else "")
            color_idx += 1
        
        ax1.set_title('Segmented Flight Paths')
        ax1.set_xlabel('UTM Easting (m)')
        ax1.set_ylabel('UTM Northing (m)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Segment statistics
        ax2 = axes[0, 1]
        directions = list(segments.keys())
        segment_counts = [len(segments[dir]) for dir in directions]
        total_points = [sum(len(seg) for seg in segments[dir]) for dir in directions]
        
        x = np.arange(len(directions))
        width = 0.35
        
        ax2.bar(x - width/2, segment_counts, width, label='Number of Segments', alpha=0.7)
        ax2_twin = ax2.twinx()
        ax2_twin.bar(x + width/2, total_points, width, label='Total Points', alpha=0.7, color='orange')
        
        ax2.set_title('Segment Statistics')
        ax2.set_xlabel('Direction')
        ax2.set_ylabel('Number of Segments')
        ax2_twin.set_ylabel('Total Points')
        ax2.set_xticks(x)
        ax2.set_xticklabels(directions, rotation=45)
        ax2.legend(loc='upper left')
        ax2_twin.legend(loc='upper right')
        
        # Plot 3: Gamma intensity by direction
        ax3 = axes[1, 0]
        if 'Total' in df.columns:
            for direction, segment_list in segments.items():
                all_totals = []
                for segment in segment_list:
                    all_totals.extend(segment['Total'].values)
                if all_totals:
                    ax3.hist(all_totals, bins=30, alpha=0.7, label=direction)
            
            ax3.set_title('Gamma Intensity Distribution by Direction')
            ax3.set_xlabel('Total Gamma Counts')
            ax3.set_ylabel('Frequency')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No gamma intensity data available', 
                    transform=ax3.transAxes, ha='center', va='center')
        
        # Plot 4: Height distribution
        ax4 = axes[1, 1]
        if 'Height' in df.columns:
            for direction, segment_list in segments.items():
                all_heights = []
                for segment in segment_list:
                    all_heights.extend(segment['Height'].values)
                if all_heights:
                    ax4.hist(all_heights, bins=30, alpha=0.7, label=direction)
            
            ax4.set_title('Height Distribution by Direction')
            ax4.set_xlabel('Height (m)')
            ax4.set_ylabel('Frequency')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No height data available', 
                    transform=ax4.transAxes, ha='center', va='center')
        
        plt.tight_layout()
        
        output_path = output_dir / 'gamma_segmentation_analysis.png'
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _create_summary_report(self, original_df: pd.DataFrame, segments: Dict[str, List[pd.DataFrame]], 
                             output_dir: Path, params: Dict[str, Any]) -> Path:
        """Create a text summary report"""
        
        report_path = output_dir / 'gamma_segmentation_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("Gamma Flight Path Segmentation Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Processing Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Script: {self.name}\n\n")
            
            f.write("Data Summary:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Original data points: {len(original_df)}\n")
            f.write(f"Total segments found: {sum(len(seg_list) for seg_list in segments.values())}\n")
            f.write(f"Total segmented points: {sum(len(seg) for seg_list in segments.values() for seg in seg_list)}\n\n")
            
            f.write("Segments by Direction:\n")
            f.write("-" * 25 + "\n")
            for direction, segment_list in segments.items():
                f.write(f"{direction}: {len(segment_list)} segments\n")
                total_points = sum(len(seg) for seg in segment_list)
                f.write(f"  Total points: {total_points}\n")
                if segment_list:
                    avg_points = total_points / len(segment_list)
                    f.write(f"  Average points per segment: {avg_points:.1f}\n")
                f.write("\n")
            
            f.write("Gamma Channel Statistics:\n")
            f.write("-" * 30 + "\n")
            gamma_cols = ['Total', 'Countrate', 'U238', 'K40', 'Th232', 'Cs137']
            for col in gamma_cols:
                if col in original_df.columns:
                    f.write(f"{col}:\n")
                    f.write(f"  Range: {original_df[col].min():.2f} to {original_df[col].max():.2f}\n")
                    f.write(f"  Mean: {original_df[col].mean():.2f}\n")
                    f.write(f"  Std: {original_df[col].std():.2f}\n")
            
            f.write("\nProcessing Parameters:\n")
            f.write("-" * 25 + "\n")
            for category, param_dict in params.items():
                f.write(f"{category}:\n")
                for param, settings in param_dict.items():
                    if isinstance(settings, dict) and 'value' in settings:
                        f.write(f"  {param}: {settings['value']}\n")
                f.write("\n")
        
        return report_path
    
    def _get_segment_statistics(self, segments: Dict[str, List[pd.DataFrame]]) -> Dict[str, Any]:
        """Get detailed statistics about segments"""
        stats = {}
        
        for direction, segment_list in segments.items():
            direction_stats = {
                'count': len(segment_list),
                'total_points': sum(len(seg) for seg in segment_list),
                'avg_points_per_segment': 0,
                'min_points': 0,
                'max_points': 0
            }
            
            if segment_list:
                point_counts = [len(seg) for seg in segment_list]
                direction_stats['avg_points_per_segment'] = np.mean(point_counts)
                direction_stats['min_points'] = min(point_counts)
                direction_stats['max_points'] = max(point_counts)
            
            stats[direction] = direction_stats
        
        return stats
    
    def _add_layer_outputs(self, result: ProcessingResult, df: pd.DataFrame, 
                         segments: Dict[str, List[pd.DataFrame]], input_file_path: Optional[str] = None) -> None:
        """Add layer outputs for future layer system integration"""
        
        # Add original flight path as layer
        result.add_layer_output(
            layer_type="gamma_flight_path",
            data=df,
            metadata={
                'description': 'Original gamma flight path',
                'total_points': len(df),
                'data_type': 'gamma'
            }
        )
        
        # Add segmented flight paths as layers
        for direction, segment_list in segments.items():
            for i, segment in enumerate(segment_list):
                result.add_layer_output(
                    layer_type="gamma_flight_segment",
                    data=segment,
                    metadata={
                        'description': f'Gamma flight segment {direction} #{i+1}',
                        'direction': direction,
                        'segment_number': i + 1,
                        'points': len(segment),
                        'data_type': 'gamma'
                    }
                )

# Export the script class for automatic discovery
SCRIPT_CLASS = GammaFlightPathSegmenter 