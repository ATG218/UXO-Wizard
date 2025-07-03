'''"""
Magbase Processing Script - Integrated version of magbase.py for UXO-Wizard Framework

This script provides comprehensive magnetic data processing including:
- GPS data interpolation and cleaning
- UTM coordinate conversion with sensor offsets
- Diurnal correction using base station data
- Total field calculation and residual anomaly computation
"""'''

import pandas as pd
import numpy as np
import os
from typing import Dict, Any, Optional, Callable
from pathlib import Path
import pyproj
from datetime import datetime
import logging
import time
import multiprocessing as mp
from functools import partial

from src.processing.base import ScriptInterface, ProcessingResult, ProcessingError

# Top-level functions for multiprocessing to avoid pickling issues
def _find_closest_time_index(target_time, times):
    """Find closest time using binary search - exactly like original magbase.py"""
    if pd.isna(target_time):
        return None
        
    # Initial binary search to find position
    left, right = 0, len(times) - 1
    while left <= right:
        mid = (left + right) // 2
        if times[mid] < target_time:
            left = mid + 1
        else:
            right = mid - 1
    
    # Check boundaries
    pos = left
    if pos <= 0:
        return 0
    elif pos >= len(times):
        return len(times) - 1
    
    # Return the closest of the two surrounding indices
    if abs(times[pos] - target_time) < abs(times[pos-1] - target_time):
        return pos
    else:
        return pos - 1

def _process_batch_mp(batch_df, base_times, base_fields, vertical_alignment=False, disable_basestation=False):
    """Process a batch of rows to find closest base station measurements - exactly like original magbase.py"""
    result_df = batch_df.copy()
    for idx, row in batch_df.iterrows():
        try:
            local_time_numeric = row.get('MagWalk_LocalTime')
            if pd.isna(local_time_numeric):
                continue
                
            local_time_numeric = float(local_time_numeric)
            closest_idx = _find_closest_time_index(local_time_numeric, base_times)
            if closest_idx is not None:
                if not disable_basestation:
                    # Calculate residuals exactly like original magbase.py
                    earth_field = base_fields[closest_idx]
                    
                    result_df.at[idx, 'R1 [nT]'] = row['Btotal1 [nT]'] - earth_field
                    result_df.at[idx, 'R2 [nT]'] = row['Btotal2 [nT]'] - earth_field
                    
                    # Calculate vertical alignment if requested (from residuals)
                    if vertical_alignment:
                        result_df.at[idx, 'VA [nT]'] = result_df.at[idx, 'R1 [nT]'] - result_df.at[idx, 'R2 [nT]']
                elif vertical_alignment:
                    # When basestation is disabled, calculate VA directly from total fields
                    # without creating redundant R1 and R2 columns
                    result_df.at[idx, 'VA [nT]'] = row['Btotal1 [nT]'] - row['Btotal2 [nT]']
        except (ValueError, TypeError) as e:
            logging.debug(f"Error matching time {local_time_numeric}: {e}")
    return result_df


class MagbaseProcessing(ScriptInterface):
    """
    Advanced magnetic data processing script implementing magbase.py functionality
    """
    
    @property
    def name(self) -> str:
        return "Magbase Processing"
    
    @property
    def description(self) -> str:
        return "Comprehensive magnetic data processing with GPS interpolation, UTM conversion, and diurnal correction"
    
    def get_parameters(self) -> Dict[str, Any]:
        """Return comprehensive parameter structure for magbase processing"""
        return {
            'file_inputs': {
                'base_station_file': {
                    'value': '',
                    'type': 'file',
                    'file_types': ['.txt', '.csv'],
                    'description': 'GSM-19 base station data file for diurnal correction'
                },
                'base_station_delimiter': {
                    'value': ',',
                    'type': 'choice',
                    'choices': [',', ';', '	'],
                    'description': 'Base station file delimiter'
                }
            },
            'sensor_configuration': {
                'sensor1_offset_east': {
                    'value': 0.0375,
                    'type': 'float',
                    'min': -5.0,
                    'max': 5.0,
                    'description': 'Sensor 1 East offset from GPS (meters)'
                },
                'sensor1_offset_north': {
                    'value': -0.56,
                    'type': 'float',
                    'min': -5.0,
                    'max': 5.0,
                    'description': 'Sensor 1 North offset from GPS (meters)'
                },
                'sensor2_offset_east': {
                    'value': 0.0375,
                    'type': 'float',
                    'min': -5.0,
                    'max': 5.0,
                    'description': 'Sensor 2 East offset from GPS (meters)'
                },
                'sensor2_offset_north': {
                    'value': 0.56,
                    'type': 'float',
                    'min': -5.0,
                    'max': 5.0,
                    'description': 'Sensor 2 North offset from GPS (meters)'
                }
            },
            'processing_options': {
                'disable_basestation': {
                    'value': False,
                    'type': 'bool',
                    'description': 'Skip base station diurnal correction'
                },
                'vertical_alignment': {
                    'value': False,
                    'type': 'bool',
                    'description': 'Calculate vertical gradient (M1-M2)'
                },
                'utm_zone': {
                    'value': 33,
                    'type': 'int',
                    'min': 1,
                    'max': 60,
                    'description': 'UTM zone for coordinate conversion'
                },
                'utm_hemisphere': {
                    'value': 'N',
                    'type': 'choice',
                    'choices': ['N', 'S'],
                    'description': 'UTM hemisphere (N=North, S=South)'
                },
                'sampling_mode': {
                    'value': 'interpolate',
                    'type': 'choice',
                    'choices': ['interpolate', 'downsample'],
                    'description': 'Choose how to handle magnetometer data without GPS points'
                }
            },
            'output_options': {
                'generate_visualization_data': {
                    'value': True,
                    'type': 'bool',
                    'description': 'Generate layer data for map visualization'
                },
                'include_diagnostics': {
                    'value': True,
                    'type': 'bool',
                    'description': 'Include processing diagnostics in output'
                }
            }
        }
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate MagWalk data structure using the flexible column detector."""
        if data.empty:
            raise ProcessingError("No MagWalk data provided")

        # Use the same detection logic as the rest of the script for consistency
        detected_columns = self._detect_magwalk_columns(data)

        # Check if essential columns were found
        if 'latitude' not in detected_columns or 'longitude' not in detected_columns:
            raise ProcessingError("Could not find latitude or longitude columns in data")
        
        if 'btotal1' not in detected_columns and not all(k in detected_columns for k in ['bx1', 'by1', 'bz1']):
            raise ProcessingError("Could not find magnetic field data (Btotal1 or Bx1/By1/Bz1)")

        logging.info(f"MagWalk data validation successful. Detected columns: {detected_columns}")
        return True
    
    def execute(self, data: pd.DataFrame, params: Dict[str, Any], 
            progress_callback: Optional[Callable] = None, input_file_path: Optional[str] = None) -> ProcessingResult:
        """Execute magbase processing with comprehensive functionality"""
        try:
            if progress_callback:
                progress_callback(0, "Initializing magbase processing...")
            
            # Extract parameters
            file_inputs = params.get('file_inputs', {})
            sensor_config = params.get('sensor_configuration', {})
            process_opts = params.get('processing_options', {})
            output_opts = params.get('output_options', {})
            
            base_station_file = file_inputs.get('base_station_file', {}).get('value', '')
            disable_basestation = process_opts.get('disable_basestation', {}).get('value', False)
            
            # Create result
            result = ProcessingResult(success=True)
            
            # Step 1: Data validation and preparation
            if progress_callback:
                progress_callback(10, "Validating and preparing data...")
            
            magwalk_df = data.copy()
            
            # Detect key columns
            detected_cols = self._detect_magwalk_columns(magwalk_df)
            logging.info(f"Detected columns: {detected_cols}")
            logging.info(f"Available columns: {list(magwalk_df.columns)}")
            
            # Step 2: Load base station data if not disabled
            base_station_df = None
            if not disable_basestation and base_station_file:
                if progress_callback:
                    progress_callback(15, "Loading base station data...")
                
                try:
                    delimiter = file_inputs.get('base_station_delimiter', {}).get('value', ',')
                    base_station_df = self._read_base_station_file(base_station_file, delimiter)
                    logging.info(f"Base station data loaded: {len(base_station_df)} records")
                except Exception as e:
                    logging.warning(f"Failed to load base station data: {str(e)}")
                    base_station_df = None
            
            # Get the selected sampling mode
            sampling_mode = process_opts.get('sampling_mode', {}).get('value', 'interpolate')

            # Step 3: Filter and sample the data based on the chosen mode
            if sampling_mode == 'downsample':
                if progress_callback:
                    progress_callback(20, "Downsampling to GPS points...")
                magwalk_df = self._downsample_to_gps_points(magwalk_df, detected_cols)
            else: # Default to interpolation
                if progress_callback:
                    progress_callback(20, "Filtering MagWalk data...")
                magwalk_df = self._filter_magwalk_data(magwalk_df, detected_cols)
                
                if progress_callback:
                    progress_callback(30, "Interpolating GPS gaps...")
                magwalk_df = self._interpolate_gps_gaps(magwalk_df, detected_cols)
            
            # Step 4.5: This step incorrectly overwrites the data and must be removed.
            # magwalk_df = self._fix_coordinates(magwalk_df, detected_cols)
            
            # Step 5: Add calculated fields (UTM conversion, sensor offsets, total field)
            if progress_callback:
                progress_callback(50, "Converting coordinates and calculating fields...")
            
            magwalk_df = self._add_calculated_fields(
                magwalk_df, 
                detected_cols,
                sensor_config,
                process_opts
            )
            
            # Step 6: Calculate residual anomalies (diurnal correction)
            if not disable_basestation and base_station_df is not None:
                if progress_callback:
                    progress_callback(70, "Calculating residual anomalies...")
                
                magwalk_df = self._calculate_residual_anomalies(
                    magwalk_df, 
                    base_station_df,
                    detected_cols,
                    process_opts,
                    progress_callback=progress_callback
                )
            else:
                logging.info("Base station correction disabled or no base station data available")
            
            # Step 7: Generate outputs
            if progress_callback:
                progress_callback(90, "Generating outputs...")
            
            # Clean up intermediate columns before setting final output
            if 'MagWalk_LocalTime' in magwalk_df.columns:
                magwalk_df = magwalk_df.drop(columns=['MagWalk_LocalTime'])
                logging.debug("Removed intermediate MagWalk_LocalTime column from output")
            
            # Final cleanup like original magbase.py
            magwalk_df = magwalk_df.loc[:, ~magwalk_df.columns.isna()]
            magwalk_df = magwalk_df.loc[:, ~magwalk_df.columns.str.contains('^Unnamed')]
            magwalk_df = magwalk_df.loc[:, magwalk_df.columns != '']
            
            # Set processed data
            result.data = magwalk_df
            
            # Generate metadata
            result.metadata = {
                'processor': 'magnetic',
                'script': 'magbase_processing',
                'data_shape': magwalk_df.shape,
                'detected_columns': detected_cols,
                'base_station_used': base_station_df is not None,
                'utm_zone': process_opts.get('utm_zone', {}).get('value', 33),
                'processing_timestamp': datetime.now().isoformat(),
                'parameters': params,
                'anomalies_found': None
            }
            
            # Add diagnostics if requested
            if output_opts.get('include_diagnostics', {}).get('value', True):
                result.metadata['diagnostics'] = self._generate_diagnostics(magwalk_df, detected_cols)
            
            
            # Generate output filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if input_file_path:
                input_prefix = Path(input_file_path).stem
            else:
                input_prefix = "magwalk_data"
            
            output_filename = f"{input_prefix}_magbase_{timestamp}.csv"
            
            # Add output file
            result.add_output_file(
                file_path=output_filename,
                file_type="csv",
                description="Processed magnetic survey data with UTM coordinates and diurnal correction",
                metadata={
                    'columns_added': ['UTM_Easting', 'UTM_Northing', 'Btotal1 [nT]', 'Btotal2 [nT]'],
                    'utm_zone': process_opts.get('utm_zone', {}).get('value', 33)
                }
            )
            
            if progress_callback:
                progress_callback(100, "Magbase processing complete!")
            
            return result
            
        except Exception as e:
            logging.error(f"Magbase processing failed: {str(e)}")
            return ProcessingResult(
                success=False,
                error_message=f"Magbase processing failed: {str(e)}"
            )
    
    def _detect_magwalk_columns(self, data: pd.DataFrame) -> Dict[str, str]:
        """Detect MagWalk column names with support for multiple naming conventions"""
        patterns = {
            'latitude': ['lat', 'latitude'],
            'longitude': ['lon', 'lng', 'longitude'],
            'timestamp': ['time', 'timestamp', 'datetime'],
            'altitude': ['alt', 'altitude', 'height'],
            'btotal1': ['btotal1', 'r1', 'magnetic1'],
            'btotal2': ['btotal2', 'r2', 'magnetic2'],
            # Specific GPS time columns
            'gps_numeric_time': ['gpstime', 'gpstime (s)', 'time'], # e.g., 90536.004
            'gps_formatted_time': ['gpstime [hh:mm:ss.sss]'], # e.g., 09:05:36.004
            # Support both naming conventions
            'bx1': ['bx1', 'bx_1', 'b1x [nt]', 'b1x'],
            'by1': ['by1', 'by_1', 'b1y [nt]', 'b1y'], 
            'bz1': ['bz1', 'bz_1', 'b1z [nt]', 'b1z'],
            'bx2': ['bx2', 'bx_2', 'b2x [nt]', 'b2x'],
            'by2': ['by2', 'by_2', 'b2y [nt]', 'b2y'],
            'bz2': ['bz2', 'bz_2', 'b2z [nt]', 'b2z']
        }
        
        detected = {}
        for col_type, keywords in patterns.items():
            for col in data.columns:
                col_lower = col.lower()
                if any(kw.lower() in col_lower for kw in keywords):
                    detected[col_type] = col
                    break
        
        return detected
    
    def _read_base_station_file(self, file_path: str, delimiter: str = ',') -> pd.DataFrame:
        """Read GSM-19 base station data file using original magbase.py logic"""
        if not os.path.exists(file_path):
            raise ProcessingError(f"Base station file not found: {file_path}")
        
        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()
            
            header_lines = []
            data_start_line = 0
            last_header_line = ""
            
            # Find header lines and data start
            for i, line in enumerate(lines):
                line = line.strip()
                if line.startswith('/'):
                    header_lines.append(line)
                    last_header_line = line
                    data_start_line = i + 1
                elif i > data_start_line and line and not line.startswith('/'):
                    break
            
            # Extract column names from header or first data line
            columns = None
            if delimiter in last_header_line:
                # Try to extract column names from the last header line
                potential_columns = last_header_line.replace('/', '').strip().split(delimiter)
                potential_columns = [col.strip() for col in potential_columns if col.strip()]
                if potential_columns:
                    columns = potential_columns
                    logging.debug(f"Extracted column names from header: {columns}")
            
            if not columns:
                # Check if the line after the last header has column names
                if data_start_line < len(lines) and delimiter in lines[data_start_line]:
                    columns = [col.strip() for col in lines[data_start_line].strip().split(delimiter) if col.strip()]
                    data_start_line += 1
                    logging.debug(f"Extracted column names from line after header: {columns}")
                else:
                    # Default column names
                    columns = ['time', 'nT', 'sq']
                    logging.debug("Using default column names")
            
            # Read data manually to handle parsing issues
            data = []
            for i in range(data_start_line, len(lines)):
                line = lines[i].strip()
                if line and not line.startswith('/'):
                    values = [val.strip() for val in line.split(delimiter)]
                    if len(values) >= 2:  # Ensure we have at least some valid data
                        # Pad with empty strings if needed
                        while len(values) < len(columns):
                            values.append('')
                        data.append(values[:len(columns)])
            
            # Create DataFrame
            df = pd.DataFrame(data, columns=columns)
            
            # Remove any unnamed or empty columns
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            df = df.loc[:, ~df.columns.isna()]
            
            # Convert numeric columns to appropriate types
            for col in df.columns:
                if col != 'time':  # Don't convert time column
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    except (ValueError, TypeError):
                        pass
            
            # Ensure we have required columns
            if 'nT' not in df.columns:
                # Try to find magnetic field column
                possible_mag_cols = [col for col in df.columns if 'nt' in col.lower() or 'magnetic' in col.lower()]
                if possible_mag_cols:
                    df['nT'] = df[possible_mag_cols[0]]
                else:
                    raise ProcessingError("Could not find magnetic field column in base station data")
            
            logging.info(f"Successfully loaded base station file with {len(df)} records")
            logging.info(f"Base station columns: {df.columns.tolist()}")
            
            return df
            
        except Exception as e:
            raise ProcessingError(f"Failed to read base station file: {str(e)}")
    
    def _filter_magwalk_data(self, df: pd.DataFrame, detected_cols: Dict[str, str]) -> pd.DataFrame:
        """
        Filter MagWalk data to remove data before first GPS point and after last GPS point.
        This matches the original magbase.py filter_magwalk_data function behavior.
        """
        original_length = len(df)
        
        # Find GPS time column (matching original logic)
        gps_time_col = None
        for col in df.columns:
            if 'GPSTime' in col and '[hh:mm:ss' in col:
                gps_time_col = col
                break
        
        if not gps_time_col:
            # Fallback to any GPSTime column
            for col in df.columns:
                if 'GPSTime' in col:
                    gps_time_col = col
                    break
        
        if not gps_time_col:
            logging.warning(f"No GPS time column found in detected columns: {detected_cols}")
            return df
        
        # Find first and last valid GPS points
        first_valid_idx = None
        last_valid_idx = None
        
        for idx, row in df.iterrows():
            gps_value = row[gps_time_col]
            # Check if GPS value is valid (not NaN, not '0', not empty)
            if (pd.notna(gps_value) and 
                str(gps_value) != '0' and 
                str(gps_value).strip() != ''):
                if first_valid_idx is None:
                    first_valid_idx = idx
                last_valid_idx = idx
        
        if first_valid_idx is None or last_valid_idx is None:
            logging.warning("No valid GPS points found, returning original data")
            return df
        
        # Filter to keep data between first and last GPS points (inclusive)
        filtered_df = df.iloc[first_valid_idx:(last_valid_idx + 1)].reset_index(drop=True)
        
        rows_removed_start = first_valid_idx
        rows_removed_end = original_length - (last_valid_idx + 1)
        
        logging.info(f"GPS filtering: {rows_removed_start} rows removed from start, "
                    f"{rows_removed_end} rows removed from end")
        logging.info(f"Filtered MagWalk data: {original_length} -> {len(filtered_df)} records")
        
        return filtered_df
    
    def _interpolate_gps_gaps(self, df: pd.DataFrame, detected_cols: Dict[str, str]) -> pd.DataFrame:
        """
        Interpolate GPS gaps using the original, proven logic from magbase.py
        to ensure identical output.
        """
        start_time = time.time()
        logging.info("Starting GPS data interpolation with original, proven logic...")

        if df.empty:
            logging.warning("Empty dataframe provided, skipping interpolation.")
            return df

        df = df.copy()

        # --- 1. Identify all relevant columns (as in the original script) ---
        numeric_time_col = next((c for c in df.columns if 'GPSTime' in c and '[hh:mm:ss' not in c), None)
        formatted_time_col = next((c for c in df.columns if 'GPSTime' in c and '[hh:mm:ss' in c), None)
        date_col = detected_cols.get('gpsdate') or next((c for c in df.columns if 'GPSDate' in c), None)
        
        if not numeric_time_col:
            logging.warning("No numeric GPSTime column found. Cannot perform interpolation.")
            return df

        # List of all columns to interpolate numerically
        cols_to_interpolate = [
            numeric_time_col,
            next((c for c in df.columns if 'AccX' in c), None),
            next((c for c in df.columns if 'AccY' in c), None),
            next((c for c in df.columns if 'AccZ' in c), None),
            detected_cols.get('latitude'),
            detected_cols.get('longitude'),
            detected_cols.get('altitude')
        ]
        # Filter out any columns that weren't found
        cols_to_interpolate = [col for col in cols_to_interpolate if col is not None]

        # --- 2. Find valid GPS points and perform manual, row-by-row interpolation ---
        gps_indices = df[df[numeric_time_col].notna() & (df[numeric_time_col] != 0)].index.tolist()

        if len(gps_indices) < 2:
            logging.warning("Not enough GPS points found for interpolation.")
            return df

        logging.info(f"Found {len(gps_indices)} GPS-marked datapoints. Processing gaps...")

        for i in range(len(gps_indices) - 1):
            start_idx, end_idx = gps_indices[i], gps_indices[i+1]
            gap_size = end_idx - start_idx
            if gap_size <= 1:
                continue

            # --- Handle all numeric columns ---
            for col in cols_to_interpolate:
                try:
                    start_val = float(df.at[start_idx, col])
                    end_val = float(df.at[end_idx, col])
                    if pd.isna(start_val) or pd.isna(end_val):
                        continue
                    
                    step = (end_val - start_val) / gap_size
                    for offset in range(1, gap_size):
                        df.at[start_idx + offset, col] = start_val + (step * offset)
                except (ValueError, TypeError):
                    continue # Skip if values can't be converted to float

            # --- Handle GPSDate Propagation ---
            if date_col:
                start_date = df.at[start_idx, date_col]
                if pd.notna(start_date):
                    for offset in range(1, gap_size):
                        df.at[start_idx + offset, date_col] = start_date

            # --- Handle Formatted Time String (hh:mm:ss.sss) ---
            if formatted_time_col:
                reference_time_str = df.at[start_idx, formatted_time_col]
                if isinstance(reference_time_str, str) and ':' in reference_time_str:
                    parts = reference_time_str.split(':')
                    hours_mins = f"{parts[0]}:{parts[1]}"
                    
                    # Use the newly interpolated numeric time for the seconds part
                    for offset in range(1, gap_size):
                        numeric_time_val = df.at[start_idx + offset, numeric_time_col]
                        if pd.notna(numeric_time_val):
                            seconds = float(numeric_time_val) % 60
                            df.at[start_idx + offset, formatted_time_col] = f"{hours_mins}:{seconds:06.3f}"
                            
        elapsed = time.time() - start_time
        logging.info(f"Interpolation complete in {elapsed:.2f} seconds.")
        return df

    def _downsample_to_gps_points(self, df: pd.DataFrame, detected_cols: Dict[str, str]) -> pd.DataFrame:
        """Downsample the data to only include points with valid GPS data."""
        original_length = len(df)
        
        lat_col = detected_cols.get('latitude')
        lon_col = detected_cols.get('longitude')

        if not lat_col or not lon_col:
            logging.warning("Cannot downsample - latitude/longitude columns not found")
            return df

        # Filter rows that have valid latitude and longitude
        downsampled_df = df[pd.notna(df[lat_col]) & pd.notna(df[lon_col]) & (df[lat_col] != 0) & (df[lon_col] != 0)].reset_index(drop=True)
        
        rows_removed = original_length - len(downsampled_df)
        logging.info(f"Downsampled data from {original_length} to {len(downsampled_df)} records, removing {rows_removed} rows without GPS data.")
        
        return downsampled_df
    
    def _fix_coordinates(self, df: pd.DataFrame, detected_cols: Dict[str, str]) -> pd.DataFrame:
        """Fix coordinate data by propagating valid GPS coordinates to all rows"""
        
        lat_col = detected_cols.get('latitude') or next((c for c in df.columns if 'lat' in c.lower()), None)
        lon_col = detected_cols.get('longitude') or next((c for c in df.columns if 'lon' in c.lower()), None)
        alt_col = detected_cols.get('altitude') or next((c for c in df.columns if 'alt' in c.lower()), None)
        
        if not lat_col or not lon_col:
            logging.warning("No coordinate columns found, skipping coordinate fix")
            return df
        
        # Find rows with valid coordinates (non-zero, non-NaN)
        valid_coords_mask = (
            pd.notna(df[lat_col]) & 
            pd.notna(df[lon_col]) & 
            (df[lat_col] != 0) & 
            (df[lon_col] != 0)
        )
        
        valid_rows = df[valid_coords_mask]
        logging.info(f"Found {len(valid_rows)} rows with valid coordinates out of {len(df)} total")
        
        if len(valid_rows) > 0:
            # Use the first valid coordinate for all rows
            valid_lat = valid_rows[lat_col].iloc[0]
            valid_lon = valid_rows[lon_col].iloc[0]
            valid_alt = valid_rows[alt_col].iloc[0] if alt_col and pd.notna(valid_rows[alt_col].iloc[0]) else None
            
            logging.info(f"Propagating coordinates - lat: {valid_lat}, lon: {valid_lon}, alt: {valid_alt}")
            
            # Propagate to all rows
            df[lat_col] = valid_lat
            df[lon_col] = valid_lon
            if alt_col and valid_alt is not None:
                df[alt_col] = valid_alt
            
            logging.info(f"Propagated coordinates to all {len(df)} rows")
        else:
            logging.warning("No valid coordinates found, cannot fix coordinates")
        
        # Fix GPS time data - use simple progressive approach
        self._fix_gps_time_simple(df)
        
        return df
    
    def _fix_gps_time_simple(self, df: pd.DataFrame) -> None:
        """Simple GPS time fix - find the original reference time and create progressive sequence"""
        
        # Find GPS time columns
        gps_numeric_col = next((c for c in df.columns if 'GPSTime' in c and '[hh:mm:ss' not in c), None)
        gps_formatted_col = next((c for c in df.columns if 'GPSTime' in c and '[hh:mm:ss' in c), None)
        gps_date_col = next((c for c in df.columns if 'GPSDate' in c), None)
        
        # Get the original reference GPS time from the first row with valid GPS data
        reference_time = None
        reference_formatted = None
        reference_date = None
        
        if gps_numeric_col:
            # Find first valid numeric GPS time
            for idx, val in df[gps_numeric_col].items():
                if pd.notna(val) and val != 0:
                    reference_time = float(val)
                    break
        
        if gps_formatted_col:
            # Find first valid formatted GPS time
            for idx, val in df[gps_formatted_col].items():
                if pd.notna(val) and str(val) != '0' and ':' in str(val):
                    reference_formatted = str(val)
                    break
        
        if gps_date_col:
            # Find first valid GPS date
            for idx, val in df[gps_date_col].items():
                if pd.notna(val) and str(val) != '0':
                    reference_date = val
                    break
        
        logging.info(f"Reference GPS time: numeric={reference_time}, formatted={reference_formatted}, date={reference_date}")
        
        # Debug: If we have both reference values, check if they match
        if reference_time is not None and reference_formatted is not None:
            # Calculate what the numeric time should be for the formatted time
            try:
                parts = reference_formatted.split(':')
                expected_hours = int(parts[0])
                expected_minutes = int(parts[1])
                expected_seconds = float(parts[2])
                expected_numeric = expected_hours * 3600 + expected_minutes * 60 + expected_seconds
                
                logging.info(f"Expected numeric time from '{reference_formatted}': {expected_numeric}")
                logging.info(f"Actual numeric time: {reference_time}")
                logging.info(f"Difference: {reference_time - expected_numeric} seconds")
                
                # Use the formatted time as the source of truth and calculate correct numeric time
                reference_time = expected_numeric
                logging.info(f"Corrected reference time to: {reference_time}")
                
            except Exception as e:
                logging.warning(f"Could not parse formatted time: {e}")
        
        # Create progressive GPS times
        if reference_time is not None and gps_numeric_col:
            sample_rate = 250  # Hz
            time_increment = 1.0 / sample_rate  # 0.004 seconds
            
            for i in range(len(df)):
                new_time = reference_time + (i * time_increment)
                df.at[i, gps_numeric_col] = new_time
            
            logging.info(f"Generated progressive numeric GPS times starting from {reference_time}")
        
        # Create progressive formatted times from numeric times
        if gps_formatted_col and gps_numeric_col:
            def numeric_to_formatted_time(numeric_time):
                if pd.isna(numeric_time):
                    return None
                
                # GPS numeric time is seconds since midnight
                total_seconds = float(numeric_time)
                hours = int(total_seconds // 3600)
                minutes = int((total_seconds % 3600) // 60)
                seconds = total_seconds % 60
                
                # Verify: 90536 = 25.148 hours = 1 day + 1.148 hours = 01:08:56
                # But we want it to be 09:05:36, so there's something wrong with the reference time
                
                return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
            
            df[gps_formatted_col] = df[gps_numeric_col].apply(numeric_to_formatted_time)
            logging.info("Generated progressive formatted GPS times")
        
        # Propagate GPS date to all rows
        if reference_date is not None and gps_date_col:
            df[gps_date_col] = reference_date
            logging.info(f"Propagated GPS date '{reference_date}' to all rows")
    
    def _fix_gps_time_columns(self, df: pd.DataFrame) -> None:
        """Fix and interpolate GPS time columns properly"""
        
        # Find GPS time columns
        gps_numeric_col = next((c for c in df.columns if 'GPSTime' in c and '[hh:mm:ss' not in c), None)
        gps_formatted_col = next((c for c in df.columns if 'GPSTime' in c and '[hh:mm:ss' in c), None)
        gps_date_col = next((c for c in df.columns if 'GPSDate' in c), None)
        
        logging.info(f"GPS time columns found - numeric: {gps_numeric_col}, formatted: {gps_formatted_col}, date: {gps_date_col}")
        
        # For GPSDate - propagate the valid date to all rows
        if gps_date_col:
            valid_dates = df[df[gps_date_col].notna() & (df[gps_date_col] != 0) & (df[gps_date_col] != '0')]
            if len(valid_dates) > 0:
                valid_date = valid_dates[gps_date_col].iloc[0]
                df[gps_date_col] = valid_date
                logging.info(f"Propagated GPS date '{valid_date}' to all rows")
        
        # For numeric GPS time - calculate based on timestamps
        if gps_numeric_col:
            self._interpolate_numeric_gps_time(df, gps_numeric_col)
        
        # For formatted GPS time - calculate based on numeric time or timestamps  
        if gps_formatted_col:
            self._interpolate_formatted_gps_time(df, gps_formatted_col, gps_numeric_col)
    
    def _interpolate_numeric_gps_time(self, df: pd.DataFrame, gps_numeric_col: str) -> None:
        """Interpolate numeric GPS time based on row progression with small increments"""
        
        # Find the first valid GPS time as reference
        valid_times = df[df[gps_numeric_col].notna() & (df[gps_numeric_col] != 0)]
        if len(valid_times) == 0:
            logging.warning("No valid numeric GPS times found")
            return
            
        # Use the reference GPS time from the first row
        reference_time = valid_times[gps_numeric_col].iloc[0]
        reference_idx = valid_times.index[0]
        
        logging.info(f"Using reference GPS time {reference_time} from row {reference_idx}")
        
        # Use a very small time increment based on typical MagWalk sampling rate
        # MagWalk samples at ~250Hz, so each sample is about 0.004 seconds apart
        time_increment = 0.004  # 4 milliseconds between samples
        
        # Calculate time for each row based on its position relative to the first row
        for i, (idx, row) in enumerate(df.iterrows()):
            # Calculate time as: reference_time + (row_position * time_increment)
            interpolated_time = reference_time + (i * time_increment)
            # Ensure time stays within valid 24-hour range (0-86399 seconds)
            interpolated_time = interpolated_time % 86400  # 86400 = 24 * 3600
            df.at[idx, gps_numeric_col] = interpolated_time
                
        logging.info(f"Interpolated numeric GPS times using {time_increment}s increments")
    
    def _interpolate_formatted_gps_time(self, df: pd.DataFrame, gps_formatted_col: str, gps_numeric_col: str = None) -> None:
        """Interpolate formatted GPS time (hh:mm:ss.sss)"""
        
        if gps_numeric_col and gps_numeric_col in df.columns:
            # Use numeric GPS time to generate formatted time
            def numeric_to_formatted(numeric_time):
                if pd.isna(numeric_time):
                    return None
                
                # Numeric GPS time is seconds since midnight
                total_seconds = float(numeric_time)
                hours = int(total_seconds // 3600) % 24  # Ensure hours are 0-23
                minutes = int((total_seconds % 3600) // 60)
                seconds = total_seconds % 60
                
                return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
            
            df[gps_formatted_col] = df[gps_numeric_col].apply(numeric_to_formatted)
            logging.info("Generated formatted GPS times from numeric GPS times")
            
        else:
            # Find first valid formatted time as reference and interpolate from there
            valid_times = df[df[gps_formatted_col].notna() & 
                           (df[gps_formatted_col] != '0') & 
                           df[gps_formatted_col].astype(str).str.contains(':')]
            
            if len(valid_times) > 0:
                reference_time_str = valid_times[gps_formatted_col].iloc[0]
                reference_idx = valid_times.index[0]
                
                # Parse reference time
                parts = reference_time_str.split(':')
                reference_hours = int(parts[0])
                reference_minutes = int(parts[1]) 
                reference_seconds = float(parts[2])
                reference_total_sec = reference_hours * 3600 + reference_minutes * 60 + reference_seconds
                
                # Use timestamp for interpolation
                timestamp_col = next((c for c in df.columns if 'Timestamp' in c), None)
                if timestamp_col:
                    reference_timestamp = df.at[reference_idx, timestamp_col]
                    
                    for idx, row in df.iterrows():
                        timestamp_diff_ms = row[timestamp_col] - reference_timestamp
                        timestamp_diff_sec = timestamp_diff_ms / 1000.0
                        
                        total_sec = reference_total_sec + timestamp_diff_sec
                        hours = int(total_sec // 3600) % 24  # Handle day rollover
                        minutes = int((total_sec % 3600) // 60)
                        seconds = total_sec % 60
                        
                        formatted_time = f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
                        df.at[idx, gps_formatted_col] = formatted_time
                        
                    logging.info("Interpolated formatted GPS times based on timestamp progression")
                else:
                    # Fallback: propagate same time to all rows
                    df[gps_formatted_col] = reference_time_str
                    logging.warning("No timestamp found, propagated same formatted time to all rows")
        
        return df
    
    def _add_calculated_fields(self, df: pd.DataFrame, detected_cols: Dict[str, str], 
                              sensor_config: Dict[str, Any], process_opts: Dict[str, Any]) -> pd.DataFrame:
        """Add calculated fields: UTM coordinates, sensor positions, total field"""
        
        lat_col = detected_cols.get('latitude')
        lon_col = detected_cols.get('longitude')
        
        if not lat_col or not lon_col:
            raise ProcessingError("Cannot calculate fields - latitude/longitude columns not found")
        
        # Get UTM zone configuration
        utm_zone = process_opts.get('utm_zone', {}).get('value', 33)
        utm_hemisphere = process_opts.get('utm_hemisphere', {}).get('value', 'N')
        
        # Create UTM transformer exactly like original magbase.py
        utm_transformer = pyproj.Transformer.from_crs(
            "EPSG:4326",   # WGS84
            f"EPSG:326{utm_zone:02d}" if utm_hemisphere == 'N' else f"EPSG:327{utm_zone:02d}",  # UTM Zone
            always_xy=True
        )
        
        # Apply conversion and create new columns exactly like original
        utm_coords = [utm_transformer.transform(lon, lat) 
                      for lon, lat in zip(df[lon_col], df[lat_col])]
        
        # Add UTM columns with sensor-specific offsets exactly like original
        base_easting = [coord[0] for coord in utm_coords]
        base_northing = [coord[1] for coord in utm_coords]
        
        # Add the base coordinates (GPS position)
        df['UTM_Easting'] = base_easting
        df['UTM_Northing'] = base_northing
        
        # Add sensor positions with offsets exactly like original
        sensor1_east = sensor_config.get('sensor1_offset_east', {}).get('value', 0.0375)
        sensor1_north = sensor_config.get('sensor1_offset_north', {}).get('value', -0.56)
        sensor2_east = sensor_config.get('sensor2_offset_east', {}).get('value', 0.0375)
        sensor2_north = sensor_config.get('sensor2_offset_north', {}).get('value', 0.56)
        
        # Add sensor 1 coordinates with offsets
        df['UTM_Easting1'] = [e + sensor1_east for e in base_easting]
        df['UTM_Northing1'] = [n + sensor1_north for n in base_northing]
        
        # Add sensor 2 coordinates with offsets
        df['UTM_Easting2'] = [e + sensor2_east for e in base_easting]
        df['UTM_Northing2'] = [n + sensor2_north for n in base_northing]
        
        # Calculate total field if component data is available
        self._calculate_total_field(df, detected_cols, 1)
        self._calculate_total_field(df, detected_cols, 2)
        
        logging.info(f"Added UTM coordinates (Zone {utm_zone}{utm_hemisphere})")
        
        # Final cleanup of any NaN columns that might have been introduced
        df = df.loc[:, ~df.columns.isna()]
        
        return df
    
    def _calculate_total_field(self, df: pd.DataFrame, detected_cols: Dict[str, str], sensor_num: int):
        """Calculate total magnetic field from components exactly like original magbase.py"""
        bx_col = detected_cols.get(f'bx{sensor_num}')
        by_col = detected_cols.get(f'by{sensor_num}')
        bz_col = detected_cols.get(f'bz{sensor_num}')
        
        logging.info(f"Total field calculation for sensor {sensor_num}: Bx={bx_col}, By={by_col}, Bz={bz_col}")
        
        if bx_col and by_col and bz_col:
            # Calculate Btotal = sqrt(Bx^2 + By^2 + Bz^2) exactly like original
            df[f'Btotal{sensor_num} [nT]'] = np.sqrt(
                df[bx_col]**2 + 
                df[by_col]**2 + 
                df[bz_col]**2
            )
            valid_count = df[f'Btotal{sensor_num} [nT]'].notna().sum()
            sample_values = df[f'Btotal{sensor_num} [nT]'].dropna().head(3).tolist()
            logging.info(f"Added Btotal{sensor_num} (Euclidean norm): {valid_count} valid values, sample: {sample_values}")
        else:
            # Check if Btotal already exists with various naming conventions
            btotal_col = detected_cols.get(f'btotal{sensor_num}')
            if btotal_col:
                df[f'Btotal{sensor_num} [nT]'] = df[btotal_col]
                valid_count = df[f'Btotal{sensor_num} [nT]'].notna().sum()
                sample_values = df[f'Btotal{sensor_num} [nT]'].dropna().head(3).tolist()
                logging.info(f"Using existing Btotal{sensor_num} column '{btotal_col}': {valid_count} valid values, sample: {sample_values}")
            else:
                logging.warning(f"Could not calculate Btotal{sensor_num}: B{sensor_num} component columns missing")
                logging.info(f"Available detected columns: {list(detected_cols.keys())}")
    
    def _calculate_residual_anomalies(self, magwalk_df: pd.DataFrame, base_station_df: pd.DataFrame,
                                    detected_cols: Dict[str, str], process_opts: Dict[str, Any],
                                    progress_callback: Optional[Callable] = None) -> pd.DataFrame:
        """Calculate residual anomalies using base station diurnal correction with multiprocessing"""
        logging.info("Calculating residual magnetic anomalies with multiprocessing...")

        df = magwalk_df.copy()
        
        if process_opts.get('disable_basestation', {}).get('value', False) or base_station_df is None or base_station_df.empty:
            logging.info("Basestation correction disabled or no data available.")
            if process_opts.get('vertical_alignment', {}).get('value', False):
                logging.info("Calculating vertical alignment (M1-M2) without basestation correction.")
                if 'Btotal1 [nT]' in df.columns and 'Btotal2 [nT]' in df.columns:
                    df['VA [nT]'] = df['Btotal1 [nT]'] - df['Btotal2 [nT]']
            return df

        magwalk_time_col = 'GPSTime [hh:mm:ss.sss]'
        if magwalk_time_col not in df.columns:
            for col in df.columns:
                if 'GPSTime' in col and '[hh:mm:ss' in col:
                    magwalk_time_col = col
                    break
        
        if magwalk_time_col not in df.columns:
            logging.warning(f"Cannot calculate residuals: {magwalk_time_col} column missing.")
            return df

        base_time_col = 'time'
        base_field_col = 'nT'
        if base_time_col not in base_station_df.columns or base_field_col not in base_station_df.columns:
            logging.warning("Base station file missing 'time' or 'nT' columns.")
            return df

        def convert_magwalk_time_to_base_time(time_str):
            # Parse the magwalk time (UTC)
            if pd.isna(time_str) or not isinstance(time_str, str):
                return None
                
            try:
                # Extract hours, minutes, seconds from time string (format: hh:mm:ss.sss)
                parts = time_str.split(':')
                hours = int(parts[0])
                minutes = int(parts[1])
                seconds = float(parts[2])
                
                # Convert from UTC to UTC+2 by adding 2 hours
                local_hours = (hours + 2) % 24
                
                # Format as a time string matching the base station format
                # Base station times are in the format HHMMSS.S (e.g., 083954.0)
                return f"{local_hours:02d}{minutes:02d}{seconds:.1f}".replace('.', '')
            except (ValueError, IndexError) as e:
                logging.debug(f"Time conversion error: {e}")
                return None

        df['MagWalk_LocalTime'] = df[magwalk_time_col].apply(convert_magwalk_time_to_base_time)
        
        # Convert MagWalk local times to numeric (matching base station format)
        df['MagWalk_LocalTime'] = pd.to_numeric(df['MagWalk_LocalTime'], errors='coerce')
        
        # Log time conversion status and sample values for debugging
        valid_conversions = df['MagWalk_LocalTime'].notna().sum()
        logging.info(f"Time conversion: {valid_conversions}/{len(df)} successful conversions")
        
        # Debug: Log sample converted times to verify they match original magbase.py
        sample_converted = df['MagWalk_LocalTime'].dropna().head(3).tolist()
        logging.info(f"Sample converted times for residual calculation: {sample_converted}")
        
        # Debug: Log basic base station info
        logging.info(f"Base station data: {len(base_station_df)} records")
        if base_time_col in base_station_df.columns:
            logging.info(f"Base station time column '{base_time_col}' found")
        if base_field_col in base_station_df.columns:
            sample_base_fields = base_station_df[base_field_col].head(3).tolist()
            logging.info(f"Sample base fields: {sample_base_fields}")
        df['R1 [nT]'] = np.nan
        df['R2 [nT]'] = np.nan
        if process_opts.get('vertical_alignment', {}).get('value', False):
            df['VA [nT]'] = np.nan

        # Convert base station times to numeric and sort for binary search (exactly like original magbase.py)
        base_times_numeric = pd.to_numeric(base_station_df[base_time_col], errors='coerce')
        base_station_df = base_station_df.assign(time_numeric=base_times_numeric)
        base_station_df = base_station_df.sort_values('time_numeric')
        base_station_df = base_station_df.reset_index(drop=True)
        
        base_times_array = base_station_df['time_numeric'].values
        base_fields_array = base_station_df[base_field_col].values
        
        # Debug: Log processed base station info
        logging.info(f"Base station time range: {base_station_df['time_numeric'].min()}-{base_station_df['time_numeric'].max()}")
        sample_base_times = base_station_df['time_numeric'].head(3).tolist()
        logging.info(f"Sample processed base times: {sample_base_times}")

        num_cores = max(1, mp.cpu_count() - 1)
        chunk_size = min(10000, max(1000, len(df) // (num_cores * 10)))
        batches = [df.iloc[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
        
        logging.info(f"Using {num_cores} cores to process {len(batches)} batches.")

        process_func = partial(_process_batch_mp, base_times=base_times_array, base_fields=base_fields_array, 
                               vertical_alignment=process_opts.get('vertical_alignment', {}).get('value', False),
                               disable_basestation=process_opts.get('disable_basestation', {}).get('value', False))

        processed_rows = 0
        with mp.Pool(num_cores) as pool:
            for i, result_batch in enumerate(pool.imap(process_func, batches)):
                df.update(result_batch)
                processed_rows += len(result_batch)
                if progress_callback:
                    progress = int((i + 1) / len(batches) * 100)
                    progress_callback(70 + int(progress * 0.2), f"Processing batch {i+1}/{len(batches)}")

        logging.info(f"Finished processing {processed_rows} rows.")
        return df
    
    def _generate_diagnostics(self, df: pd.DataFrame, detected_cols: Dict[str, str]) -> Dict[str, Any]:
        """Generate processing diagnostics"""
        diagnostics = {
            'total_records': len(df),
            'detected_columns': detected_cols,
            'spatial_extent': {
                'utm_easting_range': [float(df['UTM_Easting'].min()), float(df['UTM_Easting'].max())] if 'UTM_Easting' in df.columns else None,
                'utm_northing_range': [float(df['UTM_Northing'].min()), float(df['UTM_Northing'].max())] if 'UTM_Northing' in df.columns else None
            },
            'magnetic_field_stats': {}
        }
        
        # Add magnetic field statistics
        for col in ['Btotal1 [nT]', 'Btotal2 [nT]', 'R1 [nT]', 'R2 [nT]']:
            if col in df.columns:
                diagnostics['magnetic_field_stats'][col] = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max())
                }
        
        return diagnostics

# Export the script class
SCRIPT_CLASS = MagbaseProcessing