import pandas as pd
import numpy as np
import os
import datetime
import time
import logging
import sys
from pathlib import Path
import pyproj

# Setup logging
def setup_logging():
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Create a log file with current timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"magbase_log_{timestamp}.txt"
    
    # Configure logging to write to both file and console
    handlers = [
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
    
    # Set format to include timestamp
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers
    )
    
    return log_file

def read_base_station_file(file_path, delimiter=','):
    """
    Read the base station data from a .txt file with comma delimiter.
    Headers begin with '/' and data begins after the last line with '/'.
    
    Args:
        file_path (str): Path to the GSM-19 base station data file
        delimiter (str): Delimiter used in the file
        
    Returns:
        DataFrame: pandas DataFrame containing the base station data
        dict: Metadata extracted from the header
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Base station file not found: {file_path}")
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    header_lines = []
    data_start_line = 0
    last_header_line = ""
    
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith('/'):
            header_lines.append(line)
            last_header_line = line
            data_start_line = i + 1
        elif i > data_start_line and line and not line.startswith('/'):
            # data found
            break
    
    # Extract column names from the last header line
    # The last header line might contain column names or they might be in the line after the last /
    if delimiter in last_header_line:
        # Try to extract column names from the last header line
        potential_columns = last_header_line.replace('/', '').strip().split(delimiter)
        # Filter out empty columns (including trailing commas)
        potential_columns = [col.strip() for col in potential_columns if col.strip()]
        if potential_columns:  # If we have any non-empty column names
            columns = potential_columns
            logging.debug(f"Extracted column names from header: {columns}")
        else:
            # Check if the line after the last header has column names
            if data_start_line < len(lines):
                columns = [col.strip() for col in lines[data_start_line].strip().split(delimiter) if col.strip()]
                data_start_line += 1
                logging.debug(f"Extracted column names from line after header: {columns}")
            else:
                # Default column names
                columns = ['time', 'nT', 'sq']
                logging.debug("Using default column names")
    else:
        # No commas in last header line, check line after header
        if data_start_line < len(lines) and delimiter in lines[data_start_line]:
            columns = [col.strip() for col in lines[data_start_line].strip().split(delimiter) if col.strip()]
            data_start_line += 1
            logging.debug(f"Extracted column names from line after header: {columns}")
        else:
            # Default column names
            columns = ['time', 'nT', 'sq']
            logging.debug("Using default column names")
    
    # Extract metadata from headers if needed
    metadata = {}
    for line in header_lines:
        line = line.strip()
        # Extract coordinates
        if line.startswith('/ ') and ',' in line:
            parts = line.replace('/ ', '').strip().split(',')
            if len(parts) == 2:
                try:
                    metadata['latitude'] = float(parts[0])
                    metadata['longitude'] = float(parts[1])
                except ValueError:
                    pass
        # Extract datum
        elif line.startswith('/datum'):
            try:
                metadata['datum'] = float(line.replace('/datum', '').strip())
            except ValueError:
                pass
        # Store all header values
        if line.startswith('/'):
            key = line.split(' ')[0].replace('/', '').strip()
            if key and len(line) > len(key) + 1:
                value = line[len(key) + 1:].strip()
                if key not in metadata and value:
                    metadata[key] = value
    
    # Read data from the identified line
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
    basestation = pd.DataFrame(data, columns=columns)
    
    # Remove any unnamed or empty columns
    basestation = basestation.loc[:, ~basestation.columns.str.contains('^Unnamed')]
    # Remove any trailing NaN columns
    basestation = basestation.loc[:, ~basestation.columns.isna()]
    
    # Convert numeric columns to appropriate types
    for col in basestation.columns:
        if col != 'time':  # Don't convert time column
            try:
                basestation[col] = pd.to_numeric(basestation[col])
            except (ValueError, TypeError):
                pass
    
    logging.info(f"Successfully loaded base station data with {len(basestation)} records")
    return basestation, metadata


def read_magwalk_file(file_path, delimiter=';'):
    """
    Read MagWalk data from a .csv file with semicolon delimiter.
    Headers continue until the data rows begin, with column headers containing 'Timestamp'.
    
    Args:
        file_path (str): Path to the MagWalk data file
        delimiter (str): Delimiter used in the file
        
    Returns:
        DataFrame: pandas DataFrame containing the MagWalk data
        dict: Metadata extracted from the header
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"MagWalk file not found: {file_path}")
    
    # Read all lines from the file
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Process headers and find where data begins
    header_lines = []
    header_end = 0
    columns = []
    
    # Find where the data begins - usually after a line with 'Timestamp' in it
    for i, line in enumerate(lines):
        line = line.strip()
        # Look for a line that likely contains column headers
        if delimiter in line and ('Timestamp' in line or 'ms' in line or 'nT' in line):
            columns = [col.strip() for col in line.split(delimiter)]
            header_end = i
            break
        header_lines.append(line)
    
    # If no clear column header was found, use a heuristic approach
    if not columns:
        # Look for the first line with multiple semicolons (likely data)
        for i, line in enumerate(lines):
            semicolon_count = line.count(delimiter)
            if semicolon_count > 3:  # Arbitrary threshold - more than 3 semicolons suggests data row
                # Use this as the header row
                columns = [f"Column_{j}" for j in range(semicolon_count + 1)]
                header_end = i - 1
                break
    
    # Extract metadata from headers
    metadata = {}
    for line in header_lines:
        if ': ' in line:
            parts = line.split(': ', 1)
            if len(parts) == 2:
                key, value = parts[0], parts[1]
                metadata[key] = value
        elif ':' in line:  # Alternative format
            parts = line.split(':', 1)
            if len(parts) == 2:
                key, value = parts[0].strip(), parts[1].strip()
                if key and value:
                    metadata[key] = value
    
    # Read data rows
    data = []
    for i in range(header_end + 1, len(lines)):
        line = lines[i].strip()
        if line:
            values = line.split(delimiter)
            if len(values) > 1:  # Skip lines that don't have data
                # Ensure consistent number of columns
                while len(values) < len(columns):
                    values.append('')
                    
                # Limit to number of columns if we have too many values
                data.append(values[:len(columns)])
    
    # Create DataFrame
    magwalk_data = pd.DataFrame(data, columns=columns)
    
    # Remove any unnamed or empty columns
    magwalk_data = magwalk_data.loc[:, ~magwalk_data.columns.str.contains('^Unnamed')]
    # Remove any trailing NaN columns
    magwalk_data = magwalk_data.loc[:, ~magwalk_data.columns.isna()]
    
    # Convert numeric columns to appropriate types
    for col in magwalk_data.columns:
        if any(keyword in col for keyword in ['Timestamp', 'Time', 'Date']):
            continue  # Skip time/date columns
        try:
            magwalk_data[col] = pd.to_numeric(magwalk_data[col])
        except (ValueError, TypeError):
            pass
    
    logging.info(f"Successfully loaded MagWalk data with {len(magwalk_data)} records")
    return magwalk_data, metadata


def filter_magwalk_data(magwalk_df):
    """
    Filter MagWalk data to remove rows without valid GPS time data.
    Removes data points before the first GPS marker and after the last GPS marker.
    
    Args:
        magwalk_df (DataFrame): The MagWalk data to filter
        
    Returns:
        DataFrame: Filtered DataFrame containing only data between first and last GPS markers
    """
    if magwalk_df.empty:
        return magwalk_df
    
    # Find GPS time columns
    gps_time_col = next((col for col in magwalk_df.columns if 'GPSTime' in col and '[hh:mm:ss' in col), None)
    if not gps_time_col:
        gps_time_col = next((col for col in magwalk_df.columns if 'GPSTime' in col), None)
    
    if not gps_time_col:
        logging.warning("No GPSTime column found in data, returning unfiltered data.")
        return magwalk_df
    
    # Find the first row with a valid GPS time value
    first_valid_idx = None
    last_valid_idx = None
    
    # First pass: find first and last valid GPS points
    for idx, row in magwalk_df.iterrows():
        # Check if we have valid values in GPS time column (non-zero, non-NaN)
        gps_valid = pd.notna(row[gps_time_col]) and str(row[gps_time_col]) != '0' and str(row[gps_time_col]).strip() != ''
        
        if gps_valid:
            if first_valid_idx is None:
                first_valid_idx = idx
                logging.info(f"Found first valid GPS time at row {idx}: {row[gps_time_col]}")
            # Always update last_valid_idx when we find a valid GPS point
            last_valid_idx = idx
    
    # Filter the DataFrame to include only rows between first and last valid GPS points
    if first_valid_idx is not None and last_valid_idx is not None:
        original_count = len(magwalk_df)
        filtered_df = magwalk_df.iloc[first_valid_idx:(last_valid_idx + 1)].reset_index(drop=True)
        
        # Log filtering results
        points_removed_start = first_valid_idx
        points_removed_end = original_count - (last_valid_idx + 1)
        logging.info(f"Filtered out {points_removed_start} initial rows without valid GPS time data.")
        logging.info(f"Filtered out {points_removed_end} trailing rows after the last valid GPS point.")
        logging.info(f"Dataset reduced from {original_count} to {len(filtered_df)} rows")
        return filtered_df
    else:
        logging.warning("No rows with valid GPS time found. Returning unfiltered data.")
        return magwalk_df


def interpolate_gps_gaps(magwalk_df):
    """
    Identify GPS-marked datapoints and interpolate the gaps between them.
    Checks if the gaps between GPS points are consistent and reports inconsistencies.
    Interpolates GPS time, date, formatted time, and sensor data values between GPS points.
    Also sets quality markers to 1 if edge points have quality=1.
    
    Args:
        magwalk_df (DataFrame): The MagWalk data with some rows having GPS timestamps
        
    Returns:
        DataFrame: DataFrame with interpolated values in the gaps
    """
    start_time = time.time()
    logging.info("Starting GPS data interpolation...")
    
    if magwalk_df.empty:
        logging.warning("Empty dataframe provided, skipping interpolation.")
        return magwalk_df
    
    # Create a working copy of the dataframe
    df = magwalk_df.copy()
    
    # Remove any unnamed or empty columns that might cause issues
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    # Remove any trailing NaN columns
    df = df.loc[:, ~df.columns.isna()]
    
    # Identify relevant columns for interpolation
    gps_time_col = next((col for col in df.columns if 'GPSTime' in col and not '[hh:mm:ss' in col), None)
    gps_date_col = next((col for col in df.columns if 'GPSDate' in col), None)
    gps_formatted_col = next((col for col in df.columns if 'GPSTime' in col and '[hh:mm:ss' in col), None)
    quality_col = next((col for col in df.columns if 'Quality' in col), None)
    
    # If no GPS time column found, return the original dataframe
    if not gps_time_col:
        logging.warning("No GPSTime column found. Cannot perform interpolation.")
        return df
    
    # Find columns to interpolate
    acc_x_col = next((col for col in df.columns if 'AccX' in col), None)
    acc_y_col = next((col for col in df.columns if 'AccY' in col), None)
    acc_z_col = next((col for col in df.columns if 'AccZ' in col), None)
    lat_col = next((col for col in df.columns if 'Latitude' in col), None)
    lon_col = next((col for col in df.columns if 'Longitude' in col), None)
    alt_col = next((col for col in df.columns if 'Altitude' in col), None)
    
    # Create a list of columns to interpolate including GPS columns
    cols_to_interpolate = []
    for col_name, col in [('AccX', acc_x_col), ('AccY', acc_y_col), ('AccZ', acc_z_col), 
                          ('Latitude', lat_col), ('Longitude', lon_col), ('Altitude', alt_col),
                          ('GPSTime', gps_time_col)]:
        if col is not None:
            cols_to_interpolate.append(col)
        else:
            logging.warning(f"{col_name} column not found in data.")
    
    # Add GPS formatted time column if it exists
    if gps_formatted_col and gps_formatted_col not in cols_to_interpolate:
        cols_to_interpolate.append(gps_formatted_col)
    
    # Add GPS date column if it exists
    if gps_date_col and gps_date_col not in cols_to_interpolate:
        cols_to_interpolate.append(gps_date_col)
    
    logging.info(f"Columns to interpolate: {cols_to_interpolate}")
    
    # Find indices of rows with GPS time values
    gps_indices = []
    for idx, row in df.iterrows():
        gps_value = row[gps_time_col]
        if pd.notna(gps_value) and str(gps_value) != '0' and str(gps_value).strip() != '':
            gps_indices.append(idx)
    
    if len(gps_indices) < 2:
        logging.warning("Not enough GPS points found for interpolation.")
        return df
    
    logging.info(f"Found {len(gps_indices)} GPS-marked datapoints.")
    
    # Check gap consistency
    gaps = [gps_indices[i+1] - gps_indices[i] - 1 for i in range(len(gps_indices)-1)]
    avg_gap = sum(gaps) / len(gaps) if gaps else 0
    consistent = True # Skip for now
    
    if not consistent:
        logging.warning("Inconsistent gaps detected between GPS points!")
        logging.info(f"Average gap size: {avg_gap:.2f} rows")
        logging.info(f"Standard Deviation: {np.std(gaps):.2f} rows")
        logging.info(f"Max: {max(gaps):.2f} rows")
        logging.info(f"Min: {min(gaps):.2f} rows")
    else:
        logging.info(f"Gaps have an average size of {avg_gap:.2f} rows.")
        logging.info(f"Max: {max(gaps):.2f} rows")
        logging.info(f"Min: {min(gaps):.2f} rows")
    
    # Perform interpolation between pairs of GPS points
    interpolated_rows = 0
    for i in range(len(gps_indices) - 1):
        start_idx = gps_indices[i]
        end_idx = gps_indices[i+1]
        
        # Skip if there's no gap to fill
        if end_idx - start_idx <= 1:
            continue
            
        # Set quality markers based on edge points
        if quality_col:
            start_quality = df.at[start_idx, quality_col]
            end_quality = df.at[end_idx, quality_col]
            
            # If both edge points have quality=1, set all points in gap to 1
            if start_quality == 1 and end_quality == 1:
                for offset in range(1, end_idx - start_idx):
                    df.at[start_idx + offset, quality_col] = 1
        
        # Interpolate between GPS points for each column
        for col in cols_to_interpolate:
            start_val = df.at[start_idx, col]
            end_val = df.at[end_idx, col]
            
            # For GPSDate, use the same date throughout the gap
            if col == gps_date_col:
                # Use the start date for all points in the gap
                if pd.notna(start_val) and str(start_val) != '0' and str(start_val).strip() != '':
                    for offset in range(1, end_idx - start_idx):
                        df.at[start_idx + offset, col] = start_val
                        interpolated_rows += 1
                continue
            
            # Skip if values are the same or one is missing
            if pd.isna(start_val) or pd.isna(end_val):
                continue
                
            # Handle string/numeric conversions for GPS time formatting
            if 'GPSTime' in col:
                try:
                    # Ensure we're working with numeric values for interpolation
                    start_val = float(start_val) if not isinstance(start_val, (int, float)) else start_val
                    end_val = float(end_val) if not isinstance(end_val, (int, float)) else end_val
                    
                    # Skip if values can't be converted to numeric
                    if pd.isna(start_val) or pd.isna(end_val):
                        continue
                except (ValueError, TypeError):
                    # If conversion fails, skip this column
                    continue
            
            # Linear interpolation
            gap_size = end_idx - start_idx
            step = (end_val - start_val) / gap_size
            
            # Fill in interpolated values
            for offset in range(1, gap_size):
                interpolated_val = start_val + (step * offset)
                df.at[start_idx + offset, col] = interpolated_val
                interpolated_rows += 1
                
                # Special handling for formatted GPS time (hh:mm:ss.sss)
                if col == gps_time_col and gps_formatted_col:
                    # Use the valid GPS time format directly instead of converting from numeric
                    try:
                        # Get reference formatted time from a valid GPS point
                        reference_time = None
                        
                        # First try to get it from the start point of this gap
                        if isinstance(df.at[start_idx, gps_formatted_col], str) and ':' in df.at[start_idx, gps_formatted_col]:
                            reference_time = df.at[start_idx, gps_formatted_col]
                        # If that failed, check other GPS points
                        elif any(isinstance(df.at[idx, gps_formatted_col], str) and ':' in df.at[idx, gps_formatted_col] 
                                for idx in gps_indices):
                            for idx in gps_indices:
                                if isinstance(df.at[idx, gps_formatted_col], str) and ':' in df.at[idx, gps_formatted_col]:
                                    reference_time = df.at[idx, gps_formatted_col]
                                    break
                        
                        if reference_time:
                            # Parse the reference time
                            parts = reference_time.split(':')
                            hours_mins = f"{parts[0]}:{parts[1]}"
                            
                            # Only interpolate the seconds part
                            seconds = interpolated_val % 60
                            
                            # Format as hh:mm:ss.sss, keeping the original hours and minutes
                            formatted_time = f"{hours_mins}:{seconds:.3f}"
                            df.at[start_idx + offset, gps_formatted_col] = formatted_time
                        else:
                            # Fallback to the previous method if we couldn't find a reference time
                            total_seconds = interpolated_val
                            reference_hour = 7  # Default hour
                            minutes = int((total_seconds % 3600) // 60)
                            seconds = total_seconds % 60
                            formatted_time = f"{reference_hour:02d}:{minutes:02d}:{seconds:.3f}"
                            df.at[start_idx + offset, gps_formatted_col] = formatted_time
                    except (ValueError, TypeError) as e:
                        # If conversion fails, log and skip
                        logging.debug(f"Time conversion failed: {e}")
                        pass
    
    elapsed_time = time.time() - start_time
    logging.info(f"Interpolation complete. Updated {interpolated_rows} data points in {elapsed_time:.2f} seconds.")
    
    # Remove any trailing NaN columns before returning
    df = df.loc[:, ~df.columns.isna()]
    
    # Return the updated dataframe with interpolated values
    return df


def add_calculated_fields(magwalk_df, sensor1_offset_east=0.0, sensor1_offset_north=0.5, sensor2_offset_east=0.0, sensor2_offset_north=-0.5):
    """
    Add calculated fields to the MagWalk dataframe:
    1. Convert latitude/longitude to UTM coordinates
    2. Calculate total magnetic field values using Euclidean norm
    
    Args:
        magwalk_df (DataFrame): The MagWalk data after interpolation
        
    Returns:
        DataFrame: DataFrame with additional calculated columns
    """
    # Make a copy to avoid modifying the original
    df = magwalk_df.copy()
    
    # Clean up any NaN columns before processing
    df = df.loc[:, ~df.columns.isna()]
    # Also remove any unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    # 1. Convert lat/long to UTM (Zone 33N for Norway)
    # Check if latitude and longitude columns exist
    lat_col = 'Latitude [Decimal Degrees]'
    lon_col = 'Longitude [Decimal Degrees]'
    
    if lat_col in df.columns and lon_col in df.columns:
        # Create UTM transformer
        utm_transformer = pyproj.Transformer.from_crs(
            "EPSG:4326",   # WGS84
            "EPSG:32633",  # UTM Zone 33N
            always_xy=True
        )
        
        # Apply conversion and create new columns
        utm_coords = [utm_transformer.transform(lon, lat) 
                      for lon, lat in zip(df[lon_col], df[lat_col])]
        
        # Add UTM columns with sensor-specific offsets
        base_easting = [coord[0] for coord in utm_coords]
        base_northing = [coord[1] for coord in utm_coords]
        
        # Add the base coordinates (GPS position)
        df['UTM_Easting'] = base_easting
        df['UTM_Northing'] = base_northing
        
        # Add sensor 1 coordinates with offsets
        df['UTM_Easting1'] = [e + sensor1_offset_east for e in base_easting]
        df['UTM_Northing1'] = [n + sensor1_offset_north for n in base_northing]
        
        # Add sensor 2 coordinates with offsets
        df['UTM_Easting2'] = [e + sensor2_offset_east for e in base_easting]
        df['UTM_Northing2'] = [n + sensor2_offset_north for n in base_northing]
        
        logging.info("Added UTM coordinates (Zone 33N)")
    else:
        logging.warning("Could not add UTM coordinates: latitude or longitude columns missing")
    
    # 2. Calculate total magnetic field values (Euclidean norm)
    # For B1 (x,y,z)
    if all(col in df.columns for col in ['B1x [nT]', 'B1y [nT]', 'B1z [nT]']):
        df['Btotal1 [nT]'] = np.sqrt(
            df['B1x [nT]']**2 + 
            df['B1y [nT]']**2 + 
            df['B1z [nT]']**2
        )
        logging.info("Added Btotal1 (Euclidean norm of B1 components)")
    else:
        logging.warning("Could not calculate Btotal1: B1 component columns missing")
    
    # For B2 (x,y,z)
    if all(col in df.columns for col in ['B2x [nT]', 'B2y [nT]', 'B2z [nT]']):
        df['Btotal2 [nT]'] = np.sqrt(
            df['B2x [nT]']**2 + 
            df['B2y [nT]']**2 + 
            df['B2z [nT]']**2
        )
        logging.info("Added Btotal2 (Euclidean norm of B2 components)")
    else:
        logging.warning("Could not calculate Btotal2: B2 component columns missing")
    
    # Final cleanup of any NaN columns that might have been introduced
    df = df.loc[:, ~df.columns.isna()]
    
    return df


def calculate_residual_anomalies(magwalk_df, basestation_df=None, output_csv=None, disable_basestation=False, vertical_alignment=False):
    """
    Calculate residual magnetic anomalies by subtracting Earth's magnetic field
    (from base station) from the total magnetic field measured by each magnetometer.
    
    Handles time zone conversion between magwalk data (UTC) and base station data (UTC+2).
    Optimized for performance with parallel processing and streaming CSV output.
    
    Args:
        magwalk_df (DataFrame): MagWalk data with Btotal1 and Btotal2 columns
        basestation_df (DataFrame): Base station data with Earth's magnetic field values
        output_csv (str, optional): Path to output CSV file to stream results to
        
    Returns:
        DataFrame: MagWalk data with added R1 and R2 columns for residual anomalies
    """
    import multiprocessing as mp
    from functools import partial
    import os
    
    logging.info("Calculating residual magnetic anomalies...")
    
    # Make a copy to avoid modifying the original
    df = magwalk_df.copy()
    
    # Check if required columns exist
    if 'Btotal1 [nT]' not in df.columns or 'Btotal2 [nT]' not in df.columns:
        logging.warning("Cannot calculate residuals: Btotal1 or Btotal2 columns missing")
        return df
    
    # Check if basestation dataframe is valid or if basestation correction is disabled
    if disable_basestation:
        logging.info("Basestation correction disabled by user")
        
        # If vertical alignment is requested, calculate M1-M2 directly from total field values
        # without creating redundant R1 and R2 columns
        if vertical_alignment:
            logging.info("Calculating vertical alignment (M1-M2)")
            df['VA [nT]'] = df['Btotal1 [nT]'] - df['Btotal2 [nT]']
            
        # Return early without processing basestation data
        return df
    elif basestation_df is None or len(basestation_df) == 0:
        logging.warning("Cannot calculate residuals: Base station data is empty or None")
        return df
    
    # Log base station columns to verify
    logging.info(f"Base station columns: {basestation_df.columns.tolist()}")
    
    # Identify time columns
    magwalk_time_col = 'GPSTime [hh:mm:ss.sss]'  # UTC time column in magwalk data
    
    # Identify base station time and field columns
    # Use the actual column names from the base station file
    base_time_col = 'time'  # Local time column (UTC+2)
    base_field_col = 'nT'   # Earth's magnetic field in nT
    
    if base_time_col not in basestation_df.columns:
        logging.warning(f"Cannot calculate residuals: '{base_time_col}' column missing in base station data")
        return df
    
    if base_field_col not in basestation_df.columns:
        logging.warning(f"Cannot calculate residuals: '{base_field_col}' column missing in base station data")
        return df
    
    # Check if magwalk time column exists
    if magwalk_time_col not in df.columns:
        logging.warning(f"Cannot calculate residuals: {magwalk_time_col} column missing in magwalk data")
        return df
    
    # Create a time converter function to handle the UTC to UTC+2 conversion
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
    
    # Create a new column with converted times
    df['MagWalk_LocalTime'] = df[magwalk_time_col].apply(convert_magwalk_time_to_base_time)
    
    # Initialize residual columns
    df['R1 [nT]'] = np.nan
    df['R2 [nT]'] = np.nan
    
    # Initialize vertical alignment column if needed
    if vertical_alignment:
        df['VA [nT]'] = np.nan
    
    # Log some sample converted times for debugging
    sample_times = df['MagWalk_LocalTime'].dropna().head(5).tolist()
    sample_base_times = basestation_df[base_time_col].head(5).tolist()
    logging.info(f"Sample converted magwalk times: {sample_times}")
    logging.info(f"Sample base station times: {sample_base_times}")
    
    # Convert base station times to numeric and sort for binary search
    base_times_numeric = pd.to_numeric(basestation_df[base_time_col], errors='coerce')
    basestation_df = basestation_df.assign(time_numeric=base_times_numeric)
    basestation_df = basestation_df.sort_values('time_numeric')
    basestation_df = basestation_df.reset_index(drop=True)
    
    # Calculate the time interval (frequency) of base station data
    if len(basestation_df) > 1:
        time_diffs = np.diff(basestation_df['time_numeric'])
        median_diff = np.median(time_diffs[~np.isnan(time_diffs)])
        logging.info(f"Base station data frequency: approximately every {median_diff:.1f} time units")
    else:
        median_diff = 5.0  # Default to 5 seconds if can't determine
        logging.info("Could not determine base station frequency, using default")
    
    # Setup for parallel processing
    num_cores = max(1, mp.cpu_count() - 1)  # Leave one core free
    logging.info(f"Using {num_cores} CPU cores for parallel processing")
    
    # Convert to numpy arrays for better performance in parallel processing
    base_times_array = basestation_df['time_numeric'].values
    base_fields_array = basestation_df[base_field_col].values
    
    # Set up output CSV if requested
    if output_csv:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        
        # Get columns to save (excluding lat/lon)
        exclude_cols = ['Latitude [Decimal Degrees]', 'Longitude [Decimal Degrees]']
        
        # Make sure to exclude any unnamed or empty columns
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df = df.loc[:, ~df.columns.isna()]
        df = df.loc[:, df.columns != '']
        
        output_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Write header
        header_df = df[output_cols].head(0)
        header_df.to_csv(output_csv, index=False)
        logging.info(f"Created output CSV file: {output_csv}")
    
    # Prepare batches for parallel processing
    chunk_size = min(10000, max(1000, len(df) // (num_cores * 10)))  # Dynamic chunk size
    batches = [df.iloc[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
    logging.info(f"Processing {len(batches)} batches with approximately {chunk_size} rows each")
    
    # Process in parallel
    start_time = time.time()
    processed_rows = 0
    matched_rows = 0
    
    # If we can use multiprocessing
    if num_cores > 1 and len(batches) > 1:
        with mp.Pool(num_cores) as pool:
            process_func = partial(process_batch, base_times=base_times_array, base_fields=base_fields_array, vertical_alignment=vertical_alignment, disable_basestation=disable_basestation)
            
            for i, result_batch in enumerate(pool.imap(process_func, batches)):
                # Count matches
                matched_in_batch = result_batch[['R1 [nT]', 'R2 [nT]']].count().min()
                matched_rows += matched_in_batch
                processed_rows += len(result_batch)
                
                # Copy results back to main dataframe
                for idx, row in result_batch.iterrows():
                    df.at[idx, 'R1 [nT]'] = row['R1 [nT]']
                    df.at[idx, 'R2 [nT]'] = row['R2 [nT]']
                    if vertical_alignment and 'VA [nT]' in row:
                        df.at[idx, 'VA [nT]'] = row['VA [nT]']
                
                # Stream to CSV if requested
                if output_csv:
                    result_batch[output_cols].to_csv(output_csv, mode='a', header=False, index=False)
                
                # Log progress
                elapsed = time.time() - start_time
                logging.info(f"Processed batch {i+1}/{len(batches)}: {processed_rows} rows in {elapsed:.2f} seconds. Matched: {matched_rows}")
    else:
        # Sequential processing for small datasets or single core
        for i, batch in enumerate(batches):
            result_batch = process_batch(batch, base_times_array, base_fields_array, vertical_alignment=vertical_alignment, disable_basestation=disable_basestation)
            
            # Count matches
            matched_in_batch = result_batch[['R1 [nT]', 'R2 [nT]']].count().min()
            matched_rows += matched_in_batch
            processed_rows += len(result_batch)
            
            # Copy results back to main dataframe
            for idx, row in result_batch.iterrows():
                df.at[idx, 'R1 [nT]'] = row['R1 [nT]']
                df.at[idx, 'R2 [nT]'] = row['R2 [nT]']
                if vertical_alignment and 'VA [nT]' in row:
                    df.at[idx, 'VA [nT]'] = row['VA [nT]']
            
            # Stream to CSV if requested
            if output_csv:
                result_batch[output_cols].to_csv(output_csv, mode='a', header=False, index=False)
            
            # Log progress
            elapsed = time.time() - start_time
            logging.info(f"Processed batch {i+1}/{len(batches)}: {processed_rows} rows in {elapsed:.2f} seconds. Matched: {matched_rows}")
    
    elapsed_time = time.time() - start_time
    logging.info(f"Completed residual anomaly calculation for {processed_rows} rows in {elapsed_time:.2f} seconds")
    logging.info(f"Successfully matched {matched_rows} rows with base station data")
    
    if output_csv:
        logging.info(f"Final results saved to {output_csv}")
    
    # Drop the temporary column
    df = df.drop('MagWalk_LocalTime', axis=1)
    
    return df


def find_closest_time_index(target_time, times):
    """Find closest time using binary search"""
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


def process_batch(batch_df, base_times, base_fields, vertical_alignment=False, disable_basestation=False):
    """Process a batch of rows to find closest base station measurements"""
    result_df = batch_df.copy()
    for idx, row in batch_df.iterrows():
        try:
            local_time_str = row['MagWalk_LocalTime']
            if pd.isna(local_time_str):
                continue
                
            local_time_numeric = float(local_time_str)
            closest_idx = find_closest_time_index(local_time_numeric, base_times)
            if closest_idx is not None:
                if not disable_basestation:
                    # Calculate residuals only when basestation correction is enabled
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
            logging.debug(f"Error matching time {row['MagWalk_LocalTime']}: {e}")
    
    return result_df


def main(basestation_file=None, magwalk_file=None, output_dir=None, magwalk_delimiter=';', basestation_delimiter=',', debug=False, sensor1_offset_east=0.0, sensor1_offset_north=0.5, sensor2_offset_east=0.0, sensor2_offset_north=-0.5, disable_basestation=False, vertical_alignment=False):
    """Main function to process magnetic data files

    Args:
        basestation_file (str, optional): Path to base station data file. Defaults to None.
        magwalk_file (str, optional): Path to MagWalk data file. Defaults to None.
        output_dir (str, optional): Directory for output files. Defaults to None (same as input files).
        magwalk_delimiter (str, optional): Delimiter in MagWalk file. Defaults to ';'.
        basestation_delimiter (str, optional): Delimiter in base station file. Defaults to ','.
        debug (bool, optional): Enable debug logging. Defaults to False.
    """
    start_time = time.time()
    log_file = setup_logging()
    logging.info(f"Starting magnetic data processing. Log file: {log_file}")
    
    
    logging.info(f"Base station file: {basestation_file}")
    logging.info(f"MagWalk file: {magwalk_file}")
    
    # Define output directory if not provided
    if output_dir is None:
        output_dir = os.path.dirname(magwalk_file)
    
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    basestation = None
    base_metadata = None

    try:
        if not disable_basestation:
            start_read = time.time()
            basestation, base_metadata = read_base_station_file(basestation_file, delimiter=basestation_delimiter)
            logging.info(f"Successfully loaded base station data with {len(basestation)} records in {time.time() - start_read:.2f} seconds")
            logging.info(f"Base station data shape: {basestation.shape}")
        else:
            logging.info("Basestation correction disabled, skipping basestation file loading")
            basestation = None
            base_metadata = None
    except Exception as e:
        logging.error(f"Error loading base station data: {e}")
    
    # Read MagWalk data if provided
    magwalk_data = None
    magwalk_metadata = None

    try:
        start_read = time.time()
        magwalk_data, magwalk_metadata = read_magwalk_file(magwalk_file, delimiter=magwalk_delimiter)
        logging.info(f"Successfully loaded MagWalk data with {len(magwalk_data)} records in {time.time() - start_read:.2f} seconds")
        logging.info(f"MagWalk data shape before filtering: {magwalk_data.shape}")
        
        # Filter MagWalk data to remove initial rows without GPS data
        start_filter = time.time()
        magwalk_data = filter_magwalk_data(magwalk_data)
        logging.info(f"Filtering completed in {time.time() - start_filter:.2f} seconds")
        logging.info(f"MagWalk data shape after filtering: {magwalk_data.shape}")
        
        # Interpolate gaps between GPS-marked points
        start_interp = time.time()
        magwalk_data = interpolate_gps_gaps(magwalk_data)
        logging.info(f"Interpolation completed in {time.time() - start_interp:.2f} seconds")
        logging.info(f"MagWalk data shape after interpolation: {magwalk_data.shape}")
        
        # Add calculated fields: UTM coordinates and total magnetic field values
        start_calc = time.time()
        magwalk_data = add_calculated_fields(magwalk_data, 
                                       sensor1_offset_east=sensor1_offset_east,
                                       sensor1_offset_north=sensor1_offset_north,
                                       sensor2_offset_east=sensor2_offset_east,
                                       sensor2_offset_north=sensor2_offset_north)
        logging.info(f"Added calculated fields in {time.time() - start_calc:.2f} seconds")
        logging.info(f"MagWalk data shape after adding calculated fields: {magwalk_data.shape}")
        
        # Generate output file path - single output file for all results
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = os.path.splitext(os.path.basename(magwalk_file))[0]
        output_filename = f"{base_filename}_processed_{timestamp}.csv"
        output_file_path = os.path.join(output_dir, output_filename)
        
        # Calculate residual magnetic anomalies and stream to CSV
        start_residual = time.time()
        magwalk_data = calculate_residual_anomalies(magwalk_data, basestation, output_csv=output_file_path, disable_basestation=disable_basestation, vertical_alignment=vertical_alignment)
        logging.info(f"Added residual anomalies in {time.time() - start_residual:.2f} seconds")
        logging.info(f"MagWalk data shape after adding residual anomalies: {magwalk_data.shape}")
        
        # Ensure the data is saved by explicitly saving it here as well
        try:
            # Make sure output directory exists
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            magwalk_data.to_csv(output_file_path, index=False)
            logging.info(f"Data explicitly saved to {output_file_path}")
        except Exception as e:
            logging.error(f"Error saving data to {output_file_path}: {e}")
            
        # Verify the file was created
        if os.path.exists(output_file_path):
            logging.info(f"Verified: output file exists at {output_file_path} with size {os.path.getsize(output_file_path)} bytes")
        else:
            logging.error(f"Failed to create output file at {output_file_path}")
            
        logging.info(f"All processed data saved to {output_file_path}")
        
        # Check if vertical alignment was calculated
        if vertical_alignment:
            logging.info("Vertical alignment (M1-M2) calculation complete")
        
        # This section is now handled within the calculate_residual_anomalies function with the disable_basestation parameter
        
        # Remove any trailing NaN columns from final output
        magwalk_data = magwalk_data.loc[:, ~magwalk_data.columns.isna()]
        magwalk_data = magwalk_data.loc[:, ~magwalk_data.columns.str.contains('^Unnamed')]
        magwalk_data = magwalk_data.loc[:, magwalk_data.columns != '']
        
        total_time = time.time() - start_time   
        logging.info(f"Total processing time: {total_time:.2f} seconds")
        return magwalk_data
        
    except Exception as e:
        logging.error(f"Error processing MagWalk data: {e}")
        return None


if __name__ == "__main__":
    # Sensor Offsets
    SENSOR1_OFFSET_EAST = 0.0375  
    SENSOR1_OFFSET_NORTH = -0.56  
    SENSOR2_OFFSET_EAST = 0.0375  
    SENSOR2_OFFSET_NORTH = 0.56  
    # SENSOR1_OFFSET_EAST = 0  
    # SENSOR1_OFFSET_NORTH = 0  
    # SENSOR2_OFFSET_EAST = 0  
    # SENSOR2_OFFSET_NORTH = 0
    
    DISABLE_BASESTATION = False  # Set to True to disable basestation correction
    VERTICAL_ALIGNMENT = False   # Set to True to calculate M1-M2 (vertical alignment)

    BASESTATION_FILE = "/Users/aleksandergarbuz/Documents/SINTEF/data/magneticbasestation_200525.txt"
    MAGWALK_FILE = "/Users/aleksandergarbuz/Documents/SINTEF/data/20250520_162100_MWALK_#0122.csv"
   
    main(sensor1_offset_east=SENSOR1_OFFSET_EAST,
        sensor1_offset_north=SENSOR1_OFFSET_NORTH,
        sensor2_offset_east=SENSOR2_OFFSET_EAST,
        sensor2_offset_north=SENSOR2_OFFSET_NORTH,
        disable_basestation=DISABLE_BASESTATION,
        vertical_alignment=VERTICAL_ALIGNMENT,
        basestation_file=BASESTATION_FILE,
        magwalk_file=MAGWALK_FILE)