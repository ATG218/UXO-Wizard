"""
CSV Reader for Magnetic Survey Data
===================================

Handles CSV files from magbase and flight_path_segmenter processing pipeline.
Supports automatic field detection and batch directory processing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
import logging

logger = logging.getLogger(__name__)


class CSVReader:
    """
    CSV reader optimized for magnetic survey data from magbase/flight_path_segmenter.
    
    Handles various magnetic field naming conventions and coordinate systems.
    """
    
    # Common magnetic field column names to search for
    MAGNETIC_FIELD_NAMES = [
        "R1 [nT]", "R2 [nT]", "R3 [nT]", "R4 [nT]",
        "Btotal1 [nT]", "Btotal2 [nT]", "Btotal3 [nT]", "Btotal4 [nT]",
        "Total [nT]", "B_total [nT]", "Mag [nT]", "Field [nT]",
        "TMI [nT]", "TMI",  # Total Magnetic Intensity
        "B1x [nT]", "B1y [nT]", "B1z [nT]", "B2x [nT]", "B2y [nT]", "B2z [nT]"  # Component measurements
    ]
    
    # Coordinate column names
    COORDINATE_NAMES = {
        'x': ['X', 'x', 'Easting', 'UTM_E', 'UTM_X', 'UTM_Easting', 'UTM_Easting1', 'UTM_Easting2', 'Longitude', 'Long', 'Lon'],
        'y': ['Y', 'y', 'Northing', 'UTM_N', 'UTM_Y', 'UTM_Northing', 'UTM_Northing1', 'UTM_Northing2', 'Latitude', 'Lat'],
        'elevation': ['Z', 'z', 'Elevation', 'Alt', 'Altitude', 'Height', 'Elev']
    }
    
    # Time/sequence columns
    TIME_NAMES = ['Time', 'DateTime', 'Timestamp', 'GPS_Time', 'Seconds']
    
    def __init__(self, target_field: str = "auto"):
        """
        Initialize CSV reader.
        
        Parameters:
            target_field (str): Specific magnetic field to use, or "auto" for detection
        """
        self.target_field = target_field
        
    def detect_magnetic_field(self, df: pd.DataFrame) -> Optional[str]:
        """
        Automatically detect the magnetic field column to use.
        
        Parameters:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            Optional[str]: Name of magnetic field column, or None if not found
        """
        available_fields = []
        
        for field_name in self.MAGNETIC_FIELD_NAMES:
            if field_name in df.columns:
                available_fields.append(field_name)
        
        if not available_fields:
            logger.warning("No magnetic field columns found")
            return None
            
        if self.target_field != "auto" and self.target_field in available_fields:
            return self.target_field
            
        # Default priority order
        priority_order = ["R1 [nT]", "Btotal1 [nT]", "Total [nT]", "TMI [nT]"]
        for field in priority_order:
            if field in available_fields:
                logger.info(f"Using magnetic field: {field}")
                return field
                
        # Use first available field
        field = available_fields[0]
        logger.info(f"Using magnetic field: {field}")
        return field
    
    def detect_coordinates(self, df: pd.DataFrame) -> Dict[str, Optional[str]]:
        """
        Detect coordinate columns in the dataframe.
        
        Parameters:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            Dict[str, Optional[str]]: Mapping of coordinate types to column names
        """
        coords = {'x': None, 'y': None, 'elevation': None}
        
        for coord_type, possible_names in self.COORDINATE_NAMES.items():
            for name in possible_names:
                if name in df.columns:
                    coords[coord_type] = name
                    break
        
        return coords
    
    def detect_time_column(self, df: pd.DataFrame) -> Optional[str]:
        """
        Detect time/sequence column.
        
        Parameters:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            Optional[str]: Name of time column, or None if not found
        """
        for name in self.TIME_NAMES:
            if name in df.columns:
                return name
        return None
    
    def standardize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize dataframe column names and add derived fields.
        
        Parameters:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Standardized dataframe
        """
        df_std = df.copy()
        
        # Detect and standardize magnetic field
        mag_field = self.detect_magnetic_field(df)
        if mag_field:
            df_std['magnetic_field'] = df_std[mag_field]
        
        # Detect and standardize coordinates
        coords = self.detect_coordinates(df)
        if coords['x']:
            df_std['x'] = df_std[coords['x']]
        if coords['y']:
            df_std['y'] = df_std[coords['y']]
        if coords['elevation']:
            df_std['elevation'] = df_std[coords['elevation']]
            
        # Detect time column
        time_col = self.detect_time_column(df)
        if time_col:
            df_std['time'] = df_std[time_col]
            
        return df_std
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate that dataframe contains required fields.
        
        Parameters:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            bool: True if data is valid
        """
        required_fields = ['magnetic_field', 'x', 'y']
        missing_fields = [field for field in required_fields if field not in df.columns]
        
        if missing_fields:
            logger.error(f"Missing required fields: {missing_fields}")
            return False
            
        # Check for sufficient data points
        if len(df) < 10:
            logger.error("Insufficient data points (< 10)")
            return False
            
        # Check for excessive NaN values
        for field in required_fields:
            nan_pct = df[field].isna().mean()
            if nan_pct > 0.5:
                logger.warning(f"Field {field} has {nan_pct:.1%} NaN values")
                
        return True
    
    def read_csv_file(self, file_path: Union[str, Path]) -> Optional[pd.DataFrame]:
        """
        Read and process a single CSV file.
        
        Parameters:
            file_path (Union[str, Path]): Path to CSV file
            
        Returns:
            Optional[pd.DataFrame]: Processed dataframe, or None if failed
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return None
            
        try:
            # Try different separators and encodings
            separators = [',', ';', '\t']
            encodings = ['utf-8', 'latin-1', 'cp1252']
            
            df = None
            for sep in separators:
                for encoding in encodings:
                    try:
                        df = pd.read_csv(file_path, sep=sep, encoding=encoding)
                        if len(df.columns) > 1:  # Successfully parsed
                            break
                    except (UnicodeDecodeError, pd.errors.ParserError):
                        continue
                if df is not None and len(df.columns) > 1:
                    break
            
            if df is None or len(df.columns) <= 1:
                logger.error(f"Could not parse CSV file: {file_path}")
                return None
                
            logger.info(f"Read {len(df)} rows from {file_path.name}")
            
            # Standardize the dataframe
            df_std = self.standardize_dataframe(df)
            
            # Validate data
            if not self.validate_data(df_std):
                logger.error(f"Data validation failed for: {file_path}")
                return None
                
            # Add metadata
            df_std['source_file'] = file_path.name
            df_std['file_path'] = str(file_path)
            
            return df_std
            
        except Exception as e:
            logger.error(f"Error reading {file_path}: {str(e)}")
            return None
    
    def read_csv_directory(self, directory_path: Union[str, Path]) -> List[pd.DataFrame]:
        """
        Read all CSV files from a directory.
        
        Parameters:
            directory_path (Union[str, Path]): Path to directory containing CSV files
            
        Returns:
            List[pd.DataFrame]: List of processed dataframes
        """
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            logger.error(f"Directory not found: {directory_path}")
            return []
            
        csv_files = list(directory_path.glob("*.csv"))
        if not csv_files:
            logger.warning(f"No CSV files found in: {directory_path}")
            return []
            
        logger.info(f"Found {len(csv_files)} CSV files")
        
        dataframes = []
        for csv_file in csv_files:
            df = self.read_csv_file(csv_file)
            if df is not None:
                dataframes.append(df)
                
        logger.info(f"Successfully loaded {len(dataframes)} CSV files")
        return dataframes
    
    def combine_dataframes(self, dataframes: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Combine multiple dataframes into a single dataset.
        
        Parameters:
            dataframes (List[pd.DataFrame]): List of dataframes to combine
            
        Returns:
            pd.DataFrame: Combined dataframe
        """
        if not dataframes:
            return pd.DataFrame()
            
        if len(dataframes) == 1:
            return dataframes[0]
            
        # Find common columns
        common_columns = set(dataframes[0].columns)
        for df in dataframes[1:]:
            common_columns &= set(df.columns)
            
        # Keep only common columns and combine
        combined_dfs = [df[list(common_columns)] for df in dataframes]
        combined = pd.concat(combined_dfs, ignore_index=True)
        
        logger.info(f"Combined {len(dataframes)} files into {len(combined)} total rows")
        return combined
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get summary statistics for the dataset.
        
        Parameters:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            Dict: Summary statistics
        """
        if df.empty:
            return {}
            
        summary = {
            'total_points': len(df),
            'unique_files': df['source_file'].nunique() if 'source_file' in df.columns else 1,
            'coordinate_system': 'UTM' if df['x'].max() > 1000 else 'Geographic',
            'x_range': (df['x'].min(), df['x'].max()),
            'y_range': (df['y'].min(), df['y'].max()),
            'magnetic_field_range': (df['magnetic_field'].min(), df['magnetic_field'].max()),
            'magnetic_field_stats': df['magnetic_field'].describe().to_dict()
        }
        
        if 'elevation' in df.columns:
            summary['elevation_range'] = (df['elevation'].min(), df['elevation'].max())
            
        return summary