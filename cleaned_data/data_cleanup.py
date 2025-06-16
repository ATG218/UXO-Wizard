#!/usr/bin/env python3
"""
Tarva Gamma Sensor Data Cleanup and Analysis Tool
Cleans up the malformed JSON file and converts it to usable formats
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

def fix_json_structure(file_path):
    """
    Fix the malformed JSON structure by extracting objects from arrays.
    Handle different timestamp patterns and missing timestamps.
    """
    fixed_data = []
    
    print(f"Reading and processing {file_path}...")
    
    with open(file_path, 'r') as file:
        for line_num, line in enumerate(file, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                # Parse the line as JSON array
                data_array = json.loads(line)
                
                # Extract the object from the array
                if isinstance(data_array, list) and len(data_array) > 0:
                    obj = data_array[0]
                    
                    # Handle different timestamp patterns
                    timestamp = None
                    
                    # Pattern 1: vT at the beginning or middle
                    if 'vT' in obj:
                        timestamp = obj['vT']
                    # Pattern 2: No timestamp (metadata or some sensor records)
                    else:
                        # For GPS_0007 records, we can sometimes use the Date field
                        if obj.get('eID') == 'GPS_0007' and 'v' in obj and 'Date' in obj['v']:
                            timestamp = obj['v']['Date']
                        else:
                            # No timestamp available - we'll handle this later
                            timestamp = None
                    
                    # Add timestamp info to the object for easier processing
                    processed_obj = obj.copy()
                    processed_obj['processed_timestamp'] = timestamp
                    processed_obj['line_number'] = line_num
                    
                    fixed_data.append(processed_obj)
                    
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse line {line_num}: {e}")
                print(f"  Line content: {line[:100]}...")
                continue
    
    print(f"Successfully processed {len(fixed_data)} records")
    return fixed_data

def categorize_sensor_data(data):
    """
    Categorize sensor data by type and create separate datasets.
    """
    categories = {
        'metadata': [],
        'gps_external': [],
        'gps_internal': [],
        'altitude': [],
        'environmental': [],
        'spectrometer': [],
        'stabilized_spectrometer': [],
        'sync_data': []
    }
    
    for record in data:
        eID = record.get('eID', '')
        
        # Categorize based on eID
        if not eID:  # Metadata records
            categories['metadata'].append(record)
        elif eID == 'GPS_external':
            categories['gps_external'].append(record)
        elif eID == 'GPS_0007':
            categories['gps_internal'].append(record)
        elif eID == 'ALT_external':
            categories['altitude'].append(record)
        elif eID == 'PTH_0007':
            categories['environmental'].append(record)
        elif eID.startswith('SPECTRO_'):
            categories['spectrometer'].append(record)
        elif eID.startswith('STABSPECTRO_'):
            categories['stabilized_spectrometer'].append(record)
        elif eID.startswith('SYNC_'):
            categories['sync_data'].append(record)
    
    return categories

def convert_to_dataframes(categories):
    """
    Convert categorized data to pandas DataFrames with proper timestamp handling.
    """
    dataframes = {}
    
    for category, records in categories.items():
        if not records:
            print(f"No data found for category: {category}")
            continue
            
        df_data = []
        
        for record in records:
            # Extract timestamp
            timestamp = record.get('processed_timestamp')
            
            # Flatten the record
            flattened = {'timestamp': timestamp, 'eID': record.get('eID')}
            
            # Add sensor values if present
            if 'v' in record:
                for key, value in record['v'].items():
                    # Handle arrays (like spectrum data) by converting to string for now
                    if isinstance(value, list):
                        if key == 'Spectrum':
                            # For spectrum data, we might want to keep it as array or save separately
                            flattened[f'{key}_length'] = len(value)
                            flattened[f'{key}_sum'] = sum(value) if all(isinstance(x, (int, float)) for x in value) else None
                            flattened[f'{key}_max'] = max(value) if all(isinstance(x, (int, float)) for x in value) else None
                        else:
                            flattened[key] = str(value)
                    else:
                        flattened[key] = value
            
            # Add other fields directly
            for key, value in record.items():
                if key not in ['v', 'processed_timestamp', 'line_number', 'eID']:
                    if isinstance(value, (dict, list)):
                        flattened[key] = str(value)
                    else:
                        flattened[key] = value
            
            df_data.append(flattened)
        
        if df_data:
            df = pd.DataFrame(df_data)
            
            # Convert timestamp to datetime if possible
            if 'timestamp' in df.columns and df['timestamp'].notna().any():
                # Handle millisecond timestamps
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
                # Sort by timestamp where available
                df = df.sort_values('timestamp', na_position='first')
            
            dataframes[category] = df
            print(f"Created DataFrame for {category}: {len(df)} records")
        
    return dataframes

def generate_summary_report(dataframes):
    """
    Generate a comprehensive summary report of the data.
    """
    print("\n" + "="*60)
    print("DATA SUMMARY REPORT")
    print("="*60)
    
    total_records = sum(len(df) for df in dataframes.values())
    print(f"Total Records Processed: {total_records}")
    print(f"Data Categories Found: {len(dataframes)}")
    
    for category, df in dataframes.items():
        print(f"\n{category.upper()}:")
        print(f"  Records: {len(df)}")
        print(f"  Columns: {list(df.columns)}")
        
        if 'timestamp' in df.columns:
            valid_timestamps = df['timestamp'].notna().sum()
            print(f"  Valid timestamps: {valid_timestamps}/{len(df)}")
            
            if valid_timestamps > 0:
                # Time range
                min_time = df['timestamp'].min()
                max_time = df['timestamp'].max()
                print(f"  Time range: {min_time} to {max_time}")
                
                if 'datetime' in df.columns:
                    min_dt = df['datetime'].min()
                    max_dt = df['datetime'].max()
                    print(f"  DateTime range: {min_dt} to {max_dt}")
        
        # Show sample data
        if len(df) > 0:
            print(f"  Sample data:")
            print(f"    {df.iloc[0].to_dict()}")

def save_cleaned_data(dataframes, output_dir="cleaned_data"):
    """
    Save cleaned data to CSV files.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSaving cleaned data to {output_dir}/")
    
    for category, df in dataframes.items():
        output_file = os.path.join(output_dir, f"{category}.csv")
        df.to_csv(output_file, index=False)
        print(f"  Saved {category}: {output_file} ({len(df)} records)")
    
    # Save combined summary
    summary_file = os.path.join(output_dir, "data_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("TARVA GAMMA1 DATA SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        total_records = sum(len(df) for df in dataframes.values())
        f.write(f"Total Records: {total_records}\n")
        f.write(f"Categories: {len(dataframes)}\n\n")
        
        for category, df in dataframes.items():
            f.write(f"{category.upper()}:\n")
            f.write(f"  Records: {len(df)}\n")
            f.write(f"  Columns: {len(df.columns)}\n")
            
            if 'timestamp' in df.columns:
                valid_timestamps = df['timestamp'].notna().sum()
                f.write(f"  Valid timestamps: {valid_timestamps}/{len(df)}\n")
            
            f.write("\n")
    
    print(f"  Saved summary: {summary_file}")

def main():
    """
    Main function to process the Tarva Gamma1 JSON file.
    """
    input_file = "Tarva_Gamma1.json"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        return
    
    print("Starting Tarva Gamma1 data cleanup...")
    
    # Step 1: Fix JSON structure and handle timestamps
    fixed_data = fix_json_structure(input_file)
    
    # Step 2: Categorize data by sensor type
    categories = categorize_sensor_data(fixed_data)
    
    # Step 3: Convert to DataFrames
    dataframes = convert_to_dataframes(categories)
    
    # Step 4: Generate summary report
    generate_summary_report(dataframes)
    
    # Step 5: Save cleaned data
    save_cleaned_data(dataframes)
    
    print("\nData cleanup completed successfully!")
    print("\nNext steps:")
    print("1. Check the 'cleaned_data/' directory for CSV files")
    print("2. Review the data_summary.txt file")
    print("3. Use the CSV files for further analysis and visualization")

if __name__ == "__main__":
    main()