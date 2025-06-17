import pandas as pd
import os

def combine_csv_files(input_files, output_file='combined_data.csv'):
    """
    Combine multiple CSV files into a single CSV file.
    
    Args:
        input_files: List of CSV file paths to combine
        output_file: Name of the output combined CSV file
    """
    
    dataframes = []
    
    print("Combining CSV files...")
    print("=" * 30)
    
    # Read each CSV file
    for i, file in enumerate(input_files):
        if os.path.exists(file):
            try:
                df = pd.read_csv(file)
                print(f"Loaded {file}: {len(df)} rows, {len(df.columns)} columns")
                
                # Add a source column to track which file each row came from
                df['source_file'] = file
                
                dataframes.append(df)
                
            except Exception as e:
                print(f"Error reading {file}: {e}")
        else:
            print(f"Warning: {file} not found")
    
    if not dataframes:
        print("No valid CSV files found to combine!")
        return None
    
    # Combine all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    print(f"\nCombined data summary:")
    print(f"Total rows: {len(combined_df)}")
    print(f"Total columns: {len(combined_df.columns)}")
    
    # If there's a timestamp column, sort by it
    if 'timestamp' in combined_df.columns:
        print("Sorting by timestamp...")
        combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
        
        # Show time range
        start_time = combined_df['timestamp'].min()
        end_time = combined_df['timestamp'].max()
        duration_ms = end_time - start_time
        print(f"Time range: {start_time} to {end_time}")
        print(f"Total duration: {duration_ms/1000:.1f} seconds ({duration_ms/60000:.1f} minutes)")
    
    # Show breakdown by source file
    print(f"\nBreakdown by source file:")
    source_counts = combined_df['source_file'].value_counts()
    for source, count in source_counts.items():
        print(f"  {source}: {count} rows")
    
    # Save combined data
    combined_df.to_csv(output_file, index=False)
    print(f"\nCombined data saved to: {output_file}")
    
    return combined_df

def main():
    """Main function to combine the three test CSV files."""
    
    # Define the input files
    input_files = ['tarva/concentrations_tarva_1.csv', 'tarva/concentrations_tarva_2.csv', 'tarva/concentrations_tarva_3.csv']
    output_file = 'concentrations_tarva_clean.csv'
    
    print("CSV File Combiner")
    print("=" * 20)
    
    # Check which files exist
    existing_files = []
    for file in input_files:
        if os.path.exists(file):
            existing_files.append(file)
            print(f"✓ Found: {file}")
        else:
            print(f"✗ Missing: {file}")
    
    if not existing_files:
        print("No CSV files found to combine!")
        return
    
    print(f"\nProceeding with {len(existing_files)} files...")
    
    # Combine the files
    combined_df = combine_csv_files(existing_files, output_file)
    
    if combined_df is not None:
        print(f"\nSuccess! All files combined into {output_file}")
        
        # Optional: Remove the source_file column if you don't want it
        choice = input("\nDo you want to remove the 'source_file' tracking column? (y/n): ").lower()
        if choice == 'y':
            combined_df_clean = combined_df.drop(columns=['source_file'])
            output_file_clean = output_file.replace('.csv', '_clean.csv')
            combined_df_clean.to_csv(output_file_clean, index=False)
            print(f"Clean version (without source_file column) saved as: {output_file_clean}")

if __name__ == "__main__":
    main() 