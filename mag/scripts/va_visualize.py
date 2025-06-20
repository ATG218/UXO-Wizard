#!/usr/bin/env python3
"""
Vertical Alignment Visualization Tool

This script reads processed MagWalk CSV files and creates visualizations focusing on
vertical alignment (VA) data, which represents the difference between magnetometer 1 
and magnetometer 2 readings (M1-M2). It generates time series plots, interactive
visualizations, and interpolated grid maps of the vertical alignment data.

Usage:
    python va_visualize.py [-f CSV_PATH] [-o OUTPUT_DIR]

Example:
    python va_visualize.py -f /path/to/processed_file.csv -o /path/to/output_directory
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
from scipy.interpolate import griddata
from matplotlib.colors import Normalize

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Visualize vertical alignment data from processed MagWalk files.")
    parser.add_argument("-f", "--file", help="Path to processed CSV file", required=False)
    parser.add_argument("-o", "--outdir", help="Directory for saving output plots", required=False)
    
    args = parser.parse_args()
    return args

def read_magnetic_data(file_path):
    """Read magnetic data from CSV file into a pandas DataFrame"""
    try:
        # Try with different delimiters
        try:
            # First try with comma delimiter (standard CSV)
            df = pd.read_csv(file_path)
        except Exception as e1:
            print(f"Failed with comma delimiter: {e1}")
            # If that fails, try with semicolon (as used in MagWalk files)
            df = pd.read_csv(file_path, delimiter=';')
            
        print(f"Successfully loaded data from {file_path}")
        print(f"Number of rows: {len(df)}")
        
        # Show column information
        print("Columns in the file:")
        for i, col in enumerate(df.columns):
            print(f"  {i}: {col}")
            
        # Check if vertical alignment data exists
        if 'VA [nT]' in df.columns:
            print("\nVertical alignment data detected!")
            print(f"VA [nT] statistics:")
            print(f"  Min: {df['VA [nT]'].min():.2f} nT")
            print(f"  Max: {df['VA [nT]'].max():.2f} nT")
            print(f"  Mean: {df['VA [nT]'].mean():.2f} nT")
            print(f"  Std Dev: {df['VA [nT]'].std():.2f} nT")
        else:
            print("\nWARNING: No vertical alignment data (VA [nT]) found in the file.")
            print("This visualization tool is designed for files containing vertical alignment data.")
            print("Please ensure you're using a file processed with the vertical_alignment option enabled.")
            return None
        
        # Show the first few rows to understand the data structure
        print("\nFirst few rows of data:")
        print(df.head())
        
        return df
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None

def plot_vertical_alignment_over_time(df, output_dir):
    """Plot vertical alignment data over time"""
    plt.figure(figsize=(12, 6))
    
    # Create an index for the x-axis that represents samples rather than raw milliseconds
    x_index = range(len(df))
    
    plt.plot(x_index, df['VA [nT]'], 'g-', label='Vertical Alignment (M1-M2)', linewidth=1.5)
    
    plt.title('Vertical Alignment Over Time')
    
    # Format x-axis to show meaningful labels at intervals
    # Add tick marks at regular intervals
    num_ticks = 10  # Number of ticks to show on x-axis
    tick_positions = [int(i * len(df) / num_ticks) for i in range(num_ticks+1)]
    
    # If GPS time is available, use it for labels
    if 'GPSTime [hh:mm:ss.sss]' in df.columns:
        tick_labels = [df['GPSTime [hh:mm:ss.sss]'].iloc[pos] if pos < len(df) else '' for pos in tick_positions]
        plt.xticks(tick_positions, tick_labels, rotation=45)
        plt.xlabel('GPS Time')
    else:
        # Otherwise use sample numbers
        plt.xticks(tick_positions)
        plt.xlabel('Sample Number')
    
    plt.ylabel('Vertical Alignment (nT)')
    plt.grid(True)
    plt.tight_layout()  # Adjust layout to fit labels
    
    output_path = os.path.join(output_dir, 'vertical_alignment_over_time.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Vertical alignment plot saved to {output_path}")
    
    # Create an interactive version with plotly
    try:
        # Create interactive directory
        interactive_dir = os.path.join(output_dir, 'interactive_plots')
        os.makedirs(interactive_dir, exist_ok=True)
        
        # Create the interactive plot
        fig = go.Figure()
        
        # Add trace for vertical alignment
        fig.add_trace(go.Scatter(
            x=x_index, 
            y=df['VA [nT]'],
            mode='lines',
            name='Vertical Alignment (M1-M2)',
            line=dict(color='green', width=2)
        ))
        
        # Update layout
        fig.update_layout(
            title='Interactive Vertical Alignment Over Time',
            xaxis_title='Sample Number',
            yaxis_title='Vertical Alignment (nT)',
            hovermode='closest',
            width=1200,
            height=600,
        )
        
        # If GPS time is available, use it for hover text
        if 'GPSTime [hh:mm:ss.sss]' in df.columns:
            # Create hover text with GPS time
            hover_text = [f"GPS Time: {time}<br>VA: {va:.2f} nT" 
                          for time, va in zip(df['GPSTime [hh:mm:ss.sss]'], df['VA [nT]'])]
            fig.data[0].hovertext = hover_text
            fig.data[0].hoverinfo = 'text'
        
        # Add grid lines
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        # Save to HTML file for interactive viewing
        output_path = os.path.join(interactive_dir, 'vertical_alignment_interactive.html')
        fig.write_html(output_path)
        print(f"Interactive vertical alignment plot saved to {output_path}")
        
    except Exception as e:
        print(f"Could not create interactive plot: {e}")

def create_interpolated_va_grid(df, output_dir):
    """
    Generate an interpolated grid visualization of the vertical alignment data.
    This shows the VA values across the surveyed area.
    """
    print("Creating interpolated vertical alignment grid...")
    
    # Extract coordinates and VA values
    easting = df['UTM_Easting'].values
    northing = df['UTM_Northing'].values
    va_values = df['VA [nT]'].values
    
    # Define the grid for interpolation
    grid_resolution = 100  # Number of points in each dimension
    easting_min, easting_max = easting.min(), easting.max()
    northing_min, northing_max = northing.min(), northing.max()
    
    # Add a small buffer to the min/max values
    buffer = 0.05  # 5% buffer
    easting_range = easting_max - easting_min
    northing_range = northing_max - northing_min
    easting_min -= easting_range * buffer
    easting_max += easting_range * buffer
    northing_min -= northing_range * buffer
    northing_max += northing_range * buffer
    
    # Create a regular grid for interpolation
    xi = np.linspace(easting_min, easting_max, grid_resolution)
    yi = np.linspace(northing_min, northing_max, grid_resolution)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    points = np.column_stack((easting, northing))
    
    # Interpolate the VA values onto the grid using cubic interpolation
    va_grid = griddata(points, va_values, (xi_grid, yi_grid), method='cubic')
    
    # Create figure for the standard plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create contour plot
    contour = ax.contourf(xi_grid, yi_grid, va_grid, levels=50, cmap='viridis')
    
    # Add scatter points showing the actual data points
    scatter = ax.scatter(easting, northing, c=va_values, cmap='viridis', s=5, alpha=0.5)
    
    # Add colorbar
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label('Vertical Alignment (nT)')
    
    # Add flight path visualization
    ax.plot(easting, northing, 'k-', linewidth=0.5, alpha=0.5)
    
    # Add markers for start and end points
    ax.plot(easting[0], northing[0], 'ko', markersize=6, label='Start')
    ax.plot(easting[-1], northing[-1], 'kx', markersize=6, label='End')
    
    # Set labels and title
    ax.set_title('Vertical Alignment (M1-M2) Interpolated Grid')
    ax.set_xlabel('UTM Easting (m)')
    ax.set_ylabel('UTM Northing (m)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Save the figure
    output_path = os.path.join(output_dir, 'va_interpolated_grid.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Interpolated vertical alignment grid saved to {output_path}")
    
    # Create an interactive version with plotly
    try:
        # Create directory for interactive plots
        interactive_dir = os.path.join(output_dir, 'interactive_plots')
        os.makedirs(interactive_dir, exist_ok=True)
        
        # Create the interactive heatmap
        fig = go.Figure()
        
        # Add heatmap trace for interpolated VA values
        fig.add_trace(
            go.Heatmap(
                x=xi,
                y=yi,
                z=va_grid,
                colorscale='Viridis',
                colorbar=dict(title='nT'),
            )
        )
        
        # Add scatter trace showing actual data points
        fig.add_trace(
            go.Scatter(
                x=easting,
                y=northing,
                mode='lines+markers',
                marker=dict(color='black', size=3, opacity=0.5),
                line=dict(color='black', width=1, dash='dot'),
                name='Flight Path',
                showlegend=True
            )
        )
        
        # Update layout
        fig.update_layout(
            title='Interactive Vertical Alignment (M1-M2) Interpolated Grid',
            xaxis_title='UTM Easting (m)',
            yaxis_title='UTM Northing (m)',
            height=800,
            width=1000,
        )
        
        # Set equal aspect ratio
        fig.update_yaxes(scaleanchor='x', scaleratio=1)
        
        # Add grid lines
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        # Save to HTML file for interactive viewing
        output_path = os.path.join(interactive_dir, 'va_interpolated_grid_interactive.html')
        fig.write_html(output_path)
        print(f"Interactive interpolated VA grid saved to {output_path}")
        
    except Exception as e:
        print(f"Could not create interactive grid plot: {e}")

def filter_extreme_outliers(df, column='VA [nT]', factor=3.0):
    """
    Filter out extreme outliers in the data using the IQR method.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with data
    column : str
        Column name to filter for outliers
    factor : float
        Multiplier for IQR to determine outlier threshold (default: 3.0)
        
    Returns:
    --------
    pandas.DataFrame
        Filtered dataframe with outliers removed
    """
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    
    # Define outlier thresholds
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    # Filter out the outliers
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)].copy()
    
    # Report on filtering
    outliers_removed = len(df) - len(filtered_df)
    if outliers_removed > 0:
        print(f"Removed {outliers_removed} outliers ({outliers_removed/len(df)*100:.2f}% of data)")
        print(f"Outlier thresholds for {column}: < {lower_bound:.2f} or > {upper_bound:.2f}")
    else:
        print(f"No outliers found using IQR method with factor {factor}")
    
    return filtered_df

def main():
    """Main function to run the visualization process"""
    args = parse_arguments()
    
    # If file path is not provided via command line, use a default or ask user
    if args.file:
        csv_path = args.file
    else:
        # Default file path as in magbase_post.py or prompt user
        csv_path = input("Enter the path to the processed CSV file: ")
    
    # Load the data
    df = read_magnetic_data(csv_path)
    if df is None:
        print("Error loading data. Exiting.")
        return
    
    # Check if vertical alignment data exists
    if 'VA [nT]' not in df.columns:
        print("No vertical alignment (VA) data found in the file. Exiting.")
        return
    
    # Set up output directory
    if args.outdir:
        output_dir = args.outdir
    else:
        # Create output directory based on input file location
        output_dir = os.path.join(os.path.dirname(os.path.dirname(csv_path)), 'VA_Visualization')
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output will be saved to: {output_dir}")
    
    # Create interactive plots directory
    interactive_dir = os.path.join(output_dir, 'interactive_plots')
    os.makedirs(interactive_dir, exist_ok=True)
    
    # Plot vertical alignment over time
    plot_vertical_alignment_over_time(df, output_dir)
    
    # Create raw and filtered versions of the interpolated grid
    print("\nCreating interpolated grid with all data points...")
    create_interpolated_va_grid(df, output_dir)
    
    # Create a version with extreme outliers removed
    print("\nCreating interpolated grid with outliers removed...")
    filtered_df = filter_extreme_outliers(df, column='VA [nT]', factor=3.0)
    if len(filtered_df) < len(df):
        # Save filtered version with "_filtered" suffix in output directory
        filtered_output_dir = os.path.join(output_dir, 'filtered')
        os.makedirs(filtered_output_dir, exist_ok=True)
        create_interpolated_va_grid(filtered_df, filtered_output_dir)
    
    print("\nAll visualizations complete!")

if __name__ == "__main__":
    main()
