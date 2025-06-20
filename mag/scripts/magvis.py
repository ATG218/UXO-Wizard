#!/usr/bin/env python3
"""
Magnetic Data Visualization Tool

This script reads processed MagWalk CSV files and creates visualizations focusing on
vertical alignment (VA) data or residual data. It generates time series plots, 
interactive visualizations, and interpolated grid maps of the data.
"""

import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import utm
import plotly.graph_objects as go
from scipy.interpolate import griddata

# ====================== CONFIGURATION VARIABLES ======================
# Input file path
CSV_PATH = '/Users/aleksandergarbuz/Documents/SINTEF/data/20250611_081139_MWALK_#0122_processed_20250616_094044.csv'

# Output directory - leave empty to auto-create based on input file location
OUTPUT_DIR = '/Users/aleksandergarbuz/Documents/SINTEF/data/081139/'

# Plot options
PLOT_TIME_SERIES = True          # Plot time series of magnetic data
PLOT_INTERPOLATED_GRID = True    # Plot interpolated grid of magnetic data
CREATE_INTERACTIVE_PLOTS = True  # Create interactive HTML versions of plots
FILTER_OUTLIERS = True           # Filter extreme outliers for cleaner plots
OUTLIER_FACTOR = 3.0             # IQR multiplier for outlier detection

# Grid interpolation settings
GRID_RESOLUTION = 100            # Number of points in each dimension for grid
GRID_BUFFER_PERCENT = 0.05       # Buffer percentage around min/max coordinates
USE_CUBIC_INTERPOLATION = True   # Use cubic interpolation (otherwise linear)
COLOR_MAP = 'viridis'            # Matplotlib colormap for plots

# Cutoff settings - control which portions of data to use in visualizations
# Format: [beginning_pct, end_pct, [middle_cut_points...]]
# beginning_pct: percentage to cut from beginning (0-100)
# end_pct: percentage to cut from end (0-100)
# middle_cut_points: list of percentage points where to make cuts in the middle
#                    each pair of values represents start and end of a cut section
# Example: [10, 5, [40, 60]] - Cut 10% from start, 5% from end, and 40-60% in middle
# Leave empty to use all data: [0, 0, []]
CUTOFFS = []

# Map background settings
USE_SATELLITE_BACKGROUND = True    # Use satellite imagery as background in interactive plots
MAPBOX_STYLE = 'satellite'         # Options: satellite, satellite-streets, outdoors, light, dark, streets
# No access token needed for these base styles when using Plotly

# =================================================================

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
            
        # Detect available data types
        available_data = {}
        
        # Check for vertical alignment data
        if 'VA [nT]' in df.columns:
            available_data['va'] = 'VA [nT]'
            print("Vertical anomaly data detected!")
            print("VA [nT] statistics:")
            print(f"  Min: {df['VA [nT]'].min():.2f} nT")
            print(f"  Max: {df['VA [nT]'].max():.2f} nT")
            print(f"  Mean: {df['VA [nT]'].mean():.2f} nT")
            print(f"  Std Dev: {df['VA [nT]'].std():.2f} nT")
        else:
            print("WARNING: No vertical anomaly data (VA [nT]) found in the file.")
        
        # Check for residual data
        residual_cols = []
        for col in df.columns:
            if col.startswith('R') and col.endswith(' [nT]'):
                residual_cols.append(col)
                available_data[f'residual_{col}'] = col
                print(f"\nResidual data detected: {col}")
                print(f"{col} statistics:")
                print(f"  Min: {df[col].min():.2f} nT")
                print(f"  Max: {df[col].max():.2f} nT")
                print(f"  Mean: {df[col].mean():.2f} nT")
                print(f"  Std Dev: {df[col].std():.2f} nT")
                
        # Check for total field data
        total_cols = []
        for col in df.columns:
            if col.startswith('Btotal') and col.endswith(' [nT]'):
                total_cols.append(col)
                available_data[f'total_{col}'] = col
        
        # Check for necessary coordinate data
        if 'UTM_Easting' not in df.columns or 'UTM_Northing' not in df.columns:
            print("\nWARNING: Required coordinate columns not found (UTM_Easting, UTM_Northing)")
            print("Cannot create spatial plots without coordinate data.")
            if not (available_data.get('va') or residual_cols):
                return None
            
        # Show the first few rows to understand the data structure
        print("\nFirst few rows of data:")
        print(df.head())
        
        # Report on available data types
        print("\nAvailable data types for visualization:")
        for data_type, col_name in available_data.items():
            print(f"  - {data_type}: {col_name}")
            
        # Add metadata to the dataframe
        df.attrs['available_data'] = available_data
        
        return df
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None

def plot_data_over_time(df, output_dir, data_column, title, y_label, file_prefix, csv_info=""):
    """
    Plot magnetic data over time using both matplotlib and plotly (if enabled).
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data to plot
    output_dir : str
        Directory to save the plot
    data_column : str
        Column name in df containing the data to plot
    title : str
        Title for the plot
    y_label : str
        Label for the y-axis
    file_prefix : str
        Prefix for the saved file
    csv_info : str, optional
        Identifier information from the CSV filename to append to titles and filenames
    """
    # Create static matplotlib plot
    plt.figure(figsize=(12, 6))
    
    # Create an index for the x-axis that represents samples rather than raw milliseconds
    x_index = list(range(len(df)))
    
    plt.plot(x_index, df[data_column], 'g-', label=y_label, linewidth=1.5)
    
    # Add CSV info to title if provided
    if csv_info:
        full_title = f"{title} - {csv_info}"
    else:
        full_title = title
    plt.title(full_title, fontsize=14)
    
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
    
    plt.ylabel(y_label)
    plt.grid(True)
    plt.tight_layout()  # Adjust layout to fit labels
    
    # Save the plot with CSV info in filename if provided
    if csv_info:
        output_path = os.path.join(output_dir, f"{file_prefix}_{csv_info}_over_time.png")
    else:
        output_path = os.path.join(output_dir, f"{file_prefix}_over_time.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"{full_title} plot saved to {output_path}")
    
    # Create an interactive version with plotly if enabled
    if CREATE_INTERACTIVE_PLOTS:
        try:
            # Create interactive directory
            interactive_dir = os.path.join(output_dir, 'interactive_plots')
            os.makedirs(interactive_dir, exist_ok=True)
            
            # Create an interactive plot using plotly
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(x=x_index, y=df[data_column], mode='lines', name=y_label))
            
            fig.update_layout(
                title=full_title,
                xaxis_title='Sample Index',
                yaxis_title=y_label,
                template='plotly_white'
            )
            
            # Update layout
            fig.update_layout(
                title=f'Interactive {title}',
                xaxis_title='Sample Index',
                yaxis_title=y_label,
                hovermode='closest',
                width=1200,
                height=600,
            )
            
            # If GPS time is available, use it for hover text
            if 'GPSTime [hh:mm:ss.sss]' in df.columns:
                # Create hover text with GPS time
                hover_text = [f"GPS Time: {time}<br>Value: {val:.2f} nT" 
                            for time, val in zip(df['GPSTime [hh:mm:ss.sss]'], df[data_column])]
                fig.data[0].hovertext = hover_text
                fig.data[0].hoverinfo = 'text'
            
            # Add grid lines
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            
            # Save the interactive plot with CSV info in filename if provided
            if csv_info:
                output_path = os.path.join(interactive_dir, f"{file_prefix}_{csv_info}_interactive.html")
            else:
                if file_prefix:
                    output_path = os.path.join(interactive_dir, f"{file_prefix}_interactive.html")
                else:
                    output_path = os.path.join(interactive_dir, "magnetic_data_interactive.html")
                
            fig.write_html(output_path)
            print(f"Interactive {full_title} plot saved to {output_path}")
            
        except Exception as e:
            print(f"Could not create interactive plot: {e}")

def create_interpolated_grid(df, output_dir, data_column, title, file_prefix, csv_info=""):
    """
    Generate an interpolated grid visualization of the data.
    This shows the values across the surveyed area.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data to plot
    output_dir : str
        Directory to save the plot
    data_column : str
        Column name in df containing the data to plot
    title : str
        Title for the plot
    file_prefix : str
        Prefix for the saved file
    csv_info : str, optional
        Identifier information from the CSV filename to append to titles and filenames
    """
    # Add CSV info to title if provided
    if csv_info:
        full_title = f"{title} - {csv_info}"
    else:
        full_title = title
        
    print(f"Creating interpolated grid for {data_column}...")
    
    # Check if required coordinate columns exist
    if 'UTM_Easting' not in df.columns or 'UTM_Northing' not in df.columns:
        print("Cannot create interpolated grid: Missing coordinate columns")
        return
    
    # Extract coordinates and values
    easting = df['UTM_Easting'].values
    northing = df['UTM_Northing'].values
    data_values = df[data_column].values
    
    # Define the grid for interpolation
    easting_min, easting_max = easting.min(), easting.max()
    northing_min, northing_max = northing.min(), northing.max()
    
    # Add a small buffer to the min/max values
    buffer = GRID_BUFFER_PERCENT
    easting_range = easting_max - easting_min
    northing_range = northing_max - northing_min
    easting_min -= easting_range * buffer
    easting_max += easting_range * buffer
    northing_min -= northing_range * buffer
    northing_max += northing_range * buffer
    
    # Create a regular grid for interpolation
    xi = np.linspace(easting_min, easting_max, GRID_RESOLUTION)
    yi = np.linspace(northing_min, northing_max, GRID_RESOLUTION)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    points = np.column_stack((easting, northing))
    
    # Interpolate the values onto the grid
    method = 'cubic' if USE_CUBIC_INTERPOLATION else 'linear'
    grid_data = griddata(points, data_values, (xi_grid, yi_grid), method=method)
    
    # Create figure for the standard plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create contour plot
    contour = ax.contourf(xi_grid, yi_grid, grid_data, levels=50, cmap=COLOR_MAP)
    
    # Add scatter points showing the actual data points
    ax.scatter(easting, northing, c=data_values, cmap=COLOR_MAP, s=5, alpha=0.5)
    
    # Add colorbar
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label('Value (nT)')
    
    # Add flight path visualization
    ax.plot(easting, northing, 'k-', linewidth=0.5, alpha=0.5)
    
    # Add markers for start and end points
    ax.plot(easting[0], northing[0], 'ko', markersize=6, label='Start')
    ax.plot(easting[-1], northing[-1], 'kx', markersize=6, label='End')
    
    # Set labels and title
    ax.set_xlabel('UTM Easting (m)')
    ax.set_ylabel('UTM Northing (m)')
    ax.set_title(full_title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Save the plot with CSV info in filename if provided
    if csv_info:
        output_path = os.path.join(output_dir, f"{file_prefix}_{csv_info}_interpolated_grid.png")
    else:
        output_path = os.path.join(output_dir, f"{file_prefix}_interpolated_grid.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Interpolated grid saved to {output_path}")
    
    # Create an interactive version with plotly if enabled
    if CREATE_INTERACTIVE_PLOTS:
        try:
            # Create directory for interactive plots
            interactive_dir = os.path.join(output_dir, 'interactive_plots')
            os.makedirs(interactive_dir, exist_ok=True)
            
            # For satellite background, we'll use Plotly's Mapbox functionality
            if USE_SATELLITE_BACKGROUND:
                # Convert UTM coordinates (easting/northing) to lat/lon for accurate mapping
                # Assuming Zone 33N for Norway - adjust as needed for your specific region
                utm_zone = 33
                hemisphere = 'N'  # 'N' for Northern Hemisphere
                
                # Derive data type name from column name for labeling
                data_type = data_column.replace(' [nT]', '').replace('_', ' ').title()
                
                # Convert grid coordinates to lat/lon for heatmap overlay
                grid_lats, grid_lons = [], []
                for i in range(len(yi)):
                    row_lats, row_lons = [], []
                    for j in range(len(xi)):
                        lat, lon = utm.to_latlon(xi[j], yi[i], utm_zone, hemisphere)
                        row_lats.append(lat)
                        row_lons.append(lon)
                    grid_lats.append(row_lats)
                    grid_lons.append(row_lons)
                
                # Convert flight path coordinates to lat/lon
                flight_lats, flight_lons = [], []
                for e, n in zip(easting, northing):
                    lat, lon = utm.to_latlon(e, n, utm_zone, hemisphere)
                    flight_lats.append(lat)
                    flight_lons.append(lon)
                
                # Calculate proper map bounds from flight path data
                lat_min, lat_max = min(flight_lats), max(flight_lats)
                lon_min, lon_max = min(flight_lons), max(flight_lons)
                lat_center = (lat_min + lat_max) / 2
                lon_center = (lon_min + lon_max) / 2
                
                # Add margin to bounds for better visualization  
                lat_range = lat_max - lat_min
                lon_range = lon_max - lon_min
                margin = max(lat_range, lon_range) * 0.1  # 10% margin for better view
                
                # Adjust bounds with margin
                lat_min_bounded = lat_min - margin
                lat_max_bounded = lat_max + margin
                lon_min_bounded = lon_min - margin
                lon_max_bounded = lon_max + margin
                
                # Create the satellite map figure
                fig = go.Figure()
                
                # Calculate fixed color scale limits to prevent color changes during zoom
                z_values = [val for row in grid_data for val in row if not np.isnan(val)]
                z_min, z_max = np.min(z_values), np.max(z_values)
                
                # Add the interpolated grid as a heatmap overlay
                # Convert grid data to the format needed for mapbox heatmap
                fig.add_trace(
                    go.Densitymap(
                        lat=[lat for sublist in grid_lats for lat in sublist],
                        lon=[lon for sublist in grid_lons for lon in sublist],
                        z=[val for row in grid_data for val in row],
                        radius=20,  # Adjust radius for smoother interpolation
                        colorscale='Viridis',
                        opacity=0.6,
                        showscale=True,
                        colorbar=dict(title=f'{data_type} [nT]', x=1.02),
                        name=f'{data_type} Grid',
                        # Fix color scale to prevent changes during zoom
                        zmin=z_min,
                        zmax=z_max
                    )
                )
                
                # Add the flight path as a line trace
                fig.add_trace(
                    go.Scattermap(
                        lat=flight_lats,
                        lon=flight_lons,
                        mode='lines+markers',
                        marker=dict(size=3, color='white', opacity=0.8),
                        line=dict(width=2, color='white'),
                        name='Flight Path',
                    )
                )
                
                # Add start and end points
                fig.add_trace(
                    go.Scattermap(
                        lat=[flight_lats[0]],
                        lon=[flight_lons[0]],
                        mode='markers',
                        marker=dict(size=12, color='lime', symbol='circle'),
                        name='Start',
                    )
                )
                
                fig.add_trace(
                    go.Scattermap(
                        lat=[flight_lats[-1]],
                        lon=[flight_lons[-1]],
                        mode='markers',
                        marker=dict(size=12, color='red', symbol='circle'),
                        name='End',
                    )
                )
                
                # Add known structures from KML data
                known_structures = [
                    {"name": "Container 1", "lon": 9.375475517759535, "lat": 63.79320543182994},
                    {"name": "Container 2", "lon": 9.375633219112412, "lat": 63.79294118305212},
                    {"name": "Container 3", "lon": 9.375393311321158, "lat": 63.79273484546234},
                    {"name": "Container 4", "lon": 9.374789622071768, "lat": 63.79262335481558},
                    {"name": "Container 5", "lon": 9.374244197652779, "lat": 63.792723225999},
                    {"name": "Container 6", "lon": 9.374080873510549, "lat": 63.79293700932513},
                    {"name": "Container 7", "lon": 9.374271188454612, "lat": 63.79323198305541},
                    {"name": "Structure 8", "lon": 9.374844174599604, "lat": 63.79297636698774},
                    {"name": "Bomblet", "lon": 9.3747775, "lat": 63.793296}
                ]
                
                # Extract coordinates for known structures
                structure_lats = [s["lat"] for s in known_structures]
                structure_lons = [s["lon"] for s in known_structures]
                structure_names = [s["name"] for s in known_structures]
                
                fig.add_trace(
                    go.Scattermap(
                        lat=structure_lats,
                        lon=structure_lons,
                        mode='markers',
                        marker=dict(
                            size=14, 
                            color='black', 
                            symbol='diamond', 
                            opacity=1.0,
                        ),
                        name='Known Structures',
                        text=structure_names,
                        hovertemplate='<b>%{text}</b><br>' +
                                    'Lat: %{lat:.6f}<br>' +
                                    'Lon: %{lon:.6f}<br>' +
                                    '<extra></extra>'
                    )
                )
                
                # Configure the layout for satellite imagery with automatic zoom to data bounds
                fig.update_layout(
                    title=full_title,
                    mapbox=dict(
                        style='satellite',  # Use true satellite imagery, not stylized map
                        center=dict(lat=lat_center, lon=lon_center),
                        bearing=0,
                        pitch=0,
                    ),
                    height=900,
                    width=1200,
                    margin=dict(l=0, r=0, t=50, b=0),
                    showlegend=True,
                    legend=dict(
                        x=0.01,
                        y=0.99,
                        bgcolor='rgba(255,255,255,0.8)',
                        bordercolor='black',
                        borderwidth=1
                    )
                )
                
                # Force the map to fit bounds after layout is configured
                fig.update_mapboxes(
                    bounds=dict(
                        west=lon_min_bounded,
                        east=lon_max_bounded,
                        south=lat_min_bounded,
                        north=lat_max_bounded
                    )
                )
            else:
                # Create the original interactive heatmap without satellite background
                fig = go.Figure()
                
                # Add heatmap trace for interpolated values
                fig.add_trace(
                    go.Heatmap(
                        x=xi,
                        y=yi,
                        z=grid_data,
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
                    title=full_title,
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
            if csv_info:
                output_path = os.path.join(interactive_dir, f"{file_prefix}_{csv_info}_interpolated_grid_interactive.html")
            else:
                output_path = os.path.join(interactive_dir, f"{file_prefix}_interpolated_grid_interactive.html")
            fig.write_html(output_path)
            print(f"Interactive interpolated grid saved to {output_path}")
            
        except Exception as e:
            print(f"Could not create interactive grid plot: {e}")


def filter_extreme_outliers(df, column, factor=3.0):
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


def apply_cutoffs(df, beginning_pct=0, end_pct=0, middle_cuts=None):
    """
    Apply cutoffs to the dataframe, trimming data from beginning, end, and middle sections.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data to be trimmed
    beginning_pct : float
        Percentage of data to cut from the beginning (0-100)
    end_pct : float
        Percentage of data to cut from the end (0-100)
    middle_cuts : list
        List of percentage points for middle cuts, where each pair represents
        the start and end of a section to cut out (e.g., [40, 60] cuts out 40%-60%)
    
    Returns:
    --------
    pandas.DataFrame
        Filtered dataframe with the specified sections removed
    """
    if df is None or df.empty:
        print("No data to apply cutoffs to")
        return df
    
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Validate percentages are within range
    beginning_pct = max(0, min(beginning_pct, 100))
    end_pct = max(0, min(end_pct, 100))
    
    # Convert percentages to row indices
    total_rows = len(result_df)
    
    # Create a mask that starts with all True (keep all rows)
    keep_mask = np.ones(total_rows, dtype=bool)
    
    # Apply beginning cut
    if beginning_pct > 0:
        cut_rows = int(total_rows * beginning_pct / 100)
        keep_mask[:cut_rows] = False
        print(f"Cut {cut_rows} rows ({beginning_pct:.2f}%) from beginning")
    
    # Apply end cut
    if end_pct > 0:
        cut_rows = int(total_rows * end_pct / 100)
        if cut_rows > 0:
            keep_mask[-cut_rows:] = False
            print(f"Cut {cut_rows} rows ({end_pct:.2f}%) from end")
    
    # Apply middle cuts
    if middle_cuts and len(middle_cuts) >= 2:
        for i in range(0, len(middle_cuts), 2):
            if i+1 < len(middle_cuts):
                start_pct = middle_cuts[i]
                end_pct = middle_cuts[i+1]
                
                if start_pct >= 0 and end_pct > start_pct and end_pct <= 100:
                    start_idx = int(total_rows * start_pct / 100)
                    end_idx = int(total_rows * end_pct / 100)
                    
                    if start_idx < end_idx:
                        keep_mask[start_idx:end_idx] = False
                        print(f"Cut rows {start_idx} to {end_idx} ({start_pct:.2f}%-{end_pct:.2f}%) from middle")
    
    # Apply the mask to filter the dataframe
    filtered_df = result_df[keep_mask].copy()
    
    # Reset the index to make it continuous again
    filtered_df.reset_index(drop=True, inplace=True)
    
    # Report total rows cut
    rows_cut = total_rows - len(filtered_df)
    if rows_cut > 0:
        print(f"Total: Cut {rows_cut} rows ({rows_cut/total_rows*100:.2f}% of data)")
    else:
        print("No data was cut")
    
    return filtered_df


def interactive_select_cutoffs(df, data_column):
    """
    Display an interactive plot to let the user select data cutoffs.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data to apply cutoffs to
    data_column : str
        Column name in df containing the data to visualize
        
    Returns:
    --------
    list
        [begin_pct, end_pct, middle_cuts] cutoff values
    """
    # No need to re-import these as they are already imported at the top level
    # Just use the existing imports
    
    # Downsample large datasets for better performance
    max_points = 5000  # Maximum number of points to display for smooth interaction
    if len(df) > max_points:
        step = max(1, len(df) // max_points)
        df_display = df.iloc[::step].copy()
        print(f"Dataset downsampled from {len(df)} to {len(df_display)} points for visualization")
    else:
        df_display = df.copy()
    
    # Turn on interactive mode for faster updates
    plt.ion()
    
    # Create figure and layout with higher DPI for better rendering
    fig = plt.figure(figsize=(14, 8), dpi=100)
    fig.subplots_adjust(left=0.1, bottom=0.25, right=0.95, top=0.95)
    
    # Main axis for path plotting
    ax_main = plt.axes([0.1, 0.35, 0.85, 0.55])
    
    # Axes for sliders and buttons
    ax_begin = plt.axes([0.25, 0.15, 0.65, 0.03])
    ax_end = plt.axes([0.25, 0.1, 0.65, 0.03])
    ax_add_marker = plt.axes([0.1, 0.02, 0.2, 0.04])
    ax_reset_markers = plt.axes([0.35, 0.02, 0.2, 0.04])
    ax_apply = plt.axes([0.6, 0.02, 0.2, 0.04])
    
    # Set up the main plot
    if 'UTM_Easting' in df_display.columns and 'UTM_Northing' in df_display.columns:
        # Use UTM coords if available for spatial visualization
        easting = df_display['UTM_Easting'].values
        northing = df_display['UTM_Northing'].values
        x = easting
        y = northing
        ax_main.set_xlabel('UTM Easting (m)')
        ax_main.set_ylabel('UTM Northing (m)')
        ax_main.set_title(f'Select Cutoff Regions - {data_column} (Spatial View)')
    else:
        # Otherwise use sample index and values
        x = np.arange(len(df_display))
        y = df_display[data_column].values if data_column in df_display.columns else np.zeros(len(df_display))
        ax_main.set_xlabel('Sample Index')
        ax_main.set_ylabel(data_column if data_column in df_display.columns else 'Value')
        ax_main.set_title(f'Select Cutoff Regions - {data_column} (Time Series View)')
    
    # Plot the main data with optimized rendering
    colorVal = df_display[data_column].values if data_column in df_display.columns else np.zeros(len(df_display))
    
    # Use line plot instead of scatter for better performance
    path_all, = ax_main.plot(x, y, 'k-', lw=0.7, alpha=0.3, zorder=1, label='All Data')
    
    # Add color information using a small number of scatter points for reference
    scatter_step = max(1, len(x) // 500)  # Limit to 500 scatter points for performance
    scatter = ax_main.scatter(x[::scatter_step], y[::scatter_step], 
                             c=colorVal[::scatter_step], cmap='viridis', 
                             s=10, alpha=0.7, zorder=2)
    plt.colorbar(scatter, ax=ax_main, label=data_column)
    
    # Create excluded segments (these will be updated by sliders)
    path_excluded_start, = ax_main.plot([], [], 'r-', lw=2.5, alpha=0.8, zorder=3, label='Excluded Start')
    path_excluded_end, = ax_main.plot([], [], 'b-', lw=2.5, alpha=0.8, zorder=3, label='Excluded End')
    path_included, = ax_main.plot([], [], 'g-', lw=2, alpha=0.5, zorder=2, label='Included Data')
    
    # Add markers for start and end points
    ax_main.plot(x[0], y[0], 'go', markersize=10, label='Start')
    ax_main.plot(x[-1], y[-1], 'bo', markersize=10, label='End')

    # Legend
    ax_main.legend(loc='upper right', fontsize=9)
    
    # Middle cut markers
    middle_markers = []
    
    # Sliders
    beginning_pct_slider = Slider(ax_begin, 'Beginning Cutoff %', 0, 30, valinit=0, valstep=0.5)
    end_pct_slider = Slider(ax_end, 'End Cutoff %', 0, 30, valinit=0, valstep=0.5)
    
    # Buttons
    add_marker_button = Button(ax_add_marker, 'Add Marker', color='0.85', hovercolor='0.95')
    reset_markers_button = Button(ax_reset_markers, 'Reset Markers', color='0.85', hovercolor='0.95')
    apply_button = Button(ax_apply, 'Apply Cutoffs', color='0.85', hovercolor='0.95')
    
    # Current markers
    markers_dict = {'markers': []}
    adding_marker = [False]
    cutoffs_accepted = [False]
    
    # Progress message area
    progress_text = plt.figtext(0.5, 0.02, 'Click Add Marker to add middle cutoff points', ha='center', va='bottom', fontsize=10)
    
    # Create marker lines for middle cuts for better performance
    middle_cut_lines = []
    
    # Update function for sliders - optimized for speed
    def update_slider(val=None):
        begin_pct = beginning_pct_slider.val
        end_pct = end_pct_slider.val
        n = len(x)
        begin_idx = min(int(begin_pct * n / 100), n-1)
        end_idx = max(n - 1 - int(end_pct * n / 100), 0)
        
        # Only update what's needed - don't redraw the entire figure
        if begin_idx > 0:
            path_excluded_start.set_data(x[:begin_idx], y[:begin_idx])
        else:
            path_excluded_start.set_data([], [])
            
        if end_idx < n - 1:
            path_excluded_end.set_data(x[end_idx:], y[end_idx:])
        else:
            path_excluded_end.set_data([], [])
            
        if begin_idx <= end_idx:
            path_included.set_data(x[begin_idx:end_idx+1], y[begin_idx:end_idx+1])
        else:
            path_included.set_data([], [])
        
        # Use blit for faster rendering
        path_excluded_start.figure.canvas.draw_idle()
    
    # Connect sliders to update function with throttling to reduce redraws
    def throttled_update(val):
        # Use a simple timer to limit updates
        if not hasattr(throttled_update, 'last_update'):
            throttled_update.last_update = 0
        
        now = time.time()
        # Only update if 100ms has passed since last update
        if now - throttled_update.last_update > 0.1:
            update_slider(val)
            throttled_update.last_update = now
    
    # Connect sliders to throttled update function
    beginning_pct_slider.on_changed(throttled_update)
    end_pct_slider.on_changed(throttled_update)
    
    # Add marker function
    def add_marker(event):
        if event.inaxes != ax_main:
            return
            
        # Toggle marker adding mode
        adding_marker[0] = not adding_marker[0]
        
        if adding_marker[0]:
            add_marker_button.color = '0.65'
            progress_text.set_text('Click on the plot to add marker')
        else:
            add_marker_button.color = '0.85'
            progress_text.set_text('Click Add Marker to add middle cutoff points')
        
        add_marker_button.figure.canvas.draw_idle()
    
    # Reset markers function
    def reset_markers(event):
        for marker in markers_dict['markers']:
            marker.remove()
        markers_dict['markers'] = []
        middle_markers.clear()
        
        # Clear any middle cut lines
        for line in middle_cut_lines:
            line.remove()
        middle_cut_lines.clear()
        
        progress_text.set_text('Markers reset. Click Add Marker to add middle cutoff points')
        fig.canvas.draw_idle()
    
    # Apply cutoffs function for when the button is clicked
    def apply_cutoffs(event):
        cutoffs_accepted[0] = True
        plt.close(fig)
    
    # Handle click events for adding markers
    def on_click(event):
        if not adding_marker[0] or event.inaxes != ax_main:
            return
            
        # Find closest data point with faster nearest-neighbor search
        # Using a simplified approach for large datasets
        if len(x) > 1000:
            # Get array indices within visible area for faster processing
            xlim = ax_main.get_xlim()
            ylim = ax_main.get_ylim()
            visible_indices = np.where(
                (x >= xlim[0]) & (x <= xlim[1]) &
                (y >= ylim[0]) & (y <= ylim[1])
            )[0]
            
            if len(visible_indices) > 0:
                # Find closest point among visible points
                visible_x = x[visible_indices]
                visible_y = y[visible_indices]
                distances = np.sqrt((visible_x - event.xdata)**2 + (visible_y - event.ydata)**2)
                closest_idx = visible_indices[np.argmin(distances)]
            else:
                # Fallback if no visible points
                distances = np.sqrt((x - event.xdata)**2 + (y - event.ydata)**2)
                closest_idx = np.argmin(distances)
        else:
            # For smaller datasets, just find closest point
            distances = np.sqrt((x - event.xdata)**2 + (y - event.ydata)**2)
            closest_idx = np.argmin(distances)
        
        marker_x, marker_y = x[closest_idx], y[closest_idx]
        
        # Add marker
        marker, = ax_main.plot(marker_x, marker_y, 'rx', markersize=10)
        markers_dict['markers'].append(marker)
        middle_markers.append(closest_idx)
        
        # If we have a pair, draw a line to indicate the cut region
        if len(middle_markers) % 2 == 0:
            idx1 = middle_markers[-2]
            idx2 = middle_markers[-1]
            # Sort indices
            if idx1 > idx2:
                idx1, idx2 = idx2, idx1
            # Draw line to indicate cut region
            cut_line, = ax_main.plot(x[idx1:idx2+1], y[idx1:idx2+1], 'm-', linewidth=2, alpha=0.6)
            middle_cut_lines.append(cut_line)
            progress_text.set_text(f'{len(middle_markers)//2} marker pairs added. Add more or Apply Cutoffs')
        else:
            progress_text.set_text(f'Add another marker to complete pair {len(middle_markers)//2+1}')
            
        # Exit marker adding mode after adding
        adding_marker[0] = False
        add_marker_button.color = '0.85'
        
        # Only redraw what's necessary
        ax_main.figure.canvas.draw_idle()
    
    # Connect buttons
    add_marker_button.on_clicked(add_marker)
    reset_markers_button.on_clicked(reset_markers)
    apply_button.on_clicked(apply_cutoffs)
    
    # Connect click event
    fig.canvas.mpl_connect('button_press_event', on_click)
    
    # Initialize the plot
    update_slider()
    
    # Ensure tight layout optimization is off for better performance
    fig.set_tight_layout(False)
    
    # Show the plot and wait for user to apply cutoffs - must use blocking mode
    try:
        plt.show(block=True)  # Using block=True to ensure the dialog waits for user input
    except KeyboardInterrupt:
        print('Interactive cutoff selection interrupted')
        plt.close(fig)
        return [0, 0, []]
    
    # Return the selected cutoffs
    if cutoffs_accepted[0]:
        # Return cutoff values based on original dataframe percentages
        begin_pct = beginning_pct_slider.val
        end_pct = end_pct_slider.val
        
        # Convert the indices from downsampled data back to original indices for middle cuts
        middle_pairs = []
        for i in range(0, len(middle_markers), 2):
            if i+1 < len(middle_markers):
                # Convert indices to original data percentages
                idx1_pct = 100 * middle_markers[i] / len(x)
                idx2_pct = 100 * middle_markers[i+1] / len(x)
                middle_pairs.append([idx1_pct, idx2_pct])
                
        return [begin_pct, end_pct, middle_pairs]
    else:
        # User closed without applying, return default values
        return [0, 0, []]


def extract_csv_info(csv_path):
    """
    Extract meaningful identifiers from the CSV filename to use in plot titles and output filenames.
    
    Parameters:
    -----------
    csv_path : str
        Path to the CSV file
        
    Returns:
    --------
    str
        Identifier string extracted from the CSV filename
    """
    # Get the filename without path and extension
    filename = os.path.basename(csv_path)
    name_without_ext = os.path.splitext(filename)[0]
    
    # Try to extract identifiers from filenames that follow patterns like:
    # 20250605_082343_MWALK_#0122_processed_20250605_133207.csv
    parts = name_without_ext.split('_')
    identifiers = []
    
    # Extract date, time, processing info if available
    for part in parts:
        if part.isdigit() and (len(part) == 8 or len(part) == 6):
            identifiers.append(part)
    
    if not identifiers:
        # If no identifiers found, use filename as fallback
        return name_without_ext
    else:
        return '_'.join(identifiers)


def main():
    """
    Main function to run the visualization process.
    """
    import argparse
    parser = argparse.ArgumentParser(description='Visualize magnetic data from CSV files.')
    parser.add_argument('--interactive-cutoffs', action='store_true', help='Use interactive cutoff selection')
    args = parser.parse_args()
    
    use_interactive_cutoffs = args.interactive_cutoffs
    
    # Set up output directory
    if not OUTPUT_DIR:
        # Create output directory based on input file location
        output_dir = os.path.join(os.path.dirname(os.path.abspath(CSV_PATH)), 'visualizations')
    else:
        output_dir = OUTPUT_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output will be saved to: {output_dir}")
    
    # Read the CSV data
    df = read_magnetic_data(CSV_PATH)
    if df is None or df.empty:
        print(f"Error: Could not read data from {CSV_PATH}")
        return
        
    print(f"Loaded {len(df)} data points from {CSV_PATH}")
    
    # Extract identifiers from filename for labeling
    csv_info = extract_csv_info(CSV_PATH)
    
    # Interactive cutoff selection if enabled
    if use_interactive_cutoffs:
        print("Opening interactive cutoff selection window...")
        print("Instructions:")
        print("1. Use sliders to set beginning and end cutoff percentages")
        print("2. Click 'Add Marker' button, then click on the plot to add middle cut points")
        print("3. Add pairs of markers to define middle sections to cut")
        print("4. Click 'Reset Markers' to clear all middle markers")
        print("5. Click 'Apply Cutoffs' to save your selections")
        
        # Default to R1 column for visualization if available, otherwise use first available column
        viz_column = 'R1' if 'R1' in df.columns else df.columns[min(3, len(df.columns)-1)]
        cutoff_values = interactive_select_cutoffs(df, viz_column)
        
        # Unpack cutoff values
        beginning_pct, end_pct, middle_cuts = cutoff_values
        print(f"Selected cutoffs: Beginning {beginning_pct}%, End {end_pct}%, Middle points: {middle_cuts}")
        
        # Apply selected cutoffs to the dataframe
        df = apply_cutoffs(df, beginning_pct, end_pct, middle_cuts)
        print(f"Applied cutoffs - {len(df)} data points remaining")
    else:
        # Apply default cutoffs from configuration
        if CUTOFFS[0] > 0 or CUTOFFS[1] > 0 or (CUTOFFS[2] and len(CUTOFFS[2]) > 0):
            print(f"Applying default cutoffs: Beginning {CUTOFFS[0]}%, End {CUTOFFS[1]}%, Middle points: {CUTOFFS[2]}")
            df = apply_cutoffs(df, CUTOFFS[0], CUTOFFS[1], CUTOFFS[2])
            print(f"Applied cutoffs - {len(df)} data points remaining")
        else:
            print("No cutoffs applied - using all data points")
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output will be saved to: {output_dir}")
    
    # Create interactive plots directory
    if CREATE_INTERACTIVE_PLOTS:
        interactive_dir = os.path.join(output_dir, 'interactive_plots')
        os.makedirs(interactive_dir, exist_ok=True)
    
    # Apply cutoffs if configured
    print("\nChecking cutoff configuration...")
    if CUTOFFS and (CUTOFFS[0] > 0 or CUTOFFS[1] > 0 or (len(CUTOFFS) > 2 and len(CUTOFFS[2]) > 0)):
        print(f"Applying configured cutoffs: {CUTOFFS}")
        # Extract cutoffs from configuration
        cut_beginning_pct = CUTOFFS[0]  # Beginning cutoff percentage
        cut_end_pct = CUTOFFS[1]       # End cutoff percentage
        middle_cuts = CUTOFFS[2] if len(CUTOFFS) > 2 else []  # Middle cut points
        
        # Create a cutoffs directory for output with cutoffs applied
        cutoffs_dir = os.path.join(output_dir, 'cutoffs_applied')
        os.makedirs(cutoffs_dir, exist_ok=True)
        
        # Apply the cutoffs to the dataframe
        print("\nApplying data cutoffs:")
        df_cut = apply_cutoffs(df, cut_beginning_pct, cut_end_pct, middle_cuts)
        
        # Make both the original and cutoff-applied dataframes available
        df_original = df.copy()
        # Transfer the available_data attributes
        df_cut.attrs['available_data'] = df.attrs.get('available_data', {})
    else:
        print("No cutoffs configured. Using all data.")
        df_cut = df
        df_original = df
        cutoffs_dir = output_dir  # No separate directory needed
        
    # Get available data types from the dataframe attributes
    available_data = df.attrs.get('available_data', {})
    
    # Process vertical alignment data if available
    if 'va' in available_data and PLOT_TIME_SERIES:
        va_column = available_data['va']
        print(f"\nProcessing Vertical Anomaly data from column: {va_column}")
        
        # Plot VA over time with original data
        plot_data_over_time(df, output_dir, va_column, 
                          "Vertical Anomaly Over Time", 
                          "Vertical Anomaly (nT)", 
                          "vertical_anomaly",
                          csv_info)
        
        # Also plot with cutoffs if they were applied and resulted in changes
        if 'df_cut' in locals() and len(df_cut) < len(df):
            plot_data_over_time(df_cut, cutoffs_dir, va_column,
                              "Vertical Anomaly Over Time (with cutoffs)", 
                              "Vertical Anomaly (nT)",
                              "vertical_anomaly_cut", 
                              csv_info)
        
        # Create interpolated grid if spatial coordinates are available
        if PLOT_INTERPOLATED_GRID and 'UTM_Easting' in df.columns and 'UTM_Northing' in df.columns:
            # First with all data points
            create_interpolated_grid(df, output_dir, va_column,
                                   "Vertical Anomaly (nT) Interpolated Grid",
                                   "vertical_anomaly",
                                   csv_info)
            
            # Also with cutoffs if they were applied
            if 'df_cut' in locals() and len(df_cut) < len(df):
                create_interpolated_grid(df_cut, cutoffs_dir, va_column,
                                       "Vertical Anomaly (nT) Interpolated Grid (with cutoffs)",
                                       "vertical_anomaly_cut",
                                       csv_info)
            
            # Then with outliers filtered if enabled
            if FILTER_OUTLIERS:
                print("\nFiltering outliers for cleaner visualization...")
                filtered_df = filter_extreme_outliers(df, va_column, factor=OUTLIER_FACTOR)
                
                if len(filtered_df) < len(df):
                    # Save filtered version with "_filtered" suffix in output directory
                    filtered_output_dir = os.path.join(output_dir, 'filtered')
                    os.makedirs(filtered_output_dir, exist_ok=True)
                    
                    create_interpolated_grid(filtered_df, filtered_output_dir, va_column,
                                          "Filtered Vertical Anomaly Interpolated Grid",
                                          "vertical_anomaly_filtered",
                                          csv_info)
                    
                    # Also with cutoffs if they were applied
                    if 'df_cut' in locals() and len(df_cut) < len(df):
                        filtered_cutoffs_dir = os.path.join(cutoffs_dir, 'filtered')
                        os.makedirs(filtered_cutoffs_dir, exist_ok=True)
                        
                        filtered_df_cut = filter_extreme_outliers(df_cut, va_column, factor=OUTLIER_FACTOR)
                        create_interpolated_grid(filtered_df_cut, filtered_cutoffs_dir, va_column,
                                              "Filtered Vertical Anomaly Interpolated Grid (with cutoffs)",
                                              "vertical_anomaly_filtered_cut",
                                              csv_info)
    
    # Process residual data if available
    residual_keys = [key for key in available_data if key.startswith('residual_')]
    for key in residual_keys:
        if PLOT_TIME_SERIES:
            res_column = available_data[key]
            print(f"\nProcessing Residual data from column: {res_column}")
            
            # Get sensor number from column name (R1, R2, etc.)
            sensor_num = res_column.split()[0].replace('R', '')
            title = f"Residual {sensor_num} Field Over Time"
            y_label = f"Residual {sensor_num} (nT)"
            file_prefix = f"residual{sensor_num}"
            
            # Plot residual over time with original data
            plot_data_over_time(df, output_dir, res_column, title, y_label, file_prefix, csv_info)
            
            # Also plot with cutoffs if they were applied
            if 'df_cut' in locals() and len(df_cut) < len(df):
                plot_data_over_time(df_cut, cutoffs_dir, res_column,
                                  f"Residual {sensor_num} Field Over Time (with cutoffs)", 
                                  y_label, 
                                  f"{file_prefix}_cut", 
                                  csv_info)
            
            # Create interpolated grid if spatial coordinates are available
            if PLOT_INTERPOLATED_GRID and 'UTM_Easting' in df.columns and 'UTM_Northing' in df.columns:
                # First with all data points
                create_interpolated_grid(df, output_dir, res_column,
                                      f"Residual {sensor_num} Interpolated Grid",
                                      file_prefix,
                                      csv_info)
                
                # Also with cutoffs if they were applied
                if 'df_cut' in locals() and len(df_cut) < len(df):
                    create_interpolated_grid(df_cut, cutoffs_dir, res_column,
                                          f"Residual {sensor_num} Interpolated Grid (with cutoffs)",
                                          f"{file_prefix}_cut",
                                          csv_info)
                
                # Then with outliers filtered if enabled
                if FILTER_OUTLIERS:
                    print("\nFiltering outliers for cleaner visualization...")
                    filtered_df = filter_extreme_outliers(df, res_column, factor=OUTLIER_FACTOR)
                    
                    if len(filtered_df) < len(df):
                        # Save filtered version with "_filtered" suffix in output directory
                        filtered_output_dir = os.path.join(output_dir, 'filtered')
                        os.makedirs(filtered_output_dir, exist_ok=True)
                        
                        create_interpolated_grid(filtered_df, filtered_output_dir, res_column,
                                              f"Filtered Residual {sensor_num} Interpolated Grid",
                                              f"{file_prefix}_filtered",
                                              csv_info)
                        
                        # Also with cutoffs if they were applied
                        if 'df_cut' in locals() and len(df_cut) < len(df):
                            filtered_cutoffs_dir = os.path.join(cutoffs_dir, 'filtered')
                            os.makedirs(filtered_cutoffs_dir, exist_ok=True)
                            
                            filtered_df_cut = filter_extreme_outliers(df_cut, res_column, factor=OUTLIER_FACTOR)
                            create_interpolated_grid(filtered_df_cut, filtered_cutoffs_dir, res_column,
                                                  f"Filtered Residual {sensor_num} Interpolated Grid (with cutoffs)",
                                                  f"{file_prefix}_filtered_cut",
                                                  csv_info)
    
    print("\nAll visualizations complete!")


if __name__ == "__main__":
    main()
