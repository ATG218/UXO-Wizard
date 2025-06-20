#!/usr/bin/env python3

import os
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy.interpolate import griddata
from matplotlib.colors import Normalize
from matplotlib.patches import Patch

# Configuration variables - modify these as needed
CSV_PATH = '/Users/aleksandergarbuz/Documents/SINTEF/Magnetic_Data/FIELD_DATA_200525/processing_results/20250520_072745_MWALK_#0122_processed_20250603_134940.csv'

# Plot flags - set to True to enable the corresponding plot
PLOT_UTM_PATHS = False  # Plot the three UTM coordinate pairs
PLOT_UTM_TIME = False  # Plot UTM paths with time progression
PLOT_MAG_FIELD = False  # Plot total magnetic field over time
PLOT_RESIDUALS = False  # Plot residual magnetic field values
PLOT_UTM_COMPONENTS = False  # Plot UTM Easting and Northing values over time
PLOT_INTERACTIVE_UTM = False  # Interactive plot of UTM paths
PLOT_INTERPOLATION = True  # Plot of interpolated residual field data
PLOT_FILTER_MASKS = True  # Plot showing which segments were excluded by which filtering criteria

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
        print("Columns in the file:")
        for i, col in enumerate(df.columns):
            print(f"  {i}: {col}")
        
        # Show the first few rows to understand the data structure
        print("\nFirst few rows of data:")
        print(df.head())
        
        return df
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None


def plot_utm_paths(df, output_dir, enable=True):
    """Plot actual UTM paths showing the real positions of the different coordinate systems"""
    if not enable:
        return
    
    plt.figure(figsize=(12, 10))
    
    # Plot the three UTM paths with their actual coordinates
    plt.plot(df['UTM_Easting'], df['UTM_Northing'], 'b-', label='UTM Original', linewidth=1.5)
    plt.plot(df['UTM_Easting1'], df['UTM_Northing1'], 'r-', label='UTM 1', linewidth=1.5)
    plt.plot(df['UTM_Easting2'], df['UTM_Northing2'], 'g-', label='UTM 2', linewidth=1.5)
    
    # Mark the starting points
    plt.plot(df['UTM_Easting'].iloc[0], df['UTM_Northing'].iloc[0], 'bo', markersize=8, label='Start Original')
    plt.plot(df['UTM_Easting1'].iloc[0], df['UTM_Northing1'].iloc[0], 'ro', markersize=8, label='Start UTM 1')
    plt.plot(df['UTM_Easting2'].iloc[0], df['UTM_Northing2'].iloc[0], 'go', markersize=8, label='Start UTM 2')
    
    plt.title('UTM Coordinate Paths')
    plt.xlabel('UTM Easting (m)')
    plt.ylabel('UTM Northing (m)')
    plt.grid(True)
    plt.legend()
    
    # Set equal aspect to maintain the correct scale
    plt.axis('equal')
    
    # Save the figure
    output_path = os.path.join(output_dir, 'utm_paths.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"UTM paths plot saved to {output_path}")


def plot_utm_paths_over_time(df, output_dir, enable=True):
    """Create an animation of the UTM paths over time"""
    if not enable:
        return
    
    import matplotlib.animation as animation
    from matplotlib.animation import FuncAnimation
    
    # Create a new figure for the animation
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Set up the plot
    ax.set_title('UTM Coordinate Paths Over Time')
    ax.set_xlabel('UTM Easting (m)')
    ax.set_ylabel('UTM Northing (m)')
    ax.grid(True)
    ax.axis('equal')  # Equal aspect ratio
    
    # Mark the starting points
    ax.plot(df['UTM_Easting'].iloc[0], df['UTM_Northing'].iloc[0], 'bo', markersize=8, label='Start UTM Original')
    ax.plot(df['UTM_Easting1'].iloc[0], df['UTM_Northing1'].iloc[0], 'ro', markersize=8, label='Start UTM 1')
    ax.plot(df['UTM_Easting2'].iloc[0], df['UTM_Northing2'].iloc[0], 'go', markersize=8, label='Start UTM 2')
    
    # Prepare data - use subset of data for smoother animation
    # Take every Nth point (adjust this based on your data size)
    step = max(1, len(df) // 1000)  # Aim for ~1000 frames
    df_subset = df.iloc[::step].copy()
    
    # Set up line objects
    line1, = ax.plot([], [], 'b-', label='UTM Original', linewidth=1.5)
    line2, = ax.plot([], [], 'r-', label='UTM 1', linewidth=1.5)
    line3, = ax.plot([], [], 'g-', label='UTM 2', linewidth=1.5)
    
    # Add a time indicator text
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    
    # Get the data ranges for setting axis limits
    min_easting = min(
        df['UTM_Easting'].min(),
        df['UTM_Easting1'].min(),
        df['UTM_Easting2'].min()
    )
    max_easting = max(
        df['UTM_Easting'].max(),
        df['UTM_Easting1'].max(),
        df['UTM_Easting2'].max()
    )
    min_northing = min(
        df['UTM_Northing'].min(),
        df['UTM_Northing1'].min(),
        df['UTM_Northing2'].min()
    )
    max_northing = max(
        df['UTM_Northing'].max(),
        df['UTM_Northing1'].max(),
        df['UTM_Northing2'].max()
    )
    
    # Add some padding
    easting_range = max_easting - min_easting
    northing_range = max_northing - min_northing
    min_easting -= easting_range * 0.05
    max_easting += easting_range * 0.05
    min_northing -= northing_range * 0.05
    max_northing += northing_range * 0.05
    
    ax.set_xlim(min_easting, max_easting)
    ax.set_ylim(min_northing, max_northing)
    ax.legend()
    
    # Initialize points to draw
    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        line3.set_data([], [])
        time_text.set_text('')
        return line1, line2, line3, time_text
    
    # Animation function
    def animate(i):
        # Get data up to frame i
        data = df_subset.iloc[:i+1]
        
        # Update the lines using the actual UTM coordinates
        line1.set_data(data['UTM_Easting'], data['UTM_Northing'])
        line2.set_data(data['UTM_Easting1'], data['UTM_Northing1'])
        line3.set_data(data['UTM_Easting2'], data['UTM_Northing2'])
        
        # Update the time indicator
        if i < len(df_subset):
            if 'GPSTime [hh:mm:ss.sss]' in df_subset.columns:
                time_text.set_text(f'Time: {df_subset.iloc[i]["GPSTime [hh:mm:ss.sss]"]}')
            else:
                time_text.set_text(f'Frame: {i}/{len(df_subset)-1}')
        
        return line1, line2, line3, time_text
    
    # Create the animation
    anim = FuncAnimation(fig, animate, frames=len(df_subset),
                         init_func=init, blit=True, interval=50)
    
    # Save the animation as MP4
    output_path = os.path.join(output_dir, 'utm_paths_animation.mp4')
    
    # Set up the writer
    writer = animation.FFMpegWriter(
        fps=30, metadata=dict(artist='MagBase Post-Processing'), bitrate=1800
    )
    
    try:
        anim.save(output_path, writer=writer)
        print(f"UTM paths animation saved to {output_path}")
    except Exception as e:
        print(f"Error saving animation: {e}")
        # Fallback to saving a series of snapshots
        print("Falling back to saving snapshots of the animation...")
        
        # Create a directory for the snapshots
        frames_dir = os.path.join(output_dir, 'utm_paths_frames')
        os.makedirs(frames_dir, exist_ok=True)
        
        # Save a series of frames
        num_frames = min(50, len(df_subset))  # Limit to 50 frames
        frame_indices = [int(i * len(df_subset) / num_frames) for i in range(num_frames)]
        
        for frame_num, idx in enumerate(frame_indices):
            # Create a new figure for each frame
            frame_fig, frame_ax = plt.subplots(figsize=(12, 10))
            frame_data = df_subset.iloc[:idx+1]
            
            # Plot the three paths
            frame_ax.plot(frame_data['delta_easting'], frame_data['delta_northing'], 'b-', label='UTM Original')
            frame_ax.plot(frame_data['delta_easting1'], frame_data['delta_northing1'], 'r-', label='UTM 1')
            frame_ax.plot(frame_data['delta_easting2'], frame_data['delta_northing2'], 'g-', label='UTM 2')
            
            # Mark the starting point
            frame_ax.plot(0, 0, 'ko', markersize=8, label='Starting Point')
            
            # Set up the plot
            frame_ax.set_title(f'Delta Position - Frame {frame_num+1}/{num_frames}')
            frame_ax.set_xlabel('Delta Easting (m)')
            frame_ax.set_ylabel('Delta Northing (m)')
            frame_ax.grid(True)
            frame_ax.axis('equal')
            frame_ax.set_xlim(min_easting, max_easting)
            frame_ax.set_ylim(min_northing, max_northing)
            frame_ax.legend()
            
            # Save the frame
            frame_path = os.path.join(frames_dir, f'frame_{frame_num:03d}.png')
            frame_fig.savefig(frame_path, dpi=150, bbox_inches='tight')
            plt.close(frame_fig)
        
        print(f"Saved {num_frames} animation frames to {frames_dir}")
    
    plt.close(fig)


def plot_magnetic_field_over_time(df, output_dir, enable=True):
    """Plot total magnetic field over time"""
    if not enable:
        return
    
    plt.figure(figsize=(12, 6))
    
    # Create an index for the x-axis that represents samples rather than raw milliseconds
    x_index = range(len(df))
    
    plt.plot(x_index, df['Btotal1 [nT]'], 'b-', label='Total Field 1', linewidth=1.5)
    plt.plot(x_index, df['Btotal2 [nT]'], 'r-', label='Total Field 2', linewidth=1.5)
    
    plt.title('Total Magnetic Field Over Time')
    
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
    
    plt.ylabel('Magnetic Field (nT)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()  # Adjust layout to fit labels
    
    output_path = os.path.join(output_dir, 'magnetic_field_over_time.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Magnetic field plot saved to {output_path}")


def plot_residuals(df, output_dir, enable=True):
    """Plot residual magnetic field values"""
    if not enable:
        return
    
    plt.figure(figsize=(12, 6))
    
    # Create an index for the x-axis that represents samples rather than raw milliseconds
    x_index = range(len(df))
    
    plt.plot(x_index, df['R1 [nT]'], 'b-', label='Residual 1', linewidth=1.5)
    plt.plot(x_index, df['R2 [nT]'], 'r-', label='Residual 2', linewidth=1.5)
    
    plt.title('Magnetic Field Residuals Over Time')
    
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
    
    plt.ylabel('Residual (nT)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()  # Adjust layout to fit labels
    
    output_path = os.path.join(output_dir, 'residuals_over_time.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Residuals plot saved to {output_path}")


def plot_utm_components_over_time(df, output_dir, enable=True):
    """Plot UTM Easting and Northing coordinates separately over time"""
    if not enable:
        return
    
    # Plot UTM Easting values over time
    plt.figure(figsize=(12, 6))
    
    # Create an index for the x-axis that represents samples rather than raw milliseconds
    x_index = range(len(df))
    
    # Plot actual Easting values
    plt.plot(x_index, df['UTM_Easting'], 'b-', label='UTM Easting Original', linewidth=1.5)
    plt.plot(x_index, df['UTM_Easting1'], 'r-', label='UTM Easting 1', linewidth=1.5)
    plt.plot(x_index, df['UTM_Easting2'], 'g-', label='UTM Easting 2', linewidth=1.5)
    
    # Get appropriate y-axis limits to zoom in on the actual changes
    # Find percentiles instead of min/max to avoid outliers stretching the scale
    lower_bound = min(
        df['UTM_Easting'].min(), 
        df['UTM_Easting1'].min(), 
        df['UTM_Easting2'].min()
    )
    upper_bound = max(
        df['UTM_Easting'].max(), 
        df['UTM_Easting1'].max(), 
        df['UTM_Easting2'].max()
    )
    
    # Add some padding to the bounds
    range_size = upper_bound - lower_bound
    lower_bound -= range_size * 0.1
    upper_bound += range_size * 0.1
    
    # Set y-axis limits to focus on the actual changes
    plt.ylim(lower_bound, upper_bound)
    
    plt.title('UTM Easting Over Time')
    
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
    
    plt.ylabel('UTM Easting (m)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()  # Adjust layout to fit labels
    
    output_path = os.path.join(output_dir, 'utm_easting_over_time.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"UTM Easting plot saved to {output_path}")
    
    # Plot UTM Northing values over time
    plt.figure(figsize=(12, 6))
    
    # Plot actual Northing values
    plt.plot(x_index, df['UTM_Northing'], 'b-', label='UTM Northing Original', linewidth=1.5)
    plt.plot(x_index, df['UTM_Northing1'], 'r-', label='UTM Northing 1', linewidth=1.5)
    plt.plot(x_index, df['UTM_Northing2'], 'g-', label='UTM Northing 2', linewidth=1.5)
    
    # Get appropriate y-axis limits to zoom in on the actual changes
    # Find percentiles instead of min/max to avoid outliers stretching the scale
    lower_bound = min(
        df['UTM_Northing'].min(),
        df['UTM_Northing1'].min(),
        df['UTM_Northing2'].min()
    )
    upper_bound = max(
        df['UTM_Northing'].max(),
        df['UTM_Northing1'].max(),
        df['UTM_Northing2'].max()
    )
    
    # Add some padding to the bounds
    range_size = upper_bound - lower_bound
    lower_bound -= range_size * 0.1
    upper_bound += range_size * 0.1
    
    # Set y-axis limits to focus on the actual changes
    plt.ylim(lower_bound, upper_bound)
    
    plt.title('UTM Northing Over Time')
    
    # Format x-axis similar to the Easting plot
    if 'GPSTime [hh:mm:ss.sss]' in df.columns:
        plt.xticks(tick_positions, tick_labels, rotation=45)
        plt.xlabel('GPS Time')
    else:
        plt.xticks(tick_positions)
        plt.xlabel('Sample Number')
    
    plt.ylabel('UTM Northing (m)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'utm_northing_over_time.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"UTM Northing plot saved to {output_path}")


def visualize_filtering_masks(df, altitude_change_mask, unstable_altitude_mask, hover_mask, slow_erratic_mask, good_flight_mask, output_dir, thresholds=None):
    """Visualize which regions of the flight path were excluded by which filtering criteria"""
    if not PLOT_FILTER_MASKS:
        return
    
    # Create a directory for filter visualization plots
    filter_dir = os.path.join(output_dir, 'filter_analysis')
    os.makedirs(filter_dir, exist_ok=True)
    
    # Convert time to seconds from start for x-axis
    df['time_seconds'] = (df['GPSTime_dt'] - df['GPSTime_dt'].iloc[0]).dt.total_seconds()
    
    # Create figure with multiple subplots
    fig, axs = plt.subplots(5, 1, figsize=(15, 20), sharex=True)
    
    # Plot 1: Altitude with excluded regions highlighted
    ax = axs[0]
    # Plot altitude for all points
    ax.plot(df['time_seconds'], df['Altitude [m]'], 'b-', alpha=0.5, linewidth=1, label='All Data')
    
    # Highlight the valid regions
    valid_df = df[good_flight_mask].copy()
    ax.plot(valid_df['time_seconds'], valid_df['Altitude [m]'], 'g-', linewidth=1.5, label='Valid Flight Path')
    
    # Highlight regions with altitude changes
    excluded_altitude = df[altitude_change_mask].copy()
    if len(excluded_altitude) > 0:
        ax.scatter(excluded_altitude['time_seconds'], excluded_altitude['Altitude [m]'], 
                 color='red', s=10, alpha=0.7, label='Excluded: Rapid Altitude Changes')
    
    # Highlight regions with unstable altitude
    excluded_unstable = df[unstable_altitude_mask].copy()
    if len(excluded_unstable) > 0:
        ax.scatter(excluded_unstable['time_seconds'], excluded_unstable['Altitude [m]'], 
                 color='orange', s=10, alpha=0.7, label='Excluded: Unstable Altitude')
    
    ax.set_title('Altitude Over Time with Excluded Regions')
    ax.set_ylabel('Altitude (m)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Speed with excluded regions highlighted
    ax = axs[1]
    # Plot speed for all points
    ax.plot(df['time_seconds'], df['Speed'], 'b-', alpha=0.5, linewidth=1, label='All Data')
    
    # Highlight the valid regions
    ax.plot(valid_df['time_seconds'], valid_df['Speed'], 'g-', linewidth=1.5, label='Valid Flight Path')
    
    # Highlight hover regions
    excluded_hover = df[hover_mask].copy()
    if len(excluded_hover) > 0:
        ax.scatter(excluded_hover['time_seconds'], excluded_hover['Speed'], 
                 color='purple', s=10, alpha=0.7, label='Excluded: Hovering/Slow Movement')
    
    # Highlight regions with erratic speed changes
    excluded_erratic = df[slow_erratic_mask].copy()
    if len(excluded_erratic) > 0:
        ax.scatter(excluded_erratic['time_seconds'], excluded_erratic['Speed'], 
                 color='magenta', s=10, alpha=0.7, label='Excluded: Slow with Erratic Speed Changes')
    
    ax.set_title('Speed Over Time with Excluded Regions')
    ax.set_ylabel('Speed (m/s)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Altitude Rate and Standard Deviation
    ax = axs[2]
    ax.plot(df['time_seconds'], df['Altitude_rate_abs'], 'c-', alpha=0.7, linewidth=1, label='Altitude Rate (abs)')
    ax.plot(df['time_seconds'], df['Altitude_rolling_std_short'], 'm-', alpha=0.7, linewidth=1, label='Altitude Rolling Std (short)')
    
    # Add threshold lines if thresholds were provided
    if thresholds and 'altitude_rate_threshold' in thresholds:
        ax.axhline(y=thresholds['altitude_rate_threshold'], color='r', linestyle='--', alpha=0.5, 
                 label=f'Rate Threshold ({thresholds["altitude_rate_threshold"]} m/s)')
    if thresholds and 'altitude_std_threshold' in thresholds:
        ax.axhline(y=thresholds['altitude_std_threshold'], color='orange', linestyle='--', alpha=0.5, 
                 label=f'Std Threshold ({thresholds["altitude_std_threshold"]} m)')

    
    ax.set_title('Altitude Rate and Standard Deviation')
    ax.set_ylabel('Value')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Speed Metrics
    ax = axs[3]
    ax.plot(df['time_seconds'], df['Speed_rolling_mean'], 'b-', alpha=0.7, linewidth=1, label='Speed Rolling Mean')
    ax.plot(df['time_seconds'], df['Speed_diff_abs'], 'r-', alpha=0.4, linewidth=0.5, label='Speed Change Rate (abs)')
    ax.plot(df['time_seconds'], df['Speed_diff_rolling'], 'g-', alpha=0.7, linewidth=1, label='Speed Change Rolling Mean')
    
    # Add threshold line for hover speed if thresholds were provided
    if thresholds and 'hover_speed_threshold' in thresholds:
        ax.axhline(y=thresholds['hover_speed_threshold'], color='purple', linestyle='--', alpha=0.5, 
                 label=f'Hover Threshold ({thresholds["hover_speed_threshold"]} m/s)')
    if thresholds and 'speed_change_threshold' in thresholds:
        ax.axhline(y=thresholds['speed_change_threshold'], color='magenta', linestyle='--', alpha=0.5, 
                 label=f'Speed Change Threshold ({thresholds["speed_change_threshold"]})')
    
    ax.set_title('Speed Metrics')
    ax.set_ylabel('Value')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Plot 5: The resulting mask visualization
    ax = axs[4]
    
    # Create a colormap for the different mask types
    mask_values = np.zeros(len(df))
    
    # Create high-speed mask based on thresholds
    high_speed_mask_begin = None
    high_speed_mask_middle = None
    high_speed_mask = None
    
    if thresholds and 'high_speed_threshold_begin' in thresholds and 'high_speed_threshold_middle' in thresholds:
        # Recreate masks to visualize high-speed segments
        total_rows = len(df)
        begin_segment = np.zeros(total_rows, dtype=bool)
        middle_segment = np.zeros(total_rows, dtype=bool)
        
        # Mark beginning 10% and middle segment
        begin_segment[:int(total_rows * 0.1)] = True
        middle_segment[int(total_rows * 0.1):int(total_rows * 0.9)] = True
        
        high_speed_mask_begin = (
            begin_segment &
            (df['Speed'] > thresholds['high_speed_threshold_begin'])
        )
        
        high_speed_mask_middle = (
            middle_segment &
            (df['Speed'] > thresholds['high_speed_threshold_middle'])
        )
        
        high_speed_mask = high_speed_mask_begin | high_speed_mask_middle
    
    # Use boolean indexing directly on mask_values array
    mask_values[good_flight_mask] = 1  # Good flight segments
    mask_values[altitude_change_mask] = 2  # Altitude change
    mask_values[unstable_altitude_mask] = 3  # Unstable altitude 
    mask_values[hover_mask] = 4  # Hovering
    mask_values[slow_erratic_mask] = 5  # Erratic speed changes
    
    # Apply high-speed mask with higher priority if available
    if high_speed_mask is not None:
        mask_values[high_speed_mask] = 6  # High speed
    
    # Plot scatter points colored by mask type
    ax.scatter(df['time_seconds'], np.ones(len(df)), c=mask_values, 
              cmap='viridis', marker='|', s=20, alpha=0.7)
    
    # Create custom legend
    legend_elements = [
        Patch(facecolor='#2ca02c', label='Valid Data'),
        Patch(facecolor='#ff7f0e', label='Altitude Change'),
        Patch(facecolor='#d62728', label='Unstable Altitude'),
        Patch(facecolor='#9467bd', label='Hovering'),
        Patch(facecolor='#8c564b', label='Erratic Speed Changes')
    ]
    
    # Add high-speed to legend if applicable
    if high_speed_mask is not None:
        legend_elements.append(Patch(facecolor='#17becf', label='High Speed'))
    ax.legend(handles=legend_elements, loc='upper right')
    
    ax.set_title('Flight Segment Classification')
    ax.set_xlabel('Time (seconds)')
    ax.set_yticks([])
    ax.grid(True, alpha=0.3)
    
    # Save the figure
    plt.tight_layout()
    fig.savefig(os.path.join(filter_dir, 'flight_path_filtering_analysis.png'), dpi=300)
    print(f"Flight path filtering visualization saved to {os.path.join(filter_dir, 'flight_path_filtering_analysis.png')}")
    
    # Create an interactive Plotly version for more detailed exploration
    try:
        # Create a new figure for Plotly
        fig_plotly = go.Figure()
        
        # Add traces for altitude and speed in two subplots
        fig_plotly = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                                 subplot_titles=['Altitude Over Time', 'Speed Over Time'])
        
        # Add altitude trace
        fig_plotly.add_trace(
            go.Scatter(x=df['time_seconds'], y=df['Altitude [m]'], mode='lines', name='Altitude', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Add valid segments for altitude
        fig_plotly.add_trace(
            go.Scatter(x=valid_df['time_seconds'], y=valid_df['Altitude [m]'], mode='lines', name='Valid Segments', 
                     line=dict(color='green', width=2)),
            row=1, col=1
        )
        
        # Add speed trace
        fig_plotly.add_trace(
            go.Scatter(x=df['time_seconds'], y=df['Speed'], mode='lines', name='Speed', line=dict(color='blue')),
            row=2, col=1
        )
        
        # Add valid segments for speed
        fig_plotly.add_trace(
            go.Scatter(x=valid_df['time_seconds'], y=valid_df['Speed'], mode='lines', name='Valid Segments', 
                     line=dict(color='green', width=2)),
            row=2, col=1
        )
        
        # Update layout
        fig_plotly.update_layout(
            title='Flight Path Filtering Analysis',
            xaxis_title='Time (seconds)',
            showlegend=True,
            height=800
        )
        
        # Save interactive version
        interactive_dir = os.path.join(output_dir, 'interactive_plots')
        os.makedirs(interactive_dir, exist_ok=True)
        fig_plotly.write_html(os.path.join(interactive_dir, 'flight_path_filtering_interactive.html'))
        print(f"Interactive flight path filtering visualization saved to {os.path.join(interactive_dir, 'flight_path_filtering_interactive.html')}")
    except Exception as e:
        print(f"Warning: Could not create interactive flight path filtering visualization: {str(e)}")


def plot_interactive_utm_paths(df, output_dir, interactive_dir=None, enable=True):
    """Create an interactive Plotly plot for actual UTM positions"""
    if not enable:
        return
        
    # If interactive_dir is not provided, use output_dir
    if interactive_dir is None:
        interactive_dir = output_dir
    
    # Create a DataFrame with the actual UTM coordinates
    df_plot = pd.DataFrame({
        'UTM_Easting_Original': df['UTM_Easting'],
        'UTM_Northing_Original': df['UTM_Northing'],
        'UTM_Easting_1': df['UTM_Easting1'],
        'UTM_Northing_1': df['UTM_Northing1'],
        'UTM_Easting_2': df['UTM_Easting2'],
        'UTM_Northing_2': df['UTM_Northing2'],
    })
    
    # Add GPS Time if available
    if 'GPSTime [hh:mm:ss.sss]' in df.columns:
        df_plot['GPS_Time'] = df['GPSTime [hh:mm:ss.sss]']
    
    # Add a sample index for reference
    df_plot['Sample_Index'] = range(len(df_plot))
    
    # Create a directory for interactive plots
    html_dir = os.path.join(output_dir, 'interactive_plots')
    os.makedirs(html_dir, exist_ok=True)
    
    # Create interactive scatter plot of actual UTM positions
    fig = go.Figure()
    
    # Add traces for each UTM coordinate set
    fig.add_trace(go.Scatter(
        x=df_plot['UTM_Easting_Original'], 
        y=df_plot['UTM_Northing_Original'],
        mode='lines',
        name='UTM Original',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=df_plot['UTM_Easting_1'], 
        y=df_plot['UTM_Northing_1'],
        mode='lines',
        name='UTM 1',
        line=dict(color='red', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=df_plot['UTM_Easting_2'], 
        y=df_plot['UTM_Northing_2'],
        mode='lines',
        name='UTM 2',
        line=dict(color='green', width=2)
    ))
    
    # Add the starting points for each coordinate system
    fig.add_trace(go.Scatter(
        x=[df_plot['UTM_Easting_Original'].iloc[0]],
        y=[df_plot['UTM_Northing_Original'].iloc[0]],
        mode='markers',
        name='Start UTM Original',
        marker=dict(color='blue', size=10, symbol='circle')
    ))
    
    fig.add_trace(go.Scatter(
        x=[df_plot['UTM_Easting_1'].iloc[0]],
        y=[df_plot['UTM_Northing_1'].iloc[0]],
        mode='markers',
        name='Start UTM 1',
        marker=dict(color='red', size=10, symbol='circle')
    ))
    
    fig.add_trace(go.Scatter(
        x=[df_plot['UTM_Easting_2'].iloc[0]],
        y=[df_plot['UTM_Northing_2'].iloc[0]],
        mode='markers',
        name='Start UTM 2',
        marker=dict(color='green', size=10, symbol='circle')
    ))
    
    # Update layout
    fig.update_layout(
        title='Interactive UTM Coordinate Paths',
        xaxis_title='UTM Easting (m)',
        yaxis_title='UTM Northing (m)',
        legend=dict(x=0, y=1),
        hovermode='closest',
        width=1000,
        height=800,
    )
    
    # Set equal aspect ratio by making y-axis scale match x-axis scale
    fig.update_yaxes(scaleanchor='x', scaleratio=1)
    
    lower_x = min(df_plot['UTM_Easting_Original'].min(), 
                 df_plot['UTM_Easting_1'].min(), 
                 df_plot['UTM_Easting_2'].min())
    upper_x = max(df_plot['UTM_Easting_Original'].max(), 
                 df_plot['UTM_Easting_1'].max(), 
                 df_plot['UTM_Easting_2'].max())
    lower_y = min(df_plot['UTM_Northing_Original'].min(), 
                 df_plot['UTM_Northing_1'].min(), 
                 df_plot['UTM_Northing_2'].min())
    upper_y = max(df_plot['UTM_Northing_Original'].max(), 
                 df_plot['UTM_Northing_1'].max(), 
                 df_plot['UTM_Northing_2'].max())
    
    range_x = upper_x - lower_x
    range_y = upper_y - lower_y
    
    fig.update_xaxes(range=[lower_x - range_x * 0.1, upper_x + range_x * 0.1])
    fig.update_yaxes(range=[lower_y - range_y * 0.1, upper_y + range_y * 0.1])
    
    # Add grid lines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    # Save to HTML file for interactive viewing
    output_path = os.path.join(html_dir, 'utm_paths_interactive.html')
    fig.write_html(output_path)
    print(f"Interactive UTM paths plot saved to {output_path}")
    
    # Also create a separate interactive plot for each UTM coordinate pair over time
    # This allows examining how each component changes over time
    
    # Plot UTM Easting over time
    fig_easting = go.Figure()
    
    # Add traces for each UTM Easting
    fig_easting.add_trace(go.Scatter(
        x=df_plot['Sample_Index'], 
        y=df_plot['UTM_Easting_Original'],
        mode='lines',
        name='UTM Easting Original',
        line=dict(color='blue', width=2)
    ))
    
    fig_easting.add_trace(go.Scatter(
        x=df_plot['Sample_Index'], 
        y=df_plot['UTM_Easting_1'],
        mode='lines',
        name='UTM Easting 1',
        line=dict(color='red', width=2)
    ))
    
    fig_easting.add_trace(go.Scatter(
        x=df_plot['Sample_Index'], 
        y=df_plot['UTM_Easting_2'],
        mode='lines',
        name='UTM Easting 2',
        line=dict(color='green', width=2)
    ))
    
    # Update layout
    fig_easting.update_layout(
        title='Interactive UTM Easting Over Time',
        xaxis_title='Sample Index',
        yaxis_title='UTM Easting (m)',
        legend=dict(x=0, y=1),
        hovermode='closest',
        width=1200,
        height=600,
    )
    
    # Set y-axis range similar to the static plot
    fig_easting.update_yaxes(range=[lower_x - range_x * 0.1, upper_x + range_x * 0.1])
    
    # Add grid lines
    fig_easting.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig_easting.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    # Save to HTML file
    easting_output_path = os.path.join(interactive_dir, 'utm_easting_interactive.html')
    fig_easting.write_html(easting_output_path)
    print(f"Interactive UTM easting plot saved to {easting_output_path}")
    
    # Plot UTM Northing over time
    fig_northing = go.Figure()
    
    # Add traces for each UTM Northing
    fig_northing.add_trace(go.Scatter(
        x=df_plot['Sample_Index'], 
        y=df_plot['UTM_Northing_Original'],
        mode='lines',
        name='UTM Northing Original',
        line=dict(color='blue', width=2)
    ))
    
    fig_northing.add_trace(go.Scatter(
        x=df_plot['Sample_Index'], 
        y=df_plot['UTM_Northing_1'],
        mode='lines',
        name='UTM Northing 1',
        line=dict(color='red', width=2)
    ))
    
    fig_northing.add_trace(go.Scatter(
        x=df_plot['Sample_Index'], 
        y=df_plot['UTM_Northing_2'],
        mode='lines',
        name='UTM Northing 2',
        line=dict(color='green', width=2)
    ))
    
    # Update layout
    fig_northing.update_layout(
        title='Interactive UTM Northing Over Time',
        xaxis_title='Sample Index',
        yaxis_title='UTM Northing (m)',
        legend=dict(x=0, y=1),
        hovermode='closest',
        width=1200,
        height=600,
    )
    
    # Set y-axis range similar to the static plot
    fig_northing.update_yaxes(range=[lower_y - range_y * 0.1, upper_y + range_y * 0.1])
    
    # Add grid lines
    fig_northing.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig_northing.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    # Save to HTML file
    northing_output_path = os.path.join(interactive_dir, 'utm_northing_interactive.html')
    fig_northing.write_html(northing_output_path)
    print(f"Interactive UTM northing plot saved to {northing_output_path}")

def plot_interpolate_residuals(df, output_dir, enable=True, cut_beginning_pct=0, cut_end_pct=0, middlecut1=0, middlecut2=0):
    """
    Generates two graphs on one plot with viridis colourscale of B1 & B2 Residual data throughout the flight path, interpolating the residual data to do so.
    Uses intelligent filtering based on speed, altitude stability, and residual values to isolate the useful parts of the flight.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with flight data
    output_dir : str
        Directory to save output plots
    enable : bool, default=True
        Whether to execute this function
    cut_beginning_pct : float, default=0
        Percentage of data to cut from the beginning (0-100)
    cut_end_pct : float, default=0
        Percentage of data to cut from the end (0-100)
    middlecut1 : float, default=0
        Lower bound percentage for middle section to cut (0-100)
    middlecut2 : float, default=0
        Upper bound percentage for middle section to cut (0-100)
    """
    if not enable:
        return
        
    # Validate cutting percentages
    cut_beginning_pct = max(0, min(cut_beginning_pct, 50))  # Cap at 50%
    cut_end_pct = max(0, min(cut_end_pct, 50))             # Cap at 50%
    
    # Validate middle cut percentages (ensure middlecut1 < middlecut2)
    if middlecut1 > 0 and middlecut2 > 0 and middlecut1 < middlecut2:
        middlecut1 = max(0, min(middlecut1, 90))  # Cap at 90%
        middlecut2 = max(0, min(middlecut2, 90))  # Cap at 90%
    else:
        # Reset if invalid or both are 0
        middlecut1 = 0
        middlecut2 = 0
        
    print("Creating interpolated residual plots...")
    
    # Create a working copy of the dataframe
    work_df = df.copy()
    
    # Calculate heading (direction of movement) between consecutive points
    work_df['Heading'] = np.arctan2(
        work_df['UTM_Easting'].diff(), 
        work_df['UTM_Northing'].diff()
    ) * 180 / np.pi  # Convert to degrees
    
    # Calculate distances between consecutive points
    work_df['Distance'] = np.sqrt(
        (work_df['UTM_Easting'].diff() ** 2) + 
        (work_df['UTM_Northing'].diff() ** 2)
    )
    
    # Filter out points with NaN distance (first point)
    work_df = work_df.dropna(subset=['Distance', 'Heading'])
    
    # Calculate speed as distance/time
    work_df['GPSTime_dt'] = pd.to_datetime(work_df['GPSTime [hh:mm:ss.sss]'], format='%H:%M:%S.%f')
    work_df['Time_diff'] = work_df['GPSTime_dt'].diff().dt.total_seconds()
    work_df['Speed'] = work_df['Distance'] / work_df['Time_diff']
    work_df = work_df.dropna(subset=['Speed'])
    
    # Replace infinite values with NaN and interpolate them
    work_df['Speed'] = work_df['Speed'].replace([np.inf, -np.inf], np.nan)
    work_df['Speed'] = work_df['Speed'].fillna(work_df['Speed'].median())
    
    # Calculate altitude changes - a major indicator of takeoff/landing/return-to-home
    # Use both short-term and long-term windows to capture rapid and gradual changes
    short_window = min(500, len(work_df) // 50)   # Short window for detecting rapid changes
    long_window = min(2000, len(work_df) // 20)   # Longer window for overall patterns
    
    # Calculate altitude rate of change (meters per second)
    work_df['Altitude_diff'] = work_df['Altitude [m]'].diff()
    work_df['Altitude_rate'] = work_df['Altitude_diff'] / work_df['Time_diff']
    
    # Calculate rolling statistics for altitude and speed
    work_df['Altitude_rolling_std_short'] = work_df['Altitude [m]'].rolling(window=short_window, center=True).std()
    work_df['Altitude_rolling_std_long'] = work_df['Altitude [m]'].rolling(window=long_window, center=True).std()
    work_df['Altitude_rate_abs_rolling'] = work_df['Altitude_rate'].abs().rolling(window=short_window, center=True).mean()
    
    work_df['Speed_rolling_mean'] = work_df['Speed'].rolling(window=long_window, center=True).mean()
    work_df['Speed_rolling_std'] = work_df['Speed'].rolling(window=long_window, center=True).std()
    
    # Identify speed spikes (sudden acceleration or deceleration)
    work_df['Speed_diff'] = work_df['Speed'].diff()
    work_df['Speed_rate'] = work_df['Speed_diff'] / work_df['Time_diff']
    work_df['Speed_rate_abs_rolling'] = work_df['Speed_rate'].abs().rolling(window=short_window, center=True).mean()
    
    # Fill NaN values created by the rolling calculation
    cols_to_fill = [
        'Altitude_rolling_std_short', 'Altitude_rolling_std_long', 'Altitude_rate_abs_rolling',
        'Speed_rolling_mean', 'Speed_rolling_std', 'Speed_rate_abs_rolling'
    ]
    
    for col in cols_to_fill:
        work_df[col] = work_df[col].fillna(work_df[col].median())
    
    print("Filtering flight data based on altitude changes and speed patterns...")
    
    # Calculate shorter rolling windows for more precise detection of problematic segments
    short_window = min(300, len(work_df) // 80)   # Very short window for immediate changes
    medium_window = min(800, len(work_df) // 40)   # Medium window for detecting patterns
    
    # Calculate rolling metrics for altitude
    work_df['Altitude_diff'] = work_df['Altitude [m]'].diff()  # Frame-to-frame altitude difference
    work_df['Altitude_rate'] = work_df['Altitude_diff'] / work_df['Time_diff']  # Rate of altitude change
    work_df['Altitude_rate_abs'] = work_df['Altitude_rate'].abs()  # Absolute rate (both ascent and descent)
    
    # Calculate short and medium-term altitude variation
    work_df['Altitude_rolling_std_short'] = work_df['Altitude [m]'].rolling(window=short_window, center=True).std()
    work_df['Altitude_rolling_mean_short'] = work_df['Altitude [m]'].rolling(window=short_window, center=True).mean()
    work_df['Altitude_rate_rolling_mean'] = work_df['Altitude_rate_abs'].rolling(window=medium_window, center=True).mean()
    
    # Calculate speed metrics - both short and longer term
    work_df['Speed_rolling_std_short'] = work_df['Speed'].rolling(window=short_window, center=True).std()
    work_df['Speed_diff'] = work_df['Speed'].diff() / work_df['Time_diff']  # Speed change rate (acceleration)
    work_df['Speed_diff_abs'] = work_df['Speed_diff'].abs()
    work_df['Speed_diff_rolling'] = work_df['Speed_diff_abs'].rolling(window=short_window, center=True).mean()
    
    # Fill NaN values in all calculated columns
    cols_to_fill = [
        'Altitude_rate', 'Altitude_rate_abs', 'Altitude_rolling_std_short', 
        'Altitude_rolling_mean_short', 'Altitude_rate_rolling_mean',
        'Speed_rolling_std_short', 'Speed_diff', 'Speed_diff_abs', 'Speed_diff_rolling'
    ]
    
    for col in cols_to_fill:
        work_df[col] = work_df[col].fillna(work_df[col].median())
    
    # Create a position-based mask to identify beginning and end segments
    total_rows = len(work_df)
    
    # Apply user-defined cutting percentages
    cut_beginning_idx = int(total_rows * (cut_beginning_pct / 100)) if cut_beginning_pct > 0 else 0
    cut_end_idx = int(total_rows * (1 - cut_end_pct / 100)) if cut_end_pct > 0 else total_rows
    
    # Apply middle cut if specified
    middle_cut_start_idx = 0
    middle_cut_end_idx = 0
    if middlecut1 > 0 and middlecut2 > 0 and middlecut2 > middlecut1:
        middle_cut_start_idx = int(total_rows * (middlecut1 / 100))
        middle_cut_end_idx = int(total_rows * (middlecut2 / 100))
    
    # Create hard cutoff mask based on user parameters
    hard_cutoff_mask = np.ones(total_rows, dtype=bool)
    if cut_beginning_pct > 0:
        hard_cutoff_mask[:cut_beginning_idx] = False
    if cut_end_pct > 0:
        hard_cutoff_mask[cut_end_idx:] = False
    # Apply middle section cut if specified
    if middlecut1 > 0 and middlecut2 > 0:
        hard_cutoff_mask[middle_cut_start_idx:middle_cut_end_idx] = False
    
    # Create position masks for the different segments (for intelligent filtering)
    # Use 10% for default segment identification (after any hard cutoffs)
    begin_segment = np.zeros(total_rows, dtype=bool)
    end_segment = np.zeros(total_rows, dtype=bool)
    middle_segment = np.zeros(total_rows, dtype=bool)
    
    # Mark beginning 10% and ending 10% of remaining data after hard cutoffs
    valid_indices = np.where(hard_cutoff_mask)[0]
    if len(valid_indices) > 0:
        new_start = valid_indices[0]
        new_end = valid_indices[-1]
        segment_size = int((new_end - new_start) * 0.1)
        
        begin_segment[new_start:new_start+segment_size] = True
        end_segment[new_end-segment_size:new_end+1] = True
        middle_segment[new_start+segment_size:new_end-segment_size] = True
    
    # Define thresholds with different severity for different segments
    # More lenient for the middle, stricter for beginning/end
    mid_alt_rate_threshold = 0.35      # Higher threshold for middle (only extreme altitude changes)
    begin_end_alt_rate_threshold = 0.2  # Lower for beginning/end (more filtering)
    
    mid_alt_std_threshold = 1.2        # More tolerant of altitude variations in middle segment
    begin_end_alt_std_threshold = 0.7   # Stricter for beginning/end
    
    hover_speed_threshold = 0.3        # More lenient threshold (retain more slow movement data)
    begin_end_hover_speed_threshold = 0.4 # Slightly stricter for takeoff/landing
    
    speed_change_threshold = 0.4       # More tolerant of acceleration/deceleration in middle
    begin_end_speed_change_threshold = 0.3 # Stricter for beginning/end
    
    # Define high speed thresholds - especially important at the beginning
    high_speed_threshold_begin = 3.0    # m/s - stricter high speed threshold for beginning
    high_speed_threshold_middle = 5.0   # m/s - more lenient high speed threshold for middle
    
    # 1. Altitude change masks with segment-specific thresholds
    alt_change_mask_begin_end = (
        (begin_segment | end_segment) & 
        (work_df['Altitude_rate_rolling_mean'] > begin_end_alt_rate_threshold)
    )
    
    alt_change_mask_middle = (
        middle_segment & 
        (work_df['Altitude_rate_rolling_mean'] > mid_alt_rate_threshold)
    )
    
    # Combine altitude change masks
    altitude_change_mask = alt_change_mask_begin_end | alt_change_mask_middle
    
    # 2. Unstable altitude masks with segment-specific thresholds
    unstable_alt_mask_begin_end = (
        (begin_segment | end_segment) & 
        (work_df['Altitude_rolling_std_short'] > begin_end_alt_std_threshold)
    )
    
    unstable_alt_mask_middle = (
        middle_segment & 
        (work_df['Altitude_rolling_std_short'] > mid_alt_std_threshold)
    )
    
    # Combine unstable altitude masks
    unstable_altitude_mask = unstable_alt_mask_begin_end | unstable_alt_mask_middle
    
    # 3. Hover masks with segment-specific thresholds
    hover_mask_begin_end = (
        (begin_segment | end_segment) & 
        (work_df['Speed_rolling_mean'] < begin_end_hover_speed_threshold)
    )
    
    hover_mask_middle = (
        middle_segment & 
        (work_df['Speed_rolling_mean'] < hover_speed_threshold)
    )
    
    # Combine hover masks
    hover_mask = hover_mask_begin_end | hover_mask_middle
    
    # 4. Modified approach for speed changes - only filter out extreme cases
    # For middle segment, only filter very slow moving with extremely large changes
    slow_erratic_mask_middle = (
        middle_segment &
        (work_df['Speed_rolling_mean'] < hover_speed_threshold * 0.6) &  # Very slow moving
        (work_df['Speed_diff_rolling'] > speed_change_threshold * 2.5)    # With extremely large speed changes
    )
    
    # For begin/end segments, be a bit stricter
    slow_erratic_mask_begin_end = (
        (begin_segment | end_segment) &
        (work_df['Speed_rolling_mean'] < begin_end_hover_speed_threshold * 0.8) &
        (work_df['Speed_diff_rolling'] > begin_end_speed_change_threshold * 2.0)
    )
    
    # Combine slow erratic masks
    slow_erratic_mask = slow_erratic_mask_middle | slow_erratic_mask_begin_end
    
    # Add masks for high-speed segments, with stricter filtering at the beginning
    high_speed_mask_begin = (
        begin_segment &
        (work_df['Speed'] > high_speed_threshold_begin)  # Strict threshold for beginning
    )
    
    high_speed_mask_middle = (
        middle_segment &
        (work_df['Speed'] > high_speed_threshold_middle)  # More lenient for middle segments
    )
    
    # Combine high speed masks
    high_speed_mask = high_speed_mask_begin | high_speed_mask_middle
    
    # Create a mask to identify problematic segments with segment-specific focus
    problematic_segments = (
        altitude_change_mask |    # Rapid altitude changes (segment-specific thresholds)
        unstable_altitude_mask |  # Unstable altitude (segment-specific thresholds)
        # Only filter hovering with very low variation (keeps more useful slow data)
        (hover_mask & (work_df['Speed_rolling_std_short'] < 0.05)) |  
        slow_erratic_mask |       # Extreme erratic speed changes (segment-specific thresholds)
        high_speed_mask           # High-speed segments (stricter at beginning)
    )
    
    # The good flight segments are those that are not problematic and not in hard cutoff regions
    good_flight_mask = ~problematic_segments & hard_cutoff_mask
    
    # Print detection statistics
    altitude_pct = 100 * altitude_change_mask.sum() / len(work_df)
    unstable_alt_pct = 100 * unstable_altitude_mask.sum() / len(work_df)
    hover_pct = 100 * hover_mask.sum() / len(work_df)
    slow_erratic_pct = 100 * slow_erratic_mask.sum() / len(work_df)
    high_speed_pct = 100 * high_speed_mask.sum() / len(work_df)
    high_speed_begin_pct = 100 * high_speed_mask_begin.sum() / len(begin_segment)
    
    print(f"Detected segments with rapid altitude changes: {altitude_pct:.1f}%")
    print(f"Detected segments with unstable altitude: {unstable_alt_pct:.1f}%")
    print(f"Detected segments with hover behavior: {hover_pct:.1f}%")
    print(f"Detected slow segments with erratic speed changes: {slow_erratic_pct:.1f}%")
    print(f"Detected high-speed segments overall: {high_speed_pct:.1f}%")
    print(f"Detected high-speed segments in beginning 10%: {high_speed_begin_pct:.1f}%")
    
    # Apply the mask to select only the valid, consistent flight data
    valid_data = work_df[good_flight_mask].copy()
    
    # Calculate impact of hard cutoffs vs intelligent filtering
    hard_cutoff_kept = hard_cutoff_mask.sum()
    hard_cutoff_pct = 100 * hard_cutoff_kept / len(work_df)
    
    # Print summary statistics
    if cut_beginning_pct > 0 or cut_end_pct > 0 or (middlecut1 > 0 and middlecut2 > 0):
        cutoff_description = []
        if cut_beginning_pct > 0:
            cutoff_description.append(f"{cut_beginning_pct}% from beginning")
        if cut_end_pct > 0:
            cutoff_description.append(f"{cut_end_pct}% from end")
        if middlecut1 > 0 and middlecut2 > 0:
            middle_range = middlecut2 - middlecut1
            cutoff_description.append(f"{middle_range}% from middle ({middlecut1}% to {middlecut2}%)")
            
        print(f"Manual cutoff applied: {', '.join(cutoff_description)}")
        print(f"After manual cutoff: {hard_cutoff_kept} points ({hard_cutoff_pct:.1f}% of original)")
    
    print(f"Original data points: {len(work_df)}, After all filtering: {len(valid_data)} ({len(valid_data)/len(work_df)*100:.1f}%)")
    
    # Visualize the filtering masks to show excluded regions
    # Create a dictionary with the thresholds - use mid-segment values for visualization
    thresholds = {
        'altitude_rate_threshold': mid_alt_rate_threshold,
        'altitude_std_threshold': mid_alt_std_threshold,
        'hover_speed_threshold': hover_speed_threshold,
        'speed_change_threshold': speed_change_threshold,
        'high_speed_threshold_begin': high_speed_threshold_begin,
        'high_speed_threshold_middle': high_speed_threshold_middle
    }
    
    visualize_filtering_masks(work_df, altitude_change_mask, unstable_altitude_mask, hover_mask, slow_erratic_mask, good_flight_mask, output_dir, thresholds)
    
    if len(valid_data) < 10:
        print("Not enough valid data points for interpolation after cleaning")
        return
    
    # Extract coordinates for both versions
    easting = valid_data['UTM_Easting'].values
    northing = valid_data['UTM_Northing'].values
    r1_values = valid_data['R1 [nT]'].values
    r2_values = valid_data['R2 [nT]'].values
    
    # Define the grid for interpolation (same for both versions)
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
    
    # First create the interpolated plot with all data points after trimming start/end
    create_and_save_interpolation_plots(
        points, easting, northing, r1_values, r2_values, xi, yi, xi_grid, yi_grid,
        output_dir, "all_data", "With All Data Points"
    )
    
    # Now create a version with extreme outliers removed
    # For each residual, identify outliers using the IQR method
    def remove_extreme_outliers(values):
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        
        # Consider values beyond 3*IQR from the quartiles as extreme outliers
        # This is more permissive than the standard 1.5*IQR to keep more of the interesting anomalies
        lower_bound = q1 - 3 * iqr
        upper_bound = q3 + 3 * iqr
        
        return (values >= lower_bound) & (values <= upper_bound)
    
    # Apply outlier filtering separately to each residual dataset
    r1_valid_mask = remove_extreme_outliers(r1_values)
    r2_valid_mask = remove_extreme_outliers(r2_values)
    
    # We'll only use points where both R1 and R2 are not extreme outliers
    valid_outlier_mask = r1_valid_mask & r2_valid_mask
    
    # Filter the data
    filtered_easting = easting[valid_outlier_mask]
    filtered_northing = northing[valid_outlier_mask]
    filtered_r1_values = r1_values[valid_outlier_mask]
    filtered_r2_values = r2_values[valid_outlier_mask]
    filtered_points = np.column_stack((filtered_easting, filtered_northing))
    
    outliers_removed = len(easting) - len(filtered_easting)
    outlier_percentage = (outliers_removed / len(easting)) * 100
    print(f"Removed {outliers_removed} extreme outliers ({outlier_percentage:.2f}% of data)")
    
    # Create the filtered version
    create_and_save_interpolation_plots(
        filtered_points, filtered_easting, filtered_northing, 
        filtered_r1_values, filtered_r2_values, xi, yi, xi_grid, yi_grid,
        output_dir, "filtered", "With Extreme Outliers Removed"
    )


def create_and_save_interpolation_plots(points, easting, northing, r1_values, r2_values, 
                                      xi, yi, xi_grid, yi_grid, output_dir, suffix, title_suffix):
    """Helper function to create and save interpolation plots"""
    # Interpolate the residual values onto the grid
    r1_grid = griddata(points, r1_values, (xi_grid, yi_grid), method='cubic')
    r2_grid = griddata(points, r2_values, (xi_grid, yi_grid), method='cubic')
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), sharex=True, sharey=True)
    
    # Determine common color scale for both plots
    vmin = min(np.nanmin(r1_grid), np.nanmin(r2_grid))
    vmax = max(np.nanmax(r1_grid), np.nanmax(r2_grid))
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    # Plot B1res
    ax1.contourf(xi_grid, yi_grid, r1_grid, levels=50, cmap='viridis', norm=norm)
    ax1.scatter(easting, northing, c=r1_values, cmap='viridis', norm=norm, s=5, alpha=0.5)
    ax1.set_title(f'B1 Residual (nT) - {title_suffix}')
    ax1.set_xlabel('UTM Easting (m)')
    ax1.set_ylabel('UTM Northing (m)')
    ax1.grid(True, alpha=0.3)
    
    # Plot B2res
    contour2 = ax2.contourf(xi_grid, yi_grid, r2_grid, levels=50, cmap='viridis', norm=norm)
    ax2.scatter(easting, northing, c=r2_values, cmap='viridis', norm=norm, s=5, alpha=0.5)
    ax2.set_title(f'B2 Residual (nT) - {title_suffix}')
    ax2.set_xlabel('UTM Easting (m)')
    ax2.set_ylabel('UTM Northing (m)')
    ax2.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = fig.colorbar(contour2, ax=[ax1, ax2], orientation='horizontal', pad=0.05, fraction=0.05, aspect=50)
    cbar.set_label('Residual Magnetic Field (nT)')
    
    # Add flight path visualization
    ax1.plot(easting, northing, 'k-', linewidth=0.5, alpha=0.5)
    ax2.plot(easting, northing, 'k-', linewidth=0.5, alpha=0.5)
    
    # Add markers for start and end points
    ax1.plot(easting[0], northing[0], 'ko', markersize=6)
    ax1.plot(easting[-1], northing[-1], 'kx', markersize=6)
    ax2.plot(easting[0], northing[0], 'ko', markersize=6)
    ax2.plot(easting[-1], northing[-1], 'kx', markersize=6)
    
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, f'interpolated_residuals_{suffix}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Interpolated residuals plot ({title_suffix}) saved to {output_path}")
    
    # Create an interactive version with plotly
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Set up directory for interactive plots
        html_dir = os.path.join(output_dir, 'interactive_plots')
        os.makedirs(html_dir, exist_ok=True)
        
        # Create subplot figure
        fig = make_subplots(rows=1, cols=2, 
                          subplot_titles=[f'B1 Residual (nT) - {title_suffix}', 
                                        f'B2 Residual (nT) - {title_suffix}'],
                          shared_yaxes=True)
        
        # Add heatmap traces for B1res and B2res
        fig.add_trace(
            go.Heatmap(
                x=xi,
                y=yi,
                z=r1_grid,
                colorscale='Viridis',
                zmin=vmin,
                zmax=vmax,
                colorbar=dict(title='nT', x=-0.02),
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Heatmap(
                x=xi,
                y=yi,
                z=r2_grid,
                colorscale='Viridis',
                zmin=vmin,
                zmax=vmax,
                colorbar=dict(title='nT', x=1.02),
            ),
            row=1, col=2
        )
        
        # Add scatter traces showing actual data points
        fig.add_trace(
            go.Scatter(
                x=easting,
                y=northing,
                mode='lines+markers',
                marker=dict(color='black', size=3, opacity=0.5),
                line=dict(color='black', width=1, dash='dot'),
                name='Flight Path',
                showlegend=True
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=easting,
                y=northing,
                mode='lines+markers',
                marker=dict(color='black', size=3, opacity=0.5),
                line=dict(color='black', width=1, dash='dot'),
                name='Flight Path',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f'Interpolated Magnetic Field Residuals - {title_suffix}',
            height=800,
            width=1400,
            xaxis_title='UTM Easting (m)',
            yaxis_title='UTM Northing (m)',
            xaxis2_title='UTM Easting (m)'
        )
        
        # Save interactive plot
        interactive_output_path = os.path.join(html_dir, f'interpolated_residuals_{suffix}_interactive.html')
        fig.write_html(interactive_output_path)
        print(f"Interactive interpolated residuals plot ({title_suffix}) saved to {interactive_output_path}")
        
    except Exception as e:
        print(f"Could not create interactive plot: {e}")
        
def main():
    """Run the main program"""
    # First read the data from CSV
    df = read_magnetic_data(CSV_PATH)
    
    # Create output directories if they don't exist
    output_dir = os.path.join(os.path.dirname(os.path.dirname(CSV_PATH)), 'Plots')
    filter_dir = os.path.join(output_dir, 'filter_analysis')
    html_dir = os.path.join(output_dir, 'interactive_plots')
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(filter_dir, exist_ok=True)
    os.makedirs(html_dir, exist_ok=True)
    
    # Set cutoff percentages for flight data
    # These can be adjusted to cut off problematic portions of the flight
    cut_beginning_pct = 19.25  # Cut off first 5% of data
    cut_end_pct = 4       # Cut off last 10% of data
    middlecut1 = 48       # Lower bound for middle section to cut (0 = disabled)
    middlecut2 = 64.25        # Upper bound for middle section to cut (0 = disabled)
    
    # Run the plots
    plot_utm_paths(df, output_dir, enable=PLOT_UTM_PATHS)
    plot_utm_paths_over_time(df, output_dir, enable=PLOT_UTM_TIME)
    plot_magnetic_field_over_time(df, output_dir, enable=PLOT_MAG_FIELD)
    plot_residuals(df, output_dir, enable=PLOT_RESIDUALS)
    plot_utm_components_over_time(df, output_dir, enable=PLOT_UTM_COMPONENTS)
    plot_interactive_utm_paths(df, output_dir, html_dir, enable=PLOT_INTERACTIVE_UTM)
    plot_interpolate_residuals(df, output_dir, enable=PLOT_INTERPOLATION, 
                              cut_beginning_pct=cut_beginning_pct, 
                              cut_end_pct=cut_end_pct,
                              middlecut1=middlecut1,
                              middlecut2=middlecut2)


if __name__ == "__main__":
    main()
