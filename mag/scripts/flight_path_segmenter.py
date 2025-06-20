#!/usr/bin/env python3
"""
Flight Path Segmentation Script
===============================

Processes magnetic survey data from magbase.py output to automatically segment
flight paths into directional segments based on heading consistency.

Features:
- Automatic detection of straight-line flight segments
- Removal of start/end portions and turning segments
- Separation by heading direction for grid pattern surveys
- Comprehensive visualization of flight path and segmentation
- UTM coordinate plotting for quality control
- Modular design for easy extension
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import datetime
import logging
from scipy.ndimage import uniform_filter1d

# =============================================
# CONFIGURATION - Edit these parameters
# =============================================

# Input file path (output from magbase.py)
INPUT_CSV = "/Users/aleksandergarbuz/Documents/SINTEF/data/20250611_081139_MWALK_#0122_processed_20250616_094044.csv"

# Segmentation parameters
HEADING_WINDOW_SIZE = 51           # Window size for heading smoothing (odd number)
MIN_SEGMENT_LENGTH = 100          # Minimum points per segment
HEADING_TOLERANCE = 5.0          # Degrees tolerance for consistent heading
STABILITY_WINDOW = 200            # Points to analyze for path stability
MIN_STABLE_DURATION = 5000          # Minimum points for a stable segment
HEADING_CHANGE_THRESHOLD = 10.0   # Max heading change rate for stability (deg/point)

MIN_POINTS_FOR_VALID_LINE = 50

# Visualization settings
FIGURE_SIZE = (15, 10)
DPI = 300
SHOW_PLOTS = True                 # Set to False to skip displaying plots

# Quality control parameters
MIN_SPEED_MS = 0.5               # Minimum speed to consider valid movement (m/s)
MAX_HEADING_CHANGE_RATE = 5.0    # Max heading change per second (degrees/s)

# Heading combination parameters
COMBINE_OPPOSITE_HEADINGS = True  # Set to True to combine opposite headings (N/S and E/W)
OPPOSITE_HEADING_TOLERANCE = 30   # Tolerance for considering headings as opposite (degrees)

# =============================================
# SETUP AND UTILITIES
# =============================================

def setup_logging():
    """Setup logging for the script"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def calculate_heading(df):
    """
    Calculate heading (bearing) from UTM coordinates
    
    Args:
        df: DataFrame with UTM_Easting and UTM_Northing columns
    
    Returns:
        numpy array of headings in degrees (0-360)
    """
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

def calculate_speed(df):
    """
    Calculate speed from position data
    
    Args:
        df: DataFrame with UTM coordinates and Timestamp
    
    Returns:
        numpy array of speeds in m/s
    """
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

def smooth_heading(headings, window_size):
    """
    Smooth heading data accounting for circular nature (0-360 degrees)
    
    Args:
        headings: array of headings in degrees
        window_size: smoothing window size
    
    Returns:
        smoothed headings
    """
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

# =============================================
# VISUALIZATION FUNCTIONS
# =============================================

def create_overview_plots(df, output_dir):
    """Create overview plots of the flight path"""
    logger = logging.getLogger(__name__)
    logger.info("Creating overview plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE)
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
    headings = calculate_heading(df)
    smoothed_headings = smooth_heading(headings, HEADING_WINDOW_SIZE)
    
    axes[1, 0].plot(time_hours, headings, 'r-', alpha=0.3, linewidth=0.5, label='Raw')
    axes[1, 0].plot(time_hours, smoothed_headings, 'r-', linewidth=2, label='Smoothed')
    axes[1, 0].set_xlabel('Time (hours)')
    axes[1, 0].set_ylabel('Heading (°)')
    axes[1, 0].set_title('Heading vs Time')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 360)
    
    # 4. Speed vs Time
    speeds = calculate_speed(df)
    axes[1, 1].plot(time_hours, speeds, 'm-', linewidth=1)
    axes[1, 1].set_xlabel('Time (hours)')
    axes[1, 1].set_ylabel('Speed (m/s)')
    axes[1, 1].set_title('Speed vs Time')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=MIN_SPEED_MS, color='red', linestyle='--', alpha=0.7, label=f'Min Speed ({MIN_SPEED_MS} m/s)')
    axes[1, 1].legend()
    
    plt.tight_layout()
    overview_path = output_dir / "01_flight_overview.png"
    plt.savefig(overview_path, dpi=DPI, bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close()
    
    logger.info(f"Overview plots saved to {overview_path}")
    return headings, smoothed_headings, speeds

def plot_segmentation_analysis(df, segments, output_dir):
    """Plot segmentation analysis"""
    logger = logging.getLogger(__name__)
    logger.info("Creating segmentation analysis plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE)
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
    headings = calculate_heading(df)
    smoothed_headings = smooth_heading(headings, HEADING_WINDOW_SIZE)
    
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
                seg_headings = calculate_heading(seg_data)
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
    plt.savefig(segmentation_path, dpi=DPI, bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close()
    
    logger.info(f"Segmentation analysis saved to {segmentation_path}")

def plot_utm_verification(segments, output_dir):
    """Create UTM verification plot for final segments showing parallel lines"""
    logger = logging.getLogger(__name__)
    logger.info("Creating UTM verification plot...")
    
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
    plt.savefig(utm_path, dpi=DPI, bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close()
    
    logger.info(f"UTM verification plot saved to {utm_path}")

# =============================================
# SEGMENTATION FUNCTIONS
# =============================================

def detect_stabilized_segments(df):
    """
    Detect stabilized flight segments by analyzing heading consistency and removing
    unstable start/end portions automatically
    
    Args:
        df: DataFrame with flight data
    
    Returns:
        List of tuples (start_idx, end_idx, avg_heading)
    """
    logger = logging.getLogger(__name__)
    
    # Calculate and smooth headings
    headings = calculate_heading(df)
    smoothed_headings = smooth_heading(headings, HEADING_WINDOW_SIZE)
    speeds = calculate_speed(df)
    
    # Calculate heading change rate (stability metric)
    heading_changes = np.abs(np.diff(smoothed_headings))
    # Handle wrap-around (359° to 1°)
    heading_changes = np.minimum(heading_changes, 360 - heading_changes)
    # Pad to match original length
    heading_changes = np.append(heading_changes, heading_changes[-1])
    
    # Calculate rolling stability metrics
    stability_metric = uniform_filter1d(heading_changes, STABILITY_WINDOW, mode='constant', cval=999)
    
    # Identify stable regions (low heading change rate + adequate speed)
    is_stable = (stability_metric < HEADING_CHANGE_THRESHOLD) & (speeds > MIN_SPEED_MS)
    
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
            if i - start_idx >= MIN_STABLE_DURATION:
                # Calculate average heading for this stable segment
                segment_headings = smoothed_headings[start_idx:i]
                # Use circular mean for headings
                x = np.mean(np.cos(np.radians(segment_headings)))
                y = np.mean(np.sin(np.radians(segment_headings)))
                avg_heading = np.degrees(np.arctan2(y, x)) % 360
                
                segments.append((start_idx, i-1, avg_heading))
            in_segment = False
    
    # Handle case where we end in a stable segment
    if in_segment and len(df) - start_idx >= MIN_STABLE_DURATION:
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
            
            if heading_diff <= HEADING_TOLERANCE and gap_size <= MIN_SEGMENT_LENGTH:
                # Merge segments
                current_end = next_end
                # Recalculate average heading
                merged_headings = smoothed_headings[current_start:current_end+1]
                x = np.mean(np.cos(np.radians(merged_headings)))
                y = np.mean(np.sin(np.radians(merged_headings)))
                current_heading = np.degrees(np.arctan2(y, x)) % 360
            else:
                # Save current segment and start new one
                if current_end - current_start >= MIN_SEGMENT_LENGTH:
                    refined_segments.append((current_start, current_end, current_heading))
                current_start, current_end, current_heading = next_start, next_end, next_heading
        
        # Add final segment
        if current_end - current_start >= MIN_SEGMENT_LENGTH:
            refined_segments.append((current_start, current_end, current_heading))
    
    logger.info(f"Detected {len(refined_segments)} stabilized flight segments")
    for i, (start, end, heading) in enumerate(refined_segments):
        duration_min = (df['Timestamp [ms]'].iloc[end] - df['Timestamp [ms]'].iloc[start]) / 60000
        logger.info(f"  Segment {i+1}: points {start}-{end}, heading {heading:.1f}°, "
                   f"length {end-start+1}, duration {duration_min:.1f}min")
    
    return refined_segments

def combine_opposite_headings(direction_segments):
    """
    Combine opposite headings (e.g., North/South and East/West) based on configuration.
    
    Args:
        direction_segments: Dictionary of {direction_name: List[DataFrame]}
    
    Returns:
        Dictionary with combined opposite headings
    """
    logger = logging.getLogger(__name__)
    
    if not COMBINE_OPPOSITE_HEADINGS:
        return direction_segments
    
    # Extract heading values from direction names
    heading_data = {}
    for direction, seg_lines in direction_segments.items():
        try:
            heading_value = int(direction.split('_')[1])
            heading_data[heading_value] = (direction, seg_lines)
        except (IndexError, ValueError):
            logger.warning(f"Could not parse heading from direction name: {direction}")
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
            
            if diff <= OPPOSITE_HEADING_TOLERANCE and diff < best_diff:
                best_match = (heading2, dir2, lines2)
                best_diff = diff
        
        if best_match:
            heading2, dir2, lines2 = best_match
            
            # Combine the segments
            combined_lines = lines1 + lines2
            
            # Create a descriptive name
            h1_cardinal = get_cardinal_direction(heading1)
            h2_cardinal = get_cardinal_direction(heading2)
            combined_name = f"{h1_cardinal}_{h2_cardinal}_Combined_{heading1}_{heading2}"
            
            combined_segments[combined_name] = combined_lines
            used_headings.add(heading1)
            used_headings.add(heading2)
            
            total_points = sum(len(seg) for seg in combined_lines)
            logger.info(f"Combined {dir1} and {dir2} into '{combined_name}': {total_points} points, {len(combined_lines)} lines")
        else:
            # No opposite found, keep original
            combined_segments[dir1] = lines1
            used_headings.add(heading1)
    
    return combined_segments

def get_cardinal_direction(heading):
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

def group_segments_by_heading(segments, df):
    """
    Group segments by specific heading degree, keeping them as separate parallel lines.
    
    Args:
        segments: List of (start_idx, end_idx, avg_heading) tuples
        df: DataFrame with flight data
    
    Returns:
        Dictionary of {direction_name: List[DataFrame]} where each DataFrame is an independent line
    """
    logger = logging.getLogger(__name__)
    
    if not segments:
        logger.warning("No segments to group")
        return {}
    
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
            
            if diff <= HEADING_TOLERANCE * 2:  # Allow more tolerance for grouping
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
            if len(segment_df) >= MIN_POINTS_FOR_VALID_LINE:
                direction_lines.append(segment_df)
                total_points += len(segment_df)
        
        if direction_lines:
            direction_segments[direction] = direction_lines
            
            logger.info(f"Created direction '{direction}': {total_points} points, "
                       f"avg heading {avg_heading:.1f}°, {len(segment_indices)} parallel lines")
    
    return direction_segments

def save_segments(segments, output_dir, base_filename):
    """Save segmented data to a single CSV file per heading (using the exact degree of heading)"""
    logger = logging.getLogger(__name__)
    
    saved_files = []
    
    # Sort by number of lines and keep only the top 2 headings
    sorted_segments = sorted(segments.items(), key=lambda item: sum(len(seg) for seg in item[1]), reverse=True)
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
            logger.info(f"Saved {len(combined_df)} points for direction '{direction}' to {filename}")
            saved_files.append(filepath)
        except Exception as e:
            logger.error(f"Error saving segment '{direction}': {e}")
    
    return saved_files

# =============================================
# MAIN PROCESSING FUNCTIONS
# =============================================

def process_flight_data(input_csv):
    """
    Main processing function
    
    Args:
        input_csv: Path to input CSV file
    
    Returns:
        Tuple of (original_df, segmented_data_dict)
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Processing flight data from {input_csv}")
    
    # Read input data
    try:
        df = pd.read_csv(input_csv)
        logger.info(f"Loaded {len(df)} data points")
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        raise
    
    # Validate required columns
    required_cols = ['UTM_Easting', 'UTM_Northing', 'Timestamp [ms]', 'Latitude [Decimal Degrees]', 
                    'Longitude [Decimal Degrees]', 'Altitude [m]']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Detect stabilized segments (automatically removes unstable start/end portions)
    segments = detect_stabilized_segments(df)
    
    if not segments:
        logger.warning("No stabilized segments detected")
        return df, {}
    
    # Group by heading direction
    direction_segments = group_segments_by_heading(segments, df)
    
    # Combine opposite headings if configured
    final_segments = combine_opposite_headings(direction_segments)
    
    logger.info(f"Final segmentation complete: {len(final_segments)} directional segments")
    
    return df, final_segments

def create_summary_report(original_df, segments, output_dir):
    """Create a summary report of the segmentation"""
    logger = logging.getLogger(__name__)
    
    report_path = output_dir / "segmentation_summary.txt"
    
    with open(report_path, 'w') as f:
        f.write("Flight Path Segmentation Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Configuration:\n")
        f.write(f"  Heading window size: {HEADING_WINDOW_SIZE}\n")
        f.write(f"  Minimum segment length: {MIN_SEGMENT_LENGTH}\n")
        f.write(f"  Heading tolerance: {HEADING_TOLERANCE}°\n")
        f.write(f"  Stability window: {STABILITY_WINDOW}\n")
        f.write(f"  Min stable duration: {MIN_STABLE_DURATION}\n")
        f.write(f"  Heading change threshold: {HEADING_CHANGE_THRESHOLD}°\n\n")
        
        f.write("Original Data:\n")
        f.write(f"  Total points: {len(original_df)}\n")
        f.write(f"  Duration: {(original_df['Timestamp [ms]'].iloc[-1] - original_df['Timestamp [ms]'].iloc[0])/60000:.1f} minutes\n\n")
        
        f.write("Segmented Data:\n")
        total_segmented_points = sum(len(seg) for seg_list in segments.values() for seg in seg_list)
        f.write(f"  Number of directions: {len(segments)}\n")
        f.write(f"  Total segmented points: {total_segmented_points}\n")
        f.write(f"  Data retention: {total_segmented_points/len(original_df)*100:.1f}%\n\n")
        
        for direction, seg_lines in segments.items():
            for segment_df in seg_lines:
                if not segment_df.empty:
                    headings = calculate_heading(segment_df)
                    avg_heading = np.mean(headings)
                    heading_std = np.std(headings)

                    duration = (segment_df['Timestamp [ms]'].iloc[-1] - segment_df['Timestamp [ms]'].iloc[0]) / 60000

                    f.write(f"  Direction '{direction}':\n")
                    f.write(f"    Points: {len(segment_df)}\n")
                    f.write(f"    Duration: {duration:.1f} minutes\n")
                    f.write(f"    Average heading: {avg_heading:.1f}° ± {heading_std:.1f}°\n")
                    f.write(f"    UTM extent: E {segment_df['UTM_Easting'].min():.0f}-{segment_df['UTM_Easting'].max():.0f}, "
                           f"N {segment_df['UTM_Northing'].min():.0f}-{segment_df['UTM_Northing'].max():.0f}\n\n")
    
    logger.info(f"Summary report saved to {report_path}")

def main():
    """Main execution function"""
    logger = setup_logging()
    logger.info("Starting Flight Path Segmentation")
    
    # Validate input file
    input_path = Path(INPUT_CSV)
    if not input_path.exists():
        logger.error(f"Input file not found: {INPUT_CSV}")
        return
    
    # Create output directory
    base_filename = input_path.stem
    output_dir = input_path.parent / f"{base_filename}_segmented"
    output_dir.mkdir(exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    try:
        # Process flight data
        original_df, segments = process_flight_data(INPUT_CSV)
        
        if not segments:
            logger.error("No segments were created. Check configuration parameters.")
            return
        
        # Create visualizations
        create_overview_plots(original_df, output_dir)
        plot_segmentation_analysis(original_df, segments, output_dir)
        plot_utm_verification(segments, output_dir)

        # Save segmented data
        saved_files = save_segments(segments, output_dir, base_filename)
        
        # Create summary report
        create_summary_report(original_df, segments, output_dir)
        
        # Final summary
        logger.info("Flight path segmentation completed successfully!")
        logger.info(f"Created {len(segments)} directional segments:")
        for direction, segment_df in segments.items():
            logger.info(f"  - {direction}: {len(segment_df)} points")
        
        logger.info(f"All outputs saved to: {output_dir}")
        logger.info(f"CSV files: {len(saved_files)} created")
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        raise

if __name__ == "__main__":
    main()