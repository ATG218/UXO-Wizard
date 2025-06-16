import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def load_and_analyze_data(csv_file):
    """Load and perform initial analysis of the flight data"""
    print("Loading and analyzing flight data...")
    
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Convert timestamp to datetime
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Calculate time differences (speed indicators)
    df['time_diff'] = df['datetime'].diff().dt.total_seconds()
    
    # Calculate distance between consecutive points (rough speed estimate)
    df['lat_diff'] = df['lat'].diff()
    df['lon_diff'] = df['lon'].diff()
    df['distance'] = np.sqrt(df['lat_diff']**2 + df['lon_diff']**2)
    df['speed'] = df['distance'] / df['time_diff']  # degrees per second
    
    # Calculate direction changes (turning patterns)
    df['bearing'] = np.arctan2(df['lon_diff'], df['lat_diff']) * 180 / np.pi
    df['bearing_change'] = df['bearing'].diff().abs()
    df['bearing_change'] = np.where(df['bearing_change'] > 180, 360 - df['bearing_change'], df['bearing_change'])
    
    # Calculate altitude changes
    df['height_diff'] = df['Height'].diff().abs()
    
    print(f"Total data points: {len(df)}")
    print(f"Time range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"Duration: {(df['datetime'].max() - df['datetime'].min()).total_seconds()/3600:.2f} hours")
    print(f"Lat range: {df['lat'].min():.6f} to {df['lat'].max():.6f}")
    print(f"Lon range: {df['lon'].min():.6f} to {df['lon'].max():.6f}")
    print(f"Height range: {df['Height'].min():.1f}m to {df['Height'].max():.1f}m")
    
    return df

def analyze_flight_patterns(df):
    """Analyze flight patterns to identify different flight phases"""
    print("\nAnalyzing flight patterns...")
    
    # Remove rows with NaN values for analysis
    analysis_df = df.dropna(subset=['speed', 'bearing_change', 'height_diff'])
    
    if len(analysis_df) == 0:
        print("No valid data for analysis")
        return df
    
    # Statistical analysis
    speed_stats = analysis_df['speed'].describe()
    height_stats = analysis_df['Height'].describe()
    bearing_stats = analysis_df['bearing_change'].describe()
    
    print(f"\nSpeed statistics (degrees/second):")
    print(f"  Mean: {speed_stats['mean']:.6f}")
    print(f"  Std:  {speed_stats['std']:.6f}")
    print(f"  95th percentile: {analysis_df['speed'].quantile(0.95):.6f}")
    
    print(f"\nAltitude statistics (meters):")
    print(f"  Mean: {height_stats['mean']:.1f}")
    print(f"  Std:  {height_stats['std']:.1f}")
    
    print(f"\nBearing change statistics (degrees):")
    print(f"  Mean: {bearing_stats['mean']:.1f}")
    print(f"  Std:  {bearing_stats['std']:.1f}")
    
    return analysis_df

def create_analysis_plots(df):
    """Create plots to visualize flight patterns"""
    print("Creating analysis plots...")
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            'Flight Path (Lat vs Lon)', 'Altitude vs Time',
            'Speed vs Time', 'Bearing Changes vs Time',
            'Speed Distribution', 'Altitude Distribution'
        ],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Flight path
    fig.add_trace(
        go.Scatter(x=df['lon'], y=df['lat'], mode='markers+lines', 
                  marker=dict(size=3, color=df.index, colorscale='Viridis'),
                  name='Flight Path'),
        row=1, col=1
    )
    
    # Altitude vs time
    fig.add_trace(
        go.Scatter(x=df['datetime'], y=df['Height'], mode='lines',
                  name='Altitude'),
        row=1, col=2
    )
    
    # Speed vs time (remove extreme outliers for visualization)
    speed_clean = df['speed'].clip(upper=df['speed'].quantile(0.95))
    fig.add_trace(
        go.Scatter(x=df['datetime'], y=speed_clean, mode='lines',
                  name='Speed'),
        row=2, col=1
    )
    
    # Bearing changes vs time
    fig.add_trace(
        go.Scatter(x=df['datetime'], y=df['bearing_change'], mode='lines',
                  name='Bearing Change'),
        row=2, col=2
    )
    
    # Speed distribution
    fig.add_trace(
        go.Histogram(x=speed_clean, nbinsx=50, name='Speed Distribution'),
        row=3, col=1
    )
    
    # Altitude distribution
    fig.add_trace(
        go.Histogram(x=df['Height'], nbinsx=50, name='Altitude Distribution'),
        row=3, col=2
    )
    
    fig.update_layout(height=1200, showlegend=False, 
                     title_text="Flight Pattern Analysis")
    fig.write_html("flight_analysis.html")
    print("Flight analysis saved as 'flight_analysis.html'")

def filter_grid_data_method1_altitude(df, altitude_threshold_std=1.5):
    """Method 1: Filter based on altitude consistency (grid flights at consistent altitude)"""
    print(f"\nMethod 1: Filtering based on altitude consistency...")
    
    altitude_mean = df['Height'].mean()
    altitude_std = df['Height'].std()
    
    # Keep data within certain standard deviations of mean altitude
    altitude_lower = altitude_mean - altitude_threshold_std * altitude_std
    altitude_upper = altitude_mean + altitude_threshold_std * altitude_std
    
    grid_data = df[(df['Height'] >= altitude_lower) & (df['Height'] <= altitude_upper)]
    
    print(f"  Altitude range for grid data: {altitude_lower:.1f}m to {altitude_upper:.1f}m")
    print(f"  Original points: {len(df)}")
    print(f"  Grid points: {len(grid_data)} ({len(grid_data)/len(df)*100:.1f}%)")
    
    return grid_data

def filter_grid_data_method2_speed(df, speed_percentile=85):
    """Method 2: Filter based on speed (remove high-speed transit portions)"""
    print(f"\nMethod 2: Filtering based on speed...")
    
    # Remove NaN values first
    speed_clean = df.dropna(subset=['speed'])
    
    if len(speed_clean) == 0:
        print("  No speed data available")
        return df
    
    speed_threshold = speed_clean['speed'].quantile(speed_percentile/100)
    grid_data = df[df['speed'] <= speed_threshold]
    
    print(f"  Speed threshold: {speed_threshold:.6f} degrees/second")
    print(f"  Original points: {len(df)}")
    print(f"  Grid points: {len(grid_data)} ({len(grid_data)/len(df)*100:.1f}%)")
    
    return grid_data

def filter_grid_data_method3_spatial_clustering(df, eps=0.001, min_samples=50):
    """Method 3: Use spatial clustering to identify the main survey area"""
    print(f"\nMethod 3: Filtering using spatial clustering...")
    
    # Prepare data for clustering
    coords = df[['lat', 'lon']].values
    
    # Standardize coordinates
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)
    
    # Apply DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords_scaled)
    
    # Add cluster labels to dataframe
    df_clustered = df.copy()
    df_clustered['cluster'] = clustering.labels_
    
    # Find the largest cluster (assumed to be the main survey area)
    cluster_counts = pd.Series(clustering.labels_).value_counts()
    largest_cluster = cluster_counts.index[0] if cluster_counts.index[0] != -1 else cluster_counts.index[1]
    
    grid_data = df_clustered[df_clustered['cluster'] == largest_cluster]
    
    print(f"  Found {len(cluster_counts)} clusters")
    print(f"  Largest cluster: {largest_cluster} with {len(grid_data)} points")
    print(f"  Original points: {len(df)}")
    print(f"  Grid points: {len(grid_data)} ({len(grid_data)/len(df)*100:.1f}%)")
    
    return grid_data

def filter_grid_data_method4_combined(df):
    """Method 4: Combined approach using multiple criteria"""
    print(f"\nMethod 4: Combined filtering approach...")
    
    # Start with full dataset
    filtered_data = df.copy()
    original_count = len(filtered_data)
    
    # Step 1: Remove extreme speeds (top 10%)
    if 'speed' in filtered_data.columns:
        speed_threshold = filtered_data['speed'].quantile(0.9)
        filtered_data = filtered_data[filtered_data['speed'] <= speed_threshold]
        print(f"  After speed filter: {len(filtered_data)} points ({len(filtered_data)/original_count*100:.1f}%)")
    
    # Step 2: Keep altitude within 1 standard deviation
    altitude_mean = filtered_data['Height'].mean()
    altitude_std = filtered_data['Height'].std()
    altitude_lower = altitude_mean - 1.0 * altitude_std
    altitude_upper = altitude_mean + 1.0 * altitude_std
    filtered_data = filtered_data[(filtered_data['Height'] >= altitude_lower) & 
                                 (filtered_data['Height'] <= altitude_upper)]
    print(f"  After altitude filter: {len(filtered_data)} points ({len(filtered_data)/original_count*100:.1f}%)")
    
    # Step 3: Spatial clustering to keep main survey area
    if len(filtered_data) > 100:  # Only if we have enough points
        coords = filtered_data[['lat', 'lon']].values
        scaler = StandardScaler()
        coords_scaled = scaler.fit_transform(coords)
        
        clustering = DBSCAN(eps=0.001, min_samples=30).fit(coords_scaled)
        filtered_data['cluster'] = clustering.labels_
        
        # Keep largest cluster
        cluster_counts = pd.Series(clustering.labels_).value_counts()
        if len(cluster_counts) > 0:
            largest_cluster = cluster_counts.index[0] if cluster_counts.index[0] != -1 else (cluster_counts.index[1] if len(cluster_counts) > 1 else -1)
            if largest_cluster != -1:
                filtered_data = filtered_data[filtered_data['cluster'] == largest_cluster]
                print(f"  After spatial clustering: {len(filtered_data)} points ({len(filtered_data)/original_count*100:.1f}%)")
    
    return filtered_data

def create_comparison_map(original_df, filtered_df, method_name):
    """Create a comparison map showing original vs filtered data"""
    print(f"Creating comparison map for {method_name}...")
    
    fig = go.Figure()
    
    # Original data
    fig.add_trace(go.Scattermap(
        lat=original_df['lat'],
        lon=original_df['lon'],
        mode='markers',
        marker=dict(size=4, color='red', opacity=0.5),
        name='Original Data (All)',
        text=original_df['datetime'].dt.strftime('%H:%M:%S')
    ))
    
    # Filtered data
    fig.add_trace(go.Scattermap(
        lat=filtered_df['lat'],
        lon=filtered_df['lon'],
        mode='markers',
        marker=dict(size=6, color='blue', opacity=0.8),
        name=f'Grid Data ({method_name})',
        text=filtered_df['datetime'].dt.strftime('%H:%M:%S')
    ))
    
    # Calculate center and zoom
    lat_center = (original_df['lat'].min() + original_df['lat'].max()) / 2
    lon_center = (original_df['lon'].min() + original_df['lon'].max()) / 2
    
    lat_range = original_df['lat'].max() - original_df['lat'].min()
    lon_range = original_df['lon'].max() - original_df['lon'].min()
    max_range = max(lat_range, lon_range)
    
    if max_range > 0.01:
        zoom = 12
    elif max_range > 0.001:
        zoom = 14
    else:
        zoom = 16
    
    fig.update_layout(
        title=f'Flight Data Filtering - {method_name}',
        map=dict(
            style="carto-positron",
            center=dict(lat=lat_center, lon=lon_center),
            zoom=zoom
        ),
        height=600
    )
    
    filename = f"filtered_comparison_{method_name.lower().replace(' ', '_')}.html"
    fig.write_html(filename)
    print(f"Comparison map saved as '{filename}'")

def main():
    # Load and analyze the data
    df = load_and_analyze_data('concentrations(in).csv')
    
    # Perform analysis
    analysis_df = analyze_flight_patterns(df)
    
    # Create analysis plots
    create_analysis_plots(analysis_df)
    
    # Try different filtering methods
    print("\n" + "="*60)
    print("TESTING DIFFERENT FILTERING METHODS")
    print("="*60)
    
    # Method 1: Altitude-based filtering
    grid_data_alt = filter_grid_data_method1_altitude(df)
    create_comparison_map(df, grid_data_alt, "Altitude Based")
    
    # Method 2: Speed-based filtering
    grid_data_speed = filter_grid_data_method2_speed(df)
    create_comparison_map(df, grid_data_speed, "Speed Based")
    
    # Method 3: Spatial clustering
    grid_data_spatial = filter_grid_data_method3_spatial_clustering(df)
    create_comparison_map(df, grid_data_spatial, "Spatial Clustering")
    
    # Method 4: Combined approach
    grid_data_combined = filter_grid_data_method4_combined(df)
    create_comparison_map(df, grid_data_combined, "Combined Method")
    
    print("\n" + "="*60)
    print("SUMMARY OF FILTERING RESULTS")
    print("="*60)
    print(f"Original data points: {len(df)}")
    print(f"Altitude-based filter: {len(grid_data_alt)} points ({len(grid_data_alt)/len(df)*100:.1f}%)")
    print(f"Speed-based filter: {len(grid_data_speed)} points ({len(grid_data_speed)/len(df)*100:.1f}%)")
    print(f"Spatial clustering: {len(grid_data_spatial)} points ({len(grid_data_spatial)/len(df)*100:.1f}%)")
    print(f"Combined method: {len(grid_data_combined)} points ({len(grid_data_combined)/len(df)*100:.1f}%)")
    
    # Save the best filtered dataset
    print(f"\nSaving filtered datasets...")
    grid_data_combined.to_csv('concentrations_grid_filtered.csv', index=False)
    print("Filtered data saved as 'concentrations_grid_filtered.csv'")
    
    # Update the original visualization script to use filtered data
    print("\nRecommendation:")
    print("- Review the comparison maps to see which filtering method works best")
    print("- The 'Combined Method' usually provides the best balance")
    print("- Use 'concentrations_grid_filtered.csv' for your final visualization")

if __name__ == "__main__":
    main()