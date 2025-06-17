import pandas as pd
import plotly.graph_objects as go
import numpy as np

def quick_flight_path_view(csv_file='concentrations_tarva_clean.csv'):
    """Create a quick visualization of the flight path to help identify cut points."""
    
    # Load data
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} data points")
    
    # Sort by timestamp to get chronological order
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp').reset_index(drop=True)
        print("Data sorted chronologically by timestamp")
    else:
        print("Warning: No timestamp column found, using original order")
    
    # Add sequence numbers (now in chronological order)
    df['sequence'] = range(len(df))
    
    # Create the plot
    fig = go.Figure()
    
    # Plot the full flight path with sequence coloring
    fig.add_trace(go.Scatter(
        x=df['lon'],
        y=df['lat'],
        mode='markers+lines',
        marker=dict(
            size=4,
            color=df['sequence'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Point Number")
        ),
        line=dict(width=2, color='rgba(0,0,255,0.3)'),
        text=[f"Point {i}<br>Lat: {lat:.6f}<br>Lon: {lon:.6f}<br>Timestamp: {timestamp}" 
              for i, (lat, lon, timestamp) in enumerate(zip(df['lat'], df['lon'], df['timestamp']))],
        hovertemplate='%{text}<extra></extra>',
        name='Flight Path'
    ))
    
    # Add start marker
    fig.add_trace(go.Scatter(
        x=[df['lon'].iloc[0]],
        y=[df['lat'].iloc[0]],
        mode='markers+text',
        marker=dict(size=20, color='green', symbol='star'),
        text=['START'],
        textposition='top center',
        textfont=dict(size=14, color='green'),
        name='Start',
        hovertemplate='Start Point (Index 0)<extra></extra>'
    ))
    
    # Add end marker
    fig.add_trace(go.Scatter(
        x=[df['lon'].iloc[-1]],
        y=[df['lat'].iloc[-1]],
        mode='markers+text',
        marker=dict(size=20, color='red', symbol='star'),
        text=['END'],
        textposition='top center',
        textfont=dict(size=14, color='red'),
        name='End',
        hovertemplate=f'End Point (Index {len(df)-1})<extra></extra>'
    ))
    
    # Add some intermediate markers to help with navigation
    quarter_points = [len(df)//4, len(df)//2, 3*len(df)//4]
    colors = ['orange', 'purple', 'brown']
    
    for i, (point, color) in enumerate(zip(quarter_points, colors)):
        fig.add_trace(go.Scatter(
            x=[df['lon'].iloc[point]],
            y=[df['lat'].iloc[point]],
            mode='markers+text',
            marker=dict(size=12, color=color, symbol='diamond'),
            text=[f'{point}'],
            textposition='top center',
            textfont=dict(size=10, color=color),
            name=f'Point {point}',
            hovertemplate=f'Reference Point {point}<extra></extra>'
        ))
    
    fig.update_layout(
        title=f'Drone Flight Path Analysis<br><sub>Total Points: {len(df)} | Hover for details | Color shows sequence</sub>',
        xaxis_title='Longitude',
        yaxis_title='Latitude',
        hovermode='closest',
        width=1200,
        height=800,
        showlegend=True
    )
    
    # Print some statistics
    print(f"\nFlight Path Statistics (chronological order):")
    print(f"Start: Point 0 at ({df['lat'].iloc[0]:.6f}, {df['lon'].iloc[0]:.6f}) - Timestamp: {df['timestamp'].iloc[0]}")
    print(f"End: Point {len(df)-1} at ({df['lat'].iloc[-1]:.6f}, {df['lon'].iloc[-1]:.6f}) - Timestamp: {df['timestamp'].iloc[-1]}")
    print(f"Quarter points: {quarter_points}")
    if 'timestamp' in df.columns:
        duration_ms = df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]
        print(f"Flight duration: {duration_ms/1000:.1f} seconds ({duration_ms/60000:.1f} minutes)")
    
    # Calculate some basic metrics to help identify patterns
    center_lat = df['lat'].mean()
    center_lon = df['lon'].mean()
    df['distance_from_center'] = np.sqrt((df['lat'] - center_lat)**2 + (df['lon'] - center_lon)**2)
    
    # Find points that are far from center (likely departure/return)
    far_threshold = df['distance_from_center'].quantile(0.9)
    far_points = df[df['distance_from_center'] > far_threshold]
    
    print(f"\nSuggested analysis:")
    print(f"Points far from flight center (>90th percentile): {len(far_points)} points")
    if len(far_points) > 0:
        print(f"First far point: {far_points.index[0]}")
        print(f"Last far point: {far_points.index[-1]}")
    
    return fig, df

def filter_and_save(df, start_idx, end_idx, output_file):
    """Filter the data and save to a new file."""
    filtered_df = df.iloc[start_idx:end_idx+1].copy()
    
    # Remove helper columns if they exist
    cols_to_remove = ['sequence', 'distance_from_center']
    for col in cols_to_remove:
        if col in filtered_df.columns:
            filtered_df = filtered_df.drop(columns=[col])
    
    filtered_df.to_csv(output_file, index=False)
    
    print(f"\nFiltering Results:")
    print(f"Original data points: {len(df)}")
    print(f"Filtered data points: {len(filtered_df)}")
    print(f"Removed from beginning: {start_idx} points")
    print(f"Removed from end: {len(df) - end_idx - 1} points")
    print(f"Total removed: {len(df) - len(filtered_df)} points")
    print(f"Filtered data saved to: {output_file}")

if __name__ == "__main__":
    print("Quick Flight Path Viewer")
    print("=" * 30)
    
    # Show the flight path
    fig, df = quick_flight_path_view()
    
    # Save to HTML file instead of showing directly
    html_filename = "flight_path_visualization_tarva.html"
    fig.write_html(html_filename)
    print(f"\nVisualization saved to: {html_filename}")
    print("Open this file in your web browser to view the interactive map.")
    
    print("\nInstructions for filtering:")
    print("1. Open flight_path_visualization.html in your web browser")
    print("2. Identify where the main grid pattern starts and ends")
    print("3. Note the point numbers (shown in hover text and color scale)")
    print("4. The departure and return portions are typically at the beginning and end")
    print("5. You can run the filtering interactively below, or use the filter_and_save function")
    
    # Interactive filtering option
    while True:
        try:
            choice = input(f"\nWould you like to filter the data now? (y/n): ").lower()
            if choice != 'y':
                print("You can use the filter_and_save(df, start_idx, end_idx, 'output.csv') function later.")
                break
                
            print(f"Current data range: 0 to {len(df)-1}")
            start_idx = int(input("Enter start index (where grid begins): "))
            end_idx = int(input("Enter end index (where grid ends): "))
            
            if start_idx < 0 or end_idx >= len(df) or start_idx > end_idx:
                print("Invalid indices. Please try again.")
                continue
                
            output_file = input("Enter output filename: ")
            if not output_file.endswith('.csv'):
                output_file += '.csv'
            
            filter_and_save(df, start_idx, end_idx, output_file)
            break
            
        except ValueError:
            print("Please enter valid integers.")
        except KeyboardInterrupt:
            print("\nCancelled by user.")
            break