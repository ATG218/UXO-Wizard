import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime

def create_concentration_map(csv_file):
    """
    Create an interactive map visualization of concentration data
    with toggleable measurements.
    """
    
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Convert timestamp to datetime for better handling
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Define the measurement columns we want to visualize
    measurement_columns = ['Total', 'Countrate', 'U238', 'K40', 'Th232', 'Cs137', 'Height', 'Press', 'Temp', 'Hum']
    
    # Create color scales for different measurements
    color_scales = {
        'Total': 'Viridis',
        'Countrate': 'Plasma',
        'U238': 'Inferno',
        'K40': 'Magma',
        'Th232': 'Cividis',
        'Cs137': 'Blues',
        'Height': 'Greens',
        'Press': 'Oranges',
        'Temp': 'Reds',
        'Hum': 'Purples'
    }
    
    # Create the main figure
    fig = go.Figure()
    
    # Add traces for each measurement (initially all but the first one will be hidden)
    for i, measurement in enumerate(measurement_columns):
        # Handle potential NaN values
        valid_data = df.dropna(subset=[measurement, 'lat', 'lon'])
        
        if len(valid_data) == 0:
            continue
            
        # Create hover text with all relevant information
        hover_text = []
        for idx, row in valid_data.iterrows():
            hover_info = f"<b>{measurement}: {row[measurement]:.2f}</b><br>"
            hover_info += f"Lat: {row['lat']:.6f}<br>"
            hover_info += f"Lon: {row['lon']:.6f}<br>"
            hover_info += f"Time: {row['datetime'].strftime('%Y-%m-%d %H:%M:%S')}<br>"
            hover_info += f"<extra></extra>"  # Remove the trace name box
            hover_text.append(hover_info)
        
        # Add scatter map trace (updated from deprecated scattermapbox)
        fig.add_trace(
            go.Scattermap(
                lat=valid_data['lat'],
                lon=valid_data['lon'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=valid_data[measurement],
                    colorscale=color_scales.get(measurement, 'Viridis'),
                    colorbar=dict(
                        title=f"{measurement}",
                        x=1.02,
                        len=0.7
                    ),
                    opacity=0.8,
                    showscale=True
                ),
                text=hover_text,
                hovertemplate='%{text}',
                name=measurement,
                visible=True if i == 0 else False  # Only show the first measurement initially
            )
        )
    
    # Calculate map center and zoom level based on data extent
    lat_center = df['lat'].mean()
    lon_center = df['lon'].mean()
    lat_range = df['lat'].max() - df['lat'].min()
    lon_range = df['lon'].max() - df['lon'].min()
    
    # Estimate zoom level based on data spread
    max_range = max(lat_range, lon_range)
    if max_range > 1:
        zoom = 8
    elif max_range > 0.1:
        zoom = 10
    elif max_range > 0.01:
        zoom = 12
    else:
        zoom = 14
    
    # Update layout (updated for scattermap)
    fig.update_layout(
        title={
            'text': 'Concentration Data Visualization Map',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        map=dict(
            style="satellite",  # You can change to "satellite", "white-bg", etc.
            center=dict(lat=lat_center, lon=lon_center),
            zoom=zoom
        ),
        height=700,
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    # Create dropdown menu for selecting measurements
    dropdown_buttons = []
    for i, measurement in enumerate(measurement_columns):
        # Create visibility list - True for selected measurement, False for others
        visibility = [False] * len(measurement_columns)
        visibility[i] = True
        
        dropdown_buttons.append(
            dict(
                label=measurement,
                method="update",
                args=[
                    {"visible": visibility},
                    {"title": f"Concentration Data Visualization Map - {measurement}"}
                ]
            )
        )
    
    # Add dropdown menu
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=dropdown_buttons,
                direction="down",
                showactive=True,
                x=0.02,
                xanchor="left",
                y=0.98,
                yanchor="top",
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1
            )
        ]
    )
    
    # Add annotation to explain the dropdown
    fig.add_annotation(
        text="Select Measurement:",
        x=0.02,
        y=1.02,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=12, color="black")
    )
    
    return fig

def display_data_summary(csv_file):
    """
    Display a summary of the data
    """
    df = pd.read_csv(csv_file)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    print("=== Data Summary ===")
    print(f"Total data points: {len(df)}")
    print(f"Time range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"Latitude range: {df['lat'].min():.6f} to {df['lat'].max():.6f}")
    print(f"Longitude range: {df['lon'].min():.6f} to {df['lon'].max():.6f}")
    
    print("\n=== Measurement Statistics ===")
    measurement_columns = ['Total', 'Countrate', 'U238', 'K40', 'Th232', 'Cs137', 'Height', 'Press', 'Temp', 'Hum']
    
    for col in measurement_columns:
        if col in df.columns:
            print(f"{col:10}: Min={df[col].min():8.2f}, Max={df[col].max():8.2f}, Mean={df[col].mean():8.2f}, Std={df[col].std():8.2f}")

if __name__ == "__main__":
    # File path to your CSV
    csv_file = "combined_drone_data_clean.csv"
    
    try:
        # Display data summary
        display_data_summary(csv_file)
        
        # Create the interactive map
        print("\nCreating interactive map...")
        fig = create_concentration_map(csv_file)
        
        # Show the map
        fig.show()
        
        # Optionally save as HTML
        fig.write_html("concentration_map.html")
        print("Map saved as 'concentration_map.html'")
        
    except FileNotFoundError:
        print(f"Error: Could not find the file '{csv_file}'")
        print("Please make sure the CSV file is in the same directory as this script.")
    except Exception as e:
        print(f"An error occurred: {e}") 