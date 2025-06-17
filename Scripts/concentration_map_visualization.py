import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime

# Mapbox API token - replace with your own token
MAPBOX_TOKEN = "pk.eyJ1Ijoia3Jpc3AxNDAiLCJhIjoiY21jMGE0Zm5jMDA0bzJrczBwbnZ5czdkNCJ9.hZMQoAmmdeNWwB2fuu80Tg"

# Set the token globally for plotly
px.set_mapbox_access_token(MAPBOX_TOKEN)

def create_concentration_map(csv_file, map_style="satellite"):
    """
    Create an interactive map visualization of concentration data
    with toggleable measurements.
    
    Args:
        csv_file: Path to the CSV file
        map_style: Map style to use. Options:
                  - "satellite" (satellite imagery)
                  - "satellite-streets" (hybrid satellite + streets)
                  - "outdoors" (topographic style)
                  - "light" (light colored streets)
                  - "dark" (dark theme)
                  - "open-street-map" (basic, no API key needed)
    """
    
    print(f"✅ Using '{map_style}' map style with Mapbox API")
    
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
                visible=True if i == 0 else False,  # Only show the first measurement initially
            )
        )
    
    # Calculate map center and zoom level based on data extent
    lat_center = df['lat'].mean()
    lon_center = df['lon'].mean()
    lat_range = df['lat'].max() - df['lat'].min()
    lon_range = df['lon'].max() - df['lon'].min()
    
    # Estimate zoom level based on data spread (more aggressive for better detail)
    max_range = max(lat_range, lon_range)
    if max_range > 1:
        zoom = 9
    elif max_range > 0.1:
        zoom = 11
    elif max_range > 0.01:
        zoom = 13
    elif max_range > 0.001:
        zoom = 15
    else:
        zoom = 17
    
    # Update layout with better map configuration
    fig.update_layout(
        title={
            'text': f'Concentration Data Visualization Map ({map_style})',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        map=dict(
            style="white-bg",      # neutral base
            center=dict(lat=lat_center, lon=lon_center),
            zoom=zoom,
            layers=[{
                "sourcetype": "raster",
                "source": [
                    # Mapbox Satellite (needs token in the URL):
                    f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/tiles/{{z}}/{{x}}/{{y}}?access_token={MAPBOX_TOKEN}"
                ],
                "below": "traces",
                "minzoom": 0,
                "maxzoom": 22
            }]
        ),
        height=700,
        margin=dict(l=0, r=0, t=60, b=0)
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
                    {"title": f"Concentration Data Visualization Map - {measurement} ({map_style})"}
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
    csv_file = "tarva/concentrations_tarva_clean.csv"
    
    try:
        # Display data summary
        display_data_summary(csv_file)
        
        # Create the interactive map
        print("\nCreating interactive map...")
        
        # Choose your preferred map style:
        map_style = "satellite"  # or "satellite-streets", "outdoors", "light", "dark", "open-street-map"
        
        fig = create_concentration_map(csv_file, map_style=map_style)
        print(f"✅ Map created successfully")
        
        # Save as HTML with token config
        output_file = "concentration_map_tarva_clean.html"
        print(f"💾 Saving map to '{output_file}'...")
        fig.write_html(output_file, 
                      config={"mapboxAccessToken": MAPBOX_TOKEN},
                      include_plotlyjs="cdn")
        print(f"✅ Map saved successfully as '{output_file}'")
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find the file '{csv_file}'")
        print("Please make sure the CSV file is in the same directory as this script.")
    except Exception as e:
        print(f"❌ An error occurred: {e}") 