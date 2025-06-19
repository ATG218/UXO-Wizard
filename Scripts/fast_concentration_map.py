import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Mapbox API token
MAPBOX_TOKEN = "pk.eyJ1Ijoia3Jpc3AxNDAiLCJhIjoiY21jMGE0Zm5jMDA0bzJrczBwbnZ5czdkNCJ9.hZMQoAmmdeNWwB2fuu80Tg"
px.set_mapbox_access_token(MAPBOX_TOKEN)

def create_fast_concentration_map(csv_file):
    """
    Create a fast-loading concentration map with minimal traces
    """
    # Read data
    df = pd.read_csv(csv_file)
    
    # Convert timestamp if available
    if 'timestamp' in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Define available measurements
    measurement_columns = ['Total', 'Countrate', 'U238', 'K40', 'Th232', 'Cs137', 'Height', 'Press', 'Temp', 'Hum']
    available_columns = [col for col in measurement_columns if col in df.columns]
    
    # Create figure
    fig = go.Figure()
    
    # Add traces for each measurement (just points, no interpolation)
    for i, measurement in enumerate(available_columns):
        valid_data = df.dropna(subset=[measurement, 'lat', 'lon'])
        
        if len(valid_data) == 0:
            continue
        
        # Create hover text
        hover_text = []
        for idx, row in valid_data.iterrows():
            hover_info = f"<b>{measurement}: {row[measurement]:.2f}</b><br>"
            hover_info += f"Lat: {row['lat']:.6f}<br>"
            hover_info += f"Lon: {row['lon']:.6f}<br>"
            if 'datetime' in row:
                hover_info += f"Time: {row['datetime'].strftime('%Y-%m-%d %H:%M:%S')}<br>"
            hover_info += f"<extra></extra>"
            hover_text.append(hover_info)
        
        # Add scatter points with color scale
        fig.add_trace(
            go.Scattermapbox(
                lat=valid_data['lat'],
                lon=valid_data['lon'],
                mode='markers',
                marker=dict(
                    size=10,
                    color=valid_data[measurement],
                    colorscale='Viridis',
                    colorbar=dict(title=measurement),
                    showscale=True,
                    opacity=0.8
                ),
                text=hover_text,
                hovertemplate='%{text}',
                name=measurement,
                visible=True if i == 0 else False,
            )
        )
    
    # Calculate center and zoom
    lat_center = df['lat'].mean()
    lon_center = df['lon'].mean()
    lat_range = df['lat'].max() - df['lat'].min()
    lon_range = df['lon'].max() - df['lon'].min()
    
    max_range = max(lat_range, lon_range)
    if max_range > 1:
        zoom = 9
    elif max_range > 0.1:
        zoom = 11
    elif max_range > 0.01:
        zoom = 13
    else:
        zoom = 15
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Fast Concentration Map - Points Only',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        mapbox=dict(
            accesstoken=MAPBOX_TOKEN,
            style="satellite",
            center=dict(lat=lat_center, lon=lon_center),
            zoom=zoom
        ),
        height=700,
        margin=dict(l=0, r=0, t=60, b=0)
    )
    
    # Create simple dropdown
    dropdown_buttons = []
    for i, measurement in enumerate(available_columns):
        visibility = [False] * len(available_columns)
        visibility[i] = True
        
        dropdown_buttons.append(
            dict(
                label=measurement,
                method="update",
                args=[
                    {"visible": visibility},
                    {"title": f"Fast Concentration Map - {measurement}"}
                ]
            )
        )
    
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
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1
            )
        ]
    )
    
    return fig

if __name__ == "__main__":
    csv_file = "tiller/high_alt/concentrations_tiller_high.csv"
    
    try:
        print("Creating fast concentration map...")
        fig = create_fast_concentration_map(csv_file)
        
        output_file = "fast_concentration_map.html"
        fig.write_html(
            output_file,
            config={"mapboxAccessToken": MAPBOX_TOKEN},
            include_plotlyjs="cdn"
        )
        print(f"✅ Saved: {output_file}")
        print("This map should load much faster - it's just colored points with no interpolation!")
        
    except Exception as e:
        print(f"❌ Error: {e}") 