import pandas as pd
import folium
from folium.plugins import HeatMap
import numpy as np
from datetime import datetime
import os
from scipy.interpolate import griddata
from scipy.spatial.distance import pdist, squareform
import base64
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pykrige.ok import OrdinaryKriging

def create_interpolated_overlay(valid_data, measurement, map_bounds=None):
    """
    Create an interpolated field overlay based on actual measurement values
    using statistical interpolation, not density-based heatmaps.
    
    Args:
        valid_data: DataFrame with lat, lon, and measurement columns
        measurement: Name of the measurement column
        map_bounds: Map bounds for interpolation grid
    
    Returns:
        Folium raster overlay or None if interpolation fails
    """
    try:
        # Extract coordinates and values
        lats = valid_data['lat'].values
        lons = valid_data['lon'].values
        values = valid_data[measurement].values
        
        # Calculate statistics for value-based scaling
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        # Create interpolation grid
        lat_min, lat_max = lats.min(), lats.max()
        lon_min, lon_max = lons.min(), lons.max()
        
        # Add some padding to the grid
        lat_padding = (lat_max - lat_min) * 0.1
        lon_padding = (lon_max - lon_min) * 0.1
        
        lat_min -= lat_padding
        lat_max += lat_padding
        lon_min -= lon_padding
        lon_max += lon_padding
        
        # Create points array for distance calculations
        points = np.column_stack((lats, lons))
        
        # Calculate appropriate grid resolution based on spatial sampling
        # Find minimum distance between points to respect Nyquist criterion
        min_distances = []
        for i in range(min(len(valid_data), 100)):  # Sample subset for performance
            point = points[i]
            other_points = points[np.arange(len(points)) != i]
            if len(other_points) > 0:
                distances = np.linalg.norm(other_points - point, axis=1)
                min_distances.append(np.min(distances))
        
        if min_distances:
            avg_min_distance = np.mean(min_distances)
            # Grid spacing should be at least 1/4 of average minimum distance between points
            # to avoid constructing noise (respecting spatial Nyquist criterion)
            min_grid_spacing = avg_min_distance / 4.0
            
            # Calculate grid resolution based on area and minimum spacing
            lat_span = lat_max - lat_min
            lon_span = lon_max - lon_min
            
            lat_resolution = max(int(lat_span / min_grid_spacing), 20)
            lon_resolution = max(int(lon_span / min_grid_spacing), 20)
            
            # Use the larger dimension for a square grid, but cap for performance
            grid_resolution = min(max(lat_resolution, lon_resolution), 150)
            
            print(f"   📏 Avg minimum distance between points: {avg_min_distance:.6f}°")
            print(f"   🎯 Minimum grid spacing (1/4 distance): {min_grid_spacing:.6f}°")
            print(f"   📊 Calculated grid resolution: {grid_resolution}x{grid_resolution}")
        else:
            # Fallback to point-density based resolution
            n_points = len(valid_data)
            if n_points > 1000:
                grid_resolution = 120
            elif n_points > 500:
                grid_resolution = 100
            elif n_points > 100:
                grid_resolution = 80
            else:
                grid_resolution = 60
            print(f"   📊 Fallback grid resolution based on {n_points} points: {grid_resolution}x{grid_resolution}")
        
        lat_grid = np.linspace(lat_min, lat_max, grid_resolution)
        lon_grid = np.linspace(lon_min, lon_max, grid_resolution)
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
        
        # Use Ordinary Kriging for geostatistically optimal interpolation
        try:
            print(f"   🔬 Applying Ordinary Kriging interpolation...")
            
            # Create kriging model
            # Note: pykrige expects (lon, lat) order, not (lat, lon)
            OK = OrdinaryKriging(
                lons, lats, values,
                variogram_model='exponential',  # Can be 'linear', 'power', 'gaussian', 'spherical', 'exponential'
                verbose=True,
                enable_plotting=False,
                coordinates_type='geographic'  # For lat/lon coordinates
            )
            
            # Perform kriging interpolation
            interpolated_grid, variance_grid = OK.execute(
                'grid', lon_grid, lat_grid
            )
            
            print(f"   ✅ Kriging completed successfully")
            print(f"   📊 Mean kriging variance: {np.nanmean(variance_grid):.6f}")
            
        except Exception as kriging_error:
            print(f"   ⚠️ Kriging failed ({kriging_error}), falling back to cubic interpolation...")
            
            # Fallback to cubic interpolation if kriging fails
            grid_points = np.column_stack((lat_mesh.ravel(), lon_mesh.ravel()))
            interpolated_values = griddata(
                points, values, grid_points, 
                method='cubic', fill_value=np.nan
            )
            interpolated_grid = interpolated_values.reshape(lat_mesh.shape)
        
        # Apply statistical masking - only show areas within reasonable distance from data points
        # Calculate distance weights to avoid over-extrapolation
        max_distance = np.max(pdist(points)) * 0.3  # Max interpolation distance
        
        # Mask points too far from actual measurements
        for i in range(grid_resolution):
            for j in range(grid_resolution):
                grid_point = np.array([lat_mesh[i, j], lon_mesh[i, j]])
                min_distance = np.min(np.linalg.norm(points - grid_point, axis=1))
                if min_distance > max_distance:
                    interpolated_grid[i, j] = np.nan
        
        # Create colormap based on statistical distribution
        # Use standard deviations to define color ranges
        vmin = mean_val - 2 * std_val
        vmax = mean_val + 2 * std_val
        
        # Clip extreme values to avoid colormap issues
        interpolated_grid_clipped = np.clip(interpolated_grid, vmin, vmax)
        
        # Create matplotlib figure for the overlay
        plt.figure(figsize=(10, 10))
        plt.axis('off')
        plt.tight_layout(pad=0)
        
        # Create custom colormap
        colors = ['#000080', '#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF8000', '#FF0000']
        n_bins = 256
        cmap = mcolors.LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
        
        # Plot the interpolated field
        plt.imshow(
            interpolated_grid_clipped, 
            extent=[lon_min, lon_max, lat_min, lat_max],
            origin='lower',
            cmap=cmap,
            alpha=0.7,
            vmin=vmin,
            vmax=vmax
        )
        
        # Save to bytes for folium
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', 
                   transparent=True, dpi=100, pad_inches=0)
        buf.seek(0)
        plt.close()
        
        # Convert to base64 for folium
        img_b64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        
        # Create folium raster overlay
        bounds = [[lat_min, lon_min], [lat_max, lon_max]]
        
        overlay = folium.raster_layers.ImageOverlay(
            image=f"data:image/png;base64,{img_b64}",
            bounds=bounds,
            opacity=0.7,
            name=f'{measurement} Interpolated Field'
        )
        
        return overlay
        
    except Exception as e:
        print(f"⚠️ Could not create interpolated field for {measurement}: {e}")
        return None


def create_folium_concentration_map(csv_file, output_file=None):
    """
    Create an interactive folium map with gradient field visualizations
    of concentration data with toggleable measurements.
    
    Args:
        csv_file: Path to the CSV file
        output_file: Output HTML file name (optional)
    """
    
    print(f"📊 Creating folium concentration map from: {csv_file}")
    
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Convert timestamp to datetime for better handling
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Define the measurement columns we want to visualize
    measurement_columns = ['Countrate', 'U238', 'K40', 'Th232', 'Cs137', 'Height', 'Press', 'Temp', 'Hum']
    
    # Filter to only columns that exist in the data
    available_measurements = [col for col in measurement_columns if col in df.columns]
    print(f"📋 Available measurements: {available_measurements}")
    
    # Calculate map center
    lat_center = df['lat'].mean()
    lon_center = df['lon'].mean()
    
    # Calculate appropriate zoom level
    lat_range = df['lat'].max() - df['lat'].min()
    lon_range = df['lon'].max() - df['lon'].min()
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
    
    # Create base map
    m = folium.Map(
        location=[lat_center, lon_center],
        zoom_start=zoom,
        tiles=None  # We'll add custom tiles
    )
    
    # Add different tile layers
    folium.TileLayer(
        tiles='OpenStreetMap',
        name='OpenStreetMap',
        control=True
    ).add_to(m)
    
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Satellite',
        control=True
    ).add_to(m)
    
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Topographic',
        control=True
    ).add_to(m)
    
    # --- Kartverket terrain visualization (clone of official "Terrengdata / høydedata" viewer) ---
    
    # 1. Greyscale topographic base layer
    folium.WmsTileLayer(
        url='https://wms.geonorge.no/skwms1/wms.topo4.graatone',
        name='Kartverket Topo – greyscale',
        layers='topo4graatone_WMS',   # main layer id
        fmt='image/png',
        transparent=False,
        version='1.3.0',
        attr='© Kartverket',
        overlay=False,                # this becomes the base map
        control=True
    ).add_to(m)

    # Create feature groups for each measurement
    measurement_groups = {}
    
    for measurement in available_measurements:
        # Create a feature group for this measurement
        feature_group = folium.FeatureGroup(name=f'{measurement} Interpolated Field')
        
        # Get valid data for this measurement
        valid_data = df.dropna(subset=[measurement, 'lat', 'lon'])
        
        if len(valid_data) == 0:
            print(f"⚠️ No valid data for {measurement}")
            continue
        
        if len(valid_data) < 3:
            print(f"⚠️ Not enough data points for interpolation of {measurement} (need at least 3)")
            continue
        
        # Create interpolated field based on actual measurement values
        interpolated_overlay = create_interpolated_overlay(valid_data, measurement, m.get_bounds())
        
        if interpolated_overlay is not None:
            interpolated_overlay.add_to(feature_group)
        
        # Get statistics for reference
        values = valid_data[measurement].values
        min_val = np.min(values)
        max_val = np.max(values)
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        # Add feature group to map
        feature_group.add_to(m)
        measurement_groups[measurement] = feature_group
        
        # Create a separate feature group for this measurement's data points
        points_group = folium.FeatureGroup(name=f'{measurement} Data Points')
        
        # Use the same statistical scaling as the interpolated field
        vmin = mean_val - 2 * std_val
        vmax = mean_val + 2 * std_val
        
        # Add all data points for this specific measurement with matching colors
        for idx, row in valid_data.iterrows():
            # Calculate statistical metrics
            z_score = (row[measurement] - mean_val) / std_val if std_val > 0 else 0
            
            # Normalize value for color selection (same as interpolated field)
            if vmax != vmin:
                normalized = (row[measurement] - vmin) / (vmax - vmin)
                normalized = np.clip(normalized, 0, 1)  # Ensure 0-1 range
            else:
                normalized = 0.5
            
            # Use the exact same color scale as the interpolated field
            # This matches the matplotlib colormap: ['#000080', '#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF8000', '#FF0000']
            if normalized < 0.167:  # 0 to 1/6
                color = '#000080'  # Dark blue
            elif normalized < 0.333:  # 1/6 to 2/6
                color = '#0000FF'  # Blue
            elif normalized < 0.5:  # 2/6 to 3/6
                color = '#00FFFF'  # Cyan
            elif normalized < 0.667:  # 3/6 to 4/6
                color = '#00FF00'  # Lime/Green
            elif normalized < 0.833:  # 4/6 to 5/6
                color = '#FFFF00'  # Yellow
            else:  # 5/6 to 1
                color = '#FF8000'  # Orange to Red
            
            # For the highest values, use pure red
            if normalized >= 0.95:
                color = '#FF0000'  # Pure red for extreme high values
            
            # Create detailed popup with measurement info
            popup_text = f"""
            <b>{measurement}: {row[measurement]:.2f}</b><br>
            Lat: {row['lat']:.6f}<br>
            Lon: {row['lon']:.6f}<br>
            Time: {row['datetime'].strftime('%Y-%m-%d %H:%M:%S')}<br>
            <br><b>Statistics:</b><br>
            Mean: {mean_val:.2f}<br>
            Std Dev: {std_val:.2f}<br>
            Z-score: {z_score:.2f}<br>
            Normalized: {normalized:.3f}<br>
            Range: {min_val:.2f} - {max_val:.2f}<br>
            Color range: {vmin:.2f} - {vmax:.2f}
            """
            
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=4,
                popup=folium.Popup(popup_text, max_width=350),
                color='white',
                fillColor=color,
                fillOpacity=0.9,
                weight=1
            ).add_to(points_group)
        
        # Add points group to map
        points_group.add_to(m)
        measurement_groups[f"{measurement}_points"] = points_group
        
        print(f"✅ Added interpolated field for {measurement} (range: {min_val:.2f} - {max_val:.2f}, mean: {mean_val:.2f}, std: {std_val:.2f})")
        print(f"✅ Added {len(valid_data)} data points for {measurement} with matching color scale")
    

    # Add layer control (automatically detects FeatureGroups added to map)
    folium.LayerControl(collapsed=False).add_to(m)
    

    
    # Add color legend
    color_legend_html = '''
    <div style="position: fixed; 
                bottom: 30px; right: 10px; width: 200px; height: 120px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:12px; padding: 10px; border-radius: 5px;
                ">
    <h5 style="margin: 0 0 10px 0;">Color Scale</h5>
    <div style="background: linear-gradient(to right, #000080 0%, #0080FF 16.7%, #00FFFF 33.3%, #80FF00 50%, #FFFF00 66.7%, #FF8000 83.3%, #FF0000 100%); 
                height: 20px; width: 100%; border: 1px solid #ccc; margin-bottom: 5px;">
    </div>
    <div style="display: flex; justify-content: space-between; font-size: 10px;">
        <span>Low</span>
        <span>Mean - 2σ</span>
        <span>Mean</span>
        <span>Mean + 2σ</span>
        <span>High</span>
    </div>
    <p style="margin: 8px 0 0 0; font-size: 10px; text-align: center;">
        <b>Statistical Scaling:</b><br>
        Colors based on mean ± 2 standard deviations<br>
        for each measurement independently
    </p>
    </div>
    '''
    
    m.get_root().html.add_child(folium.Element(color_legend_html))
    
    # Add "Uncheck All" button with JavaScript functionality
    uncheck_all_html = '''
    <div style="position: fixed; 
                top: 10px; left: 10px; width: 120px; height: auto; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:12px; padding: 8px; border-radius: 5px;
                ">
    <button onclick="uncheckAllLayers()" 
            style="width: 100%; padding: 6px; background-color: #ff6b6b; color: white; 
                   border: none; border-radius: 3px; cursor: pointer; font-size: 11px;">
        📋 Uncheck All Layers
    </button>
    <button onclick="checkAllLayers()" 
            style="width: 100%; padding: 6px; background-color: #51cf66; color: white; 
                   border: none; border-radius: 3px; cursor: pointer; font-size: 11px; margin-top: 3px;">
        ✓ Check All Layers
    </button>
    </div>
    
    <script>
    function uncheckAllLayers() {
        // Wait a bit for the map to be fully loaded
        setTimeout(function() {
            // Find all checkboxes in the layer control
            var checkboxes = document.querySelectorAll('.leaflet-control-layers-overlays input[type="checkbox"]');
            checkboxes.forEach(function(checkbox) {
                if (checkbox.checked) {
                    checkbox.click(); // Trigger the click event to properly hide layers
                }
            });
        }, 100);
    }
    
    function checkAllLayers() {
        // Wait a bit for the map to be fully loaded
        setTimeout(function() {
            // Find all checkboxes in the layer control
            var checkboxes = document.querySelectorAll('.leaflet-control-layers-overlays input[type="checkbox"]');
            checkboxes.forEach(function(checkbox) {
                if (!checkbox.checked) {
                    checkbox.click(); // Trigger the click event to properly show layers
                }
            });
        }, 100);
    }
    
    // Auto-uncheck all layers when page loads (optional)
    document.addEventListener('DOMContentLoaded', function() {
        setTimeout(function() {
            // Uncomment the line below if you want all layers to start unchecked
            // uncheckAllLayers();
        }, 1000);
    });
    </script>
    '''
    
    m.get_root().html.add_child(folium.Element(uncheck_all_html))
    
    # Generate output filename if not provided
    if output_file is None:
        base_name = os.path.splitext(os.path.basename(csv_file))[0]
        output_file = f"concentration_map_{base_name}.html"
    
    return m, output_file


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
    measurement_columns = ['Countrate', 'U238', 'K40', 'Th232', 'Cs137', 'Height', 'Press', 'Temp', 'Hum']
    
    for col in measurement_columns:
        if col in df.columns:
            print(f"{col:10}: Min={df[col].min():8.2f}, Max={df[col].max():8.2f}, Mean={df[col].mean():8.2f}, Std={df[col].std():8.2f}")


if __name__ == "__main__":
    # File path to your CSV
    csv_file = "concentrations(in)(3).csv"
    
    try:
        # Display data summary
        display_data_summary(csv_file)
        
        # Create the interactive map
        print("\n🗺️ Creating folium concentration map...")
        
        map_obj, output_file = create_folium_concentration_map(csv_file)
        
        # Save the map
        print(f"💾 Saving map to '{output_file}'...")
        map_obj.save(output_file)
        print(f"✅ Map saved successfully as '{output_file}'")
        print(f"🌐 Open '{output_file}' in your web browser to view the interactive map")
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find the file '{csv_file}'")
        print("Please make sure the CSV file exists and the path is correct.")
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        import traceback
        traceback.print_exc() 