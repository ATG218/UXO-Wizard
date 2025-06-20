"""
Interactive Folium Maps
======================

Creates interactive maps with toggleable layers following grid_interpolator.py patterns.
Optimized for geophysical data visualization with multiple filter results.
"""

import folium
from folium import plugins
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import branca.colormap as cm
import logging

logger = logging.getLogger(__name__)


class InteractiveMaps:
    """
    Interactive mapping class for geophysical data visualization.
    
    Creates Folium maps with multiple toggleable layers for different
    filter results and analysis outputs.
    """
    
    def __init__(self, mapbox_token: Optional[str] = None):
        """
        Initialize interactive maps.
        
        Parameters:
            mapbox_token (Optional[str]): Mapbox API token for satellite imagery
        """
        self.mapbox_token = mapbox_token
        self.default_zoom = 12
        self.default_tiles = 'OpenStreetMap'
        
    def create_base_map(self, center_lat: float, center_lon: float,
                       zoom_start: Optional[int] = None) -> folium.Map:
        """
        Create base folium map.
        
        Parameters:
            center_lat (float): Center latitude
            center_lon (float): Center longitude
            zoom_start (Optional[int]): Initial zoom level
            
        Returns:
            folium.Map: Base map object
        """
        if zoom_start is None:
            zoom_start = self.default_zoom
            
        # Create base map with OpenStreetMap
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=zoom_start,
            tiles='OpenStreetMap',
            control_scale=False  # Disable default scale to avoid overlap
        )
        
        # Add measurement tool (you liked this)
        plugins.MeasureControl(position='topleft').add_to(m)
        
        # Add readable scale control  
        plugins.MiniMap(position='bottomright', width=120, height=80, minimized=True).add_to(m)
        
        # Add Kartverket DEM layer exactly as in grid_interpolator.py
        folium.WmsTileLayer(
            url='https://wms.geonorge.no/skwms1/wms.topo4.graatone',
            name='Kartverket Topo – greyscale',
            layers='topo4graatone_WMS',
            fmt='image/png',
            transparent=True,
            version='1.3.0',
            attr='©️ Kartverket',
            overlay=False,
            control=True
        ).add_to(m)
        
        return m
    
    def create_geophysical_colormap(self, data: np.ndarray, 
                                  colormap_name: str = 'RdYlBu_r',
                                  n_colors: int = 256) -> cm.LinearColormap:
        """
        Create colormap optimized for geophysical data.
        
        Parameters:
            data (np.ndarray): Data array for range calculation
            colormap_name (str): Matplotlib colormap name
            n_colors (int): Number of colors in colormap
            
        Returns:
            branca.colormap.LinearColormap: Folium colormap
        """
        # Calculate robust statistics to avoid outlier dominance
        valid_data = data[~np.isnan(data)]
        if len(valid_data) == 0:
            vmin, vmax = 0, 1
        else:
            # Use percentiles for robust range
            vmin = np.percentile(valid_data, 2)
            vmax = np.percentile(valid_data, 98)
            
            # Ensure we have a reasonable range
            if vmax - vmin < 1e-10:
                center = np.mean(valid_data)
                vmin = center - np.std(valid_data)
                vmax = center + np.std(valid_data)
        
        # Create colormap
        colormap = cm.LinearColormap(
            colors=plt.cm.get_cmap(colormap_name)(np.linspace(0, 1, n_colors)),
            vmin=vmin,
            vmax=vmax,
            caption=f'Range: {vmin:.2f} to {vmax:.2f}'
        )
        
        return colormap
    
    def create_diverging_colormap(self, data: np.ndarray,
                                center_value: float = 0.0) -> cm.LinearColormap:
        """
        Create diverging colormap centered at specified value.
        
        Parameters:
            data (np.ndarray): Data array
            center_value (float): Center value for diverging colormap
            
        Returns:
            branca.colormap.LinearColormap: Diverging colormap
        """
        valid_data = data[~np.isnan(data)]
        if len(valid_data) == 0:
            return self.create_geophysical_colormap(data)
            
        # Calculate symmetric range around center
        abs_max = np.max(np.abs(valid_data - center_value))
        vmin = center_value - abs_max
        vmax = center_value + abs_max
        
        # Create diverging colormap (RdBu_r is good for magnetic anomalies)
        colormap = cm.LinearColormap(
            colors=['#2166ac', '#4393c3', '#92c5de', '#d1e5f0', 
                   '#f7f7f7', '#fdbf6f', '#fd8d3c', '#e31a1c', '#b10026'],
            vmin=vmin,
            vmax=vmax,
            caption=f'Range: {vmin:.2f} to {vmax:.2f}'
        )
        
        return colormap
    
    def add_grid_overlay(self, m: folium.Map, grid_data: np.ndarray,
                        grid_lat: np.ndarray, grid_lon: np.ndarray,
                        layer_name: str = 'Grid Data',
                        opacity: float = 0.8) -> folium.Map:
        """
        Add grid data as raster overlay with precise positioning and proper toggleability.
        
        Parameters:
            m (folium.Map): Base map
            grid_data (np.ndarray): 2D grid data
            grid_lat, grid_lon (np.ndarray): Lat/lon grids
            layer_name (str): Name for the layer
            opacity (float): Layer opacity
            
        Returns:
            folium.Map: Map with grid overlay added
        """
        try:
            # Create raster overlay using EXACT same logic as PNG creation
            img_str = self.create_raster_overlay_from_png_logic(grid_data, layer_name)
            
            # Define precise bounds using corner coordinates for exact positioning
            # Use the actual grid coordinate corners, not min/max which can be off
            lat_min = grid_lat[0, 0]    # Bottom-left
            lat_max = grid_lat[-1, -1]  # Top-right
            lon_min = grid_lon[0, 0]    # Bottom-left
            lon_max = grid_lon[-1, -1]  # Top-right
            
            bounds = [
                [lat_min, lon_min],  # Southwest corner
                [lat_max, lon_max]   # Northeast corner
            ]
            
            logger.info(f"Adding {layer_name} with precise bounds: [{lat_min:.6f}, {lon_min:.6f}] to [{lat_max:.6f}, {lon_max:.6f}]")
            logger.info(f"Grid data shape: {grid_data.shape}, valid points: {np.sum(~np.isnan(grid_data))}")
            
            # Create feature group for toggleability
            feature_group = folium.FeatureGroup(name=layer_name, overlay=True, control=True, show=True)
            
            # Add raster overlay to feature group with precise positioning
            overlay = folium.raster_layers.ImageOverlay(
                image=f"data:image/png;base64,{img_str}",
                bounds=bounds,
                opacity=opacity,
                interactive=True,
                cross_origin=False
            )
            overlay.add_to(feature_group)
            
            # Add feature group to map
            feature_group.add_to(m)
            
            logger.info(f"Successfully added precisely positioned grid overlay: {layer_name}")
            
        except Exception as e:
            logger.error(f"Failed to add grid overlay {layer_name}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
        return m
    
    def add_flight_path_overlay(self, m: folium.Map, df: pd.DataFrame,
                               lat_col: str = 'y', lon_col: str = 'x',
                               mag_col: str = 'magnetic_field',
                               max_points: int = 5000) -> folium.Map:
        """
        Add flight path as colored line overlay.
        
        Parameters:
            m (folium.Map): Base map
            df (pd.DataFrame): DataFrame with flight data
            lat_col (str): Latitude column name
            lon_col (str): Longitude column name
            mag_col (str): Magnetic field column name
            max_points (int): Maximum points to plot (for performance)
            
        Returns:
            folium.Map: Map with flight path overlay
        """
        try:
            # Skip flight path for UTM data to avoid coordinate issues
            if df[lat_col].max() > 1000:  # UTM coordinates
                logger.info("Skipping flight path overlay for UTM coordinates")
                return m
                
            # Only proceed with geographic coordinates
            df_plot = df.copy()
                
            # Downsample for performance if needed
            if len(df_plot) > max_points:
                step = len(df_plot) // max_points
                df_plot = df_plot.iloc[::step].copy()
            
            # Remove NaN values
            valid_mask = (~df_plot[lat_col].isna() & 
                         ~df_plot[lon_col].isna() & 
                         ~df_plot[mag_col].isna())
            df_plot = df_plot[valid_mask]
            
            if len(df_plot) == 0:
                logger.warning("No valid flight path data to plot")
                return m
                
            # Create feature group for flight path
            flight_group = folium.FeatureGroup(name='Flight Path', control=True)
            
            # Add simple markers without complex coloring for now
            for _, row in df_plot.iterrows():
                folium.CircleMarker(
                    location=[row[lat_col], row[lon_col]],
                    radius=2,
                    popup=f"Mag: {row[mag_col]:.2f} nT",
                    color='red',
                    fill=True,
                    fillColor='red',
                    fillOpacity=0.6,
                    weight=1
                ).add_to(flight_group)
            
            flight_group.add_to(m)
            logger.info(f"Added flight path with {len(df_plot)} points")
            
        except Exception as e:
            logger.error(f"Failed to add flight path overlay: {str(e)}")
            
        return m
    
    def add_contour_overlay(self, m: folium.Map, grid_data: np.ndarray,
                          extent: Tuple[float, float, float, float],
                          contour_levels: Optional[List[float]] = None,
                          layer_name: str = 'Contours') -> folium.Map:
        """
        Add contour lines as overlay.
        
        Parameters:
            m (folium.Map): Base map
            grid_data (np.ndarray): 2D grid data
            extent (Tuple): (lon_min, lon_max, lat_min, lat_max)
            contour_levels (Optional[List[float]]): Specific contour levels
            layer_name (str): Layer name
            
        Returns:
            folium.Map: Map with contour overlay
        """
        try:
            # Create contour feature group
            contour_group = folium.FeatureGroup(name=layer_name, control=True)
            
            # Generate contour levels if not provided
            if contour_levels is None:
                valid_data = grid_data[~np.isnan(grid_data)]
                if len(valid_data) > 0:
                    vmin, vmax = np.percentile(valid_data, [10, 90])
                    contour_levels = np.linspace(vmin, vmax, 8)
                else:
                    contour_levels = []
            
            if len(contour_levels) == 0:
                return m
                
            # Create coordinate grids
            ny, nx = grid_data.shape
            lon_grid = np.linspace(extent[0], extent[1], nx)
            lat_grid = np.linspace(extent[2], extent[3], ny)
            
            # Create contours using matplotlib
            fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
            cs = ax.contour(lon_grid, lat_grid, grid_data, levels=contour_levels)
            plt.close(fig)
            
            # Convert contours to GeoJSON and add to map
            for i, collection in enumerate(cs.collections):
                for path in collection.get_paths():
                    # Convert path to coordinates
                    vertices = path.vertices
                    if len(vertices) > 2:  # Valid contour line
                        coordinates = [[float(lon), float(lat)] for lon, lat in vertices]
                        
                        folium.PolyLine(
                            locations=[[lat, lon] for lon, lat in coordinates],
                            color='black',
                            weight=1,
                            opacity=0.8,
                            popup=f"Contour: {contour_levels[i]:.2f}"
                        ).add_to(contour_group)
            
            contour_group.add_to(m)
            
            logger.info(f"Added {len(contour_levels)} contour levels")
            
        except Exception as e:
            logger.error(f"Failed to add contour overlay: {str(e)}")
            
        return m
    
    def add_anomaly_markers(self, m: folium.Map, anomalies: pd.DataFrame,
                          lat_col: str = 'latitude', lon_col: str = 'longitude',
                          confidence_col: str = 'confidence') -> folium.Map:
        """
        Add anomaly detection results as markers.
        
        Parameters:
            m (folium.Map): Base map
            anomalies (pd.DataFrame): Anomaly detection results
            lat_col (str): Latitude column
            lon_col (str): Longitude column
            confidence_col (str): Confidence score column
            
        Returns:
            folium.Map: Map with anomaly markers
        """
        try:
            # Create anomaly feature group
            anomaly_group = folium.FeatureGroup(name='Detected Anomalies', control=True)
            
            # Color code by confidence
            for _, row in anomalies.iterrows():
                confidence = row[confidence_col] if confidence_col in row else 0.5
                
                # Color based on confidence (red = high, yellow = medium, blue = low)
                if confidence > 0.8:
                    color = 'red'
                    radius = 8
                elif confidence > 0.5:
                    color = 'orange'
                    radius = 6
                else:
                    color = 'blue'
                    radius = 4
                
                folium.CircleMarker(
                    location=[row[lat_col], row[lon_col]],
                    radius=radius,
                    popup=f"Anomaly Confidence: {confidence:.3f}<br>Location: {row[lat_col]:.6f}, {row[lon_col]:.6f}",
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.8,
                    weight=2
                ).add_to(anomaly_group)
            
            anomaly_group.add_to(m)
            
            logger.info(f"Added {len(anomalies)} anomaly markers")
            
        except Exception as e:
            logger.error(f"Failed to add anomaly markers: {str(e)}")
            
        return m
    
    def create_raster_overlay_from_png_logic(self, grid_data: np.ndarray, name: str) -> str:
        """Create raster overlay using EXACTLY the same logic as PNG creation"""
        
        import matplotlib.pyplot as plt
        
        # Use EXACT same colormap logic as PNG creation
        if 'RTP' in name or 'Original' in name:
            cmap = 'RdYlBu_r'
        elif 'THG' in name or 'Analytic' in name:
            cmap = 'viridis'
        elif 'Tilt' in name:
            cmap = 'RdBu_r'
        else:
            cmap = 'plasma'
        
        # Use EXACT same normalization as PNG creation with outlier protection
        valid_data = grid_data[~np.isnan(grid_data)]
        if len(valid_data) > 0:
            # Remove extreme outliers first
            mean_val = np.mean(valid_data)
            std_val = np.std(valid_data)
            
            # Filter out values beyond 5 standard deviations
            outlier_mask = np.abs(valid_data - mean_val) <= 5 * std_val
            if np.any(outlier_mask):
                filtered_data = valid_data[outlier_mask]
                vmin = np.percentile(filtered_data, 2)
                vmax = np.percentile(filtered_data, 98)
                
                outliers_removed = len(valid_data) - len(filtered_data)
                if outliers_removed > 0:
                    logger.warning(f"{name}: Removed {outliers_removed} extreme outliers (beyond 5σ)")
            else:
                vmin = np.percentile(valid_data, 2)
                vmax = np.percentile(valid_data, 98)
        else:
            vmin, vmax = 0, 1
        
        logger.info(f"Raster {name}: range [{vmin:.3f}, {vmax:.3f}], colormap: {cmap}")
        
        # Create figure WITHOUT axes or decorations - just the data
        fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
        
        # Create the image with EXACT same settings as PNG
        # Pre-clip data to prevent any transparency issues with out-of-range values
        import matplotlib.pyplot as plt
        
        # Create a copy of the data and manually clip it to the colormap range
        clipped_data = grid_data.copy()
        
        # Identify valid (non-NaN) data
        valid_mask = ~np.isnan(clipped_data)
        
        # Clip only the valid data to the vmin/vmax range
        clipped_data[valid_mask] = np.clip(clipped_data[valid_mask], vmin, vmax)
        
        # Create alpha channel: 1.0 for valid data, 0.0 for NaN
        alpha_channel = valid_mask.astype(float)
        
        # Use simple imshow without complex normalization
        ax.imshow(clipped_data, origin='lower', 
                 cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal', alpha=alpha_channel)
        
        # Remove ALL decorations to get pure data image
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        
        # Save to buffer with transparency as user requested
        buffer = BytesIO()
        plt.savefig(buffer, format='PNG', dpi=100, bbox_inches='tight', 
                   pad_inches=0, transparent=True, facecolor='none')
        plt.close(fig)
        
        # Get base64 string
        buffer.seek(0)
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        logger.info(f"Created raster overlay for {name} using exact PNG logic")
        return img_str
    
    
