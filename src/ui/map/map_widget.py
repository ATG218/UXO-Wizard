"""
Main Map Widget for UXO Wizard - pyqtlet2 based implementation

Features:
- Real-time layer management without HTML reloads
- Primary: Kartverket (Norwegian Mapping Authority) topographic maps
- Backup: OpenStreetMap and Satellite imagery
- Dynamic layer styling and visibility control
- Interactive tools with bidirectional data communication
"""

import os
# Make sure qtpy and pyqtlet2 use the same Qt binding
os.environ["QT_API"] = "pyside6"

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QToolBar, 
    QToolButton, QSplitter
)
from qtpy.QtCore import Qt, Signal, QUrl, QTimer
from qtpy.QtGui import QKeySequence, QShortcut
from pyqtlet2 import L, MapWidget
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from loguru import logger
import json
import base64
from io import BytesIO
from datetime import datetime

from .layer_types import UXOLayer, LayerType, GeometryType, LayerStyle
from .layer_manager import LayerManager


class UXOMapWidget(QWidget):
    """Main map widget with real-time layer management using pyqtlet2"""
    
    # Signals
    coordinates_clicked = Signal(float, float)  # lat, lon
    area_selected = Signal(list)  # List of coordinates
    feature_selected = Signal(str, list)  # layer_name, feature_ids
    map_ready = Signal()  # Emitted when map is fully initialized
    
    def __init__(self):
        super().__init__()
        self.layer_manager = LayerManager()
        self.leaflet_layers: Dict[str, L.layerGroup] = {}
        self._layer_cache: Dict[str, L.layerGroup] = {}  # Cache for hidden layers
        self._map_created = False
        self.current_draw_layer = None
        
        self.setup_ui()
        self.connect_signals()

    def setup_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Toolbar
        toolbar = QToolBar()
        
        # Navigation controls
        self.center_btn = QToolButton()
        self.center_btn.setText("🎯")
        self.center_btn.setToolTip("Center on Default Location (Tarva Island)")
        self.center_btn.clicked.connect(self.center_default)
        toolbar.addWidget(self.center_btn)
        
        self.zoom_to_data_btn = QToolButton()
        self.zoom_to_data_btn.setText("📍")
        self.zoom_to_data_btn.setToolTip("Zoom to Data Extent")
        self.zoom_to_data_btn.clicked.connect(self.zoom_to_data)
        toolbar.addWidget(self.zoom_to_data_btn)
        
        self.fullscreen_btn = QToolButton()
        self.fullscreen_btn.setText("🔳")
        self.fullscreen_btn.setToolTip("Toggle Fullscreen")
        self.fullscreen_btn.setCheckable(True)
        self.fullscreen_btn.clicked.connect(self.toggle_fullscreen)
        toolbar.addWidget(self.fullscreen_btn)
        
        toolbar.addSeparator()
        
        # Drawing and measurement tools
        self.measure_btn = QToolButton()
        self.measure_btn.setText("📏")
        self.measure_btn.setToolTip("Measure Distance")
        self.measure_btn.setCheckable(True)
        self.measure_btn.clicked.connect(self.toggle_measure)
        toolbar.addWidget(self.measure_btn)
        
        self.draw_btn = QToolButton()
        self.draw_btn.setText("✏️")
        self.draw_btn.setToolTip("Draw Tools")
        self.draw_btn.setCheckable(True)
        self.draw_btn.clicked.connect(self.toggle_draw)
        toolbar.addWidget(self.draw_btn)
        
        self.clear_btn = QToolButton()
        self.clear_btn.setText("🗑️")
        self.clear_btn.setToolTip("Clear Drawings")
        self.clear_btn.clicked.connect(self.clear_drawings)
        toolbar.addWidget(self.clear_btn)
        
        toolbar.addSeparator()
        
        # Export tools
        self.export_btn = QToolButton()
        self.export_btn.setText("💾")
        self.export_btn.setToolTip("Export Map")
        self.export_btn.clicked.connect(self.export_map)
        toolbar.addWidget(self.export_btn)
        
        # Map widget
        self.map_widget = MapWidget()

        # Let local HTML fetch remote (tile) URLs  ← Qt 6 default blocks this
        from qtpy.QtWebEngineCore import QWebEngineSettings
        self.map_widget.settings().setAttribute(
            QWebEngineSettings.LocalContentCanAccessRemoteUrls, True
        )
        
        # Layout
        layout.addWidget(toolbar)
        layout.addWidget(self.map_widget)
        self.setLayout(layout)
        
        # Add escape key shortcut for exiting fullscreen
        self.escape_shortcut = QShortcut(QKeySequence(Qt.Key_Escape), self)
        self.escape_shortcut.activated.connect(self.exit_fullscreen)
        
    def setup_map(self):
        """Initialize the pyqtlet2 map with Tarva island as default center"""
        logger.info("Setting up pyqtlet2 map centered on Tarva island")

        # Create map centered on Tarva island (63°43'N, 9°22'E)
        self.map = L.map(self.map_widget)
        self.map.setView([63.8167, 9.3667], 12)  # Tarva island coordinates, zoom 12 for detailed view
        
        # Add map layers
        self._add_base_layers()
        
        # Add map controls
        self._add_map_controls()
        
        # Initialize the JavaScript-side layer management system
        self._init_js_layer_manager()
        
        # Set up event handlers
        self._setup_event_handlers()
        
        logger.info("Norwegian UXO map ready with pyqtlet2")
        self.map_ready.emit()
        
    def _add_base_layers(self):
        """Add base map layers - simplified for debugging"""
        logger.debug("Adding base map layers (OpenStreetMap only)")
        
        try:
            # Simple OpenStreetMap layer for debugging
            osm = L.tileLayer(
                'https://cache.kartverket.no/v1/wmts/1.0.0/topo/default/webmercator/{z}/{y}/{x}.png',
                {
                    'attribution': '© Kartverket',
                    'maxZoom': 18,
                    'tileSize': 256
                }
            )
            osm.addTo(self.map)
            self.base_osm = osm
            logger.debug("OpenStreetMap layer added successfully")
            
        except Exception as e:
            logger.error(f"Failed to add OpenStreetMap layer: {e}")
            
    def _add_map_controls(self):
        """Add map controls"""
        try:
            # Note: pyqtlet2 has limited control support compared to standard Leaflet
            # Scale control and layer control may need to be implemented differently
            
            # For now, we'll manage layers through our custom layer panel
            logger.debug("Using custom layer control panel instead of Leaflet controls")
            
        except Exception as e:
            logger.warning(f"Failed to add some map controls: {e}")
            
    def _setup_event_handlers(self):
        """Set up map event handlers"""
        try:
            # Click handler
            @self.map.clicked.connect
            def on_map_click(event):
                lat = event['latlng']['lat']
                lng = event['latlng']['lng']
                self.coordinates_clicked.emit(lat, lng)
                logger.debug(f"Map clicked at: {lat}, {lng}")
                
            # Note: pyqtlet2 doesn't support mousemove events out of the box
            # We'll rely on click events for now
                
        except Exception as e:
            logger.warning(f"Failed to set up some event handlers: {e}")
            
    def connect_signals(self):
        """Connect layer manager signals"""
        self.layer_manager.layer_added.connect(self._on_layer_added)
        self.layer_manager.layer_removed.connect(self._on_layer_removed)
        self.layer_manager.layer_visibility_changed.connect(self._on_layer_visibility_changed)
        self.layer_manager.layer_style_changed.connect(self._on_layer_style_changed)
        self.layer_manager.layer_opacity_changed.connect(self._on_layer_opacity_changed)
        
    def add_layer_realtime(self, uxo_layer: UXOLayer):
        """Add layer without HTML reload"""
        logger.info(f"Adding layer in real-time: {uxo_layer.name}")
        
        # Add to layer manager
        self.layer_manager.add_layer(uxo_layer)
        
    def _on_layer_added(self, layer: UXOLayer):
        """Handle layer addition"""
        try:
            # Create and store the layer in JavaScript
            success = self._create_leaflet_layer(layer)
            
            if success:
                # If the layer should be visible, call the JS function to show it
                if layer.is_visible:
                    layer_name_js = json.dumps(layer.name)
                    self.map_widget.page.runJavaScript(f"showUxoLayer({layer_name_js});")
                    logger.info(f"Layer '{layer.name}' created and shown on map")
                    self.zoom_to_visible_layers()
                else:
                    logger.info(f"Layer '{layer.name}' created and stored (hidden)")
            else:
                logger.warning(f"Could not create leaflet layer for '{layer.name}'")

        except Exception as e:
            logger.error(f"Failed to handle layer addition '{layer.name}': {e}")
            import traceback
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            
    def _create_leaflet_layer(self, layer: UXOLayer) -> bool:
        """
        Creates and stores a leaflet layer in the JavaScript context.
        Returns True on success, False on failure.
        """
        if layer.layer_type in [LayerType.POINTS, LayerType.ANNOTATION] and layer.geometry_type == GeometryType.POINT:
            return self._create_point_layer_geojson(layer)
        elif layer.layer_type == LayerType.RASTER:
            return self._create_raster_layer(layer)
        elif layer.layer_type == LayerType.VECTOR and layer.geometry_type == GeometryType.LINE:
            return self._create_vector_layer_geojson(layer)
        else:
            logger.warning(f"Unsupported combination of layer type '{layer.layer_type.value}' and geometry type '{layer.geometry_type.value}'")
            return False

    def _create_point_layer_geojson(self, layer: UXOLayer):
        """Create point layer from DataFrame using a single GeoJSON object for efficiency"""
        if not isinstance(layer.data, pd.DataFrame):
            logger.error(f"Point layer '{layer.name}' requires DataFrame data, but got {type(layer.data)}")
            return False

        lat_col, lon_col = self._detect_coordinate_columns(layer.data)
        if not lat_col or not lon_col:
            logger.error(f"Could not detect coordinate columns for layer '{layer.name}'")
            return False

        # Check for graduated colors setup
        style = layer.style
        use_graduated_colors = style.use_graduated_colors and style.color_field
        color_field = style.color_field
        color_ramp = style.color_ramp or ["#000080", "#0000FF", "#00FFFF", "#00FF00", "#FFFF00", "#FF8000", "#FF0000"]
        
        # Calculate value range for graduated colors if needed
        vmin = vmax = None
        if use_graduated_colors and color_field in layer.data.columns:
            # First check if vmin/vmax are provided in metadata (from gamma interpolator)
            if (layer.metadata and 
                isinstance(layer.metadata, dict) and 
                'vmin' in layer.metadata and 
                'vmax' in layer.metadata):
                vmin = float(layer.metadata['vmin'])
                vmax = float(layer.metadata['vmax'])
                logger.info(f"Using metadata range for '{layer.name}': [{vmin:.2f}, {vmax:.2f}]")
            else:
                # Calculate statistical range like folium example
                valid_values = layer.data[color_field].dropna()
                if len(valid_values) > 0:
                    mean_val = float(valid_values.mean())
                    std_val = float(valid_values.std())
                    # Use statistical scaling like folium example: mean ± 2σ
                    vmin = mean_val - 2 * std_val
                    vmax = mean_val + 2 * std_val
                    logger.info(f"Calculated statistical range for '{layer.name}': field='{color_field}', range=[{vmin:.2f}, {vmax:.2f}]")

        # Helper function to get color for a value (matches folium example logic)
        def get_color_for_value(value):
            if not use_graduated_colors or vmin is None or vmax is None:
                return style.point_color
            
            if pd.isna(value):
                return style.point_color
            
            # Normalize value to 0-1 range
            if vmax != vmin:
                normalized = (value - vmin) / (vmax - vmin)
                normalized = max(0, min(1, normalized))  # Clamp to [0,1]
            else:
                normalized = 0.5
            
            # Map normalized value to color ramp using folium-style discrete ranges
            # This matches the logic from the folium example
            if normalized < 0.167:  # 0 to 1/6
                return color_ramp[0] if len(color_ramp) > 0 else "#000080"
            elif normalized < 0.333:  # 1/6 to 2/6
                return color_ramp[1] if len(color_ramp) > 1 else "#0000FF"
            elif normalized < 0.5:  # 2/6 to 3/6
                return color_ramp[2] if len(color_ramp) > 2 else "#00FFFF"
            elif normalized < 0.667:  # 3/6 to 4/6
                return color_ramp[3] if len(color_ramp) > 3 else "#00FF00"
            elif normalized < 0.833:  # 4/6 to 5/6
                return color_ramp[4] if len(color_ramp) > 4 else "#FFFF00"
            elif normalized < 0.95:  # 5/6 to 95%
                return color_ramp[5] if len(color_ramp) > 5 else "#FF8000"
            else:  # 95% to 100% (extreme high values)
                return color_ramp[-1] if len(color_ramp) > 0 else "#FF0000"

        # Convert DataFrame to GeoJSON FeatureCollection
        features = []
        for _, row in layer.data.iterrows():
            try:
                lat = float(row[lat_col])
                lon = float(row[lon_col])
                properties = row.to_dict()

                # Ensure all properties are JSON serializable
                for key, value in properties.items():
                    if pd.isna(value):
                        properties[key] = None
                    elif isinstance(value, (datetime, pd.Timestamp)):
                        properties[key] = value.isoformat()

                # Add color information to properties for graduated colors
                if use_graduated_colors and color_field in properties:
                    properties['_point_color'] = get_color_for_value(properties[color_field])
                else:
                    properties['_point_color'] = style.point_color

                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [lon, lat]
                    },
                    "properties": properties
                }
                features.append(feature)
            except (ValueError, TypeError) as e:
                logger.warning(f"Skipping invalid coordinates in layer '{layer.name}': {e}")
                continue

        if not features:
            logger.warning(f"No valid points could be processed for layer '{layer.name}'")
            return False

        geojson_data_str = json.dumps({
            "type": "FeatureCollection",
            "features": features
        })

        # Build JavaScript for layer creation with graduated colors support
        layer_name_js = json.dumps(layer.name)
        
        # Create pointToLayer function that uses the _point_color property
        point_to_layer_func = f"""
            function(feature, latlng) {{
                var pointColor = feature.properties._point_color || '{style.point_color}';
                return L.circleMarker(latlng, {{
                    radius: {style.point_size},
                    color: pointColor,
                    weight: 1,
                    opacity: {style.point_opacity},
                    fillColor: pointColor,
                    fillOpacity: {style.point_opacity}
                }});
            }}
        """
        
        # Enhanced popup function that shows color field info if available
        if use_graduated_colors and color_field:
            popup_extra = f"""
                    if (feature.properties['{color_field}'] !== undefined && feature.properties['{color_field}'] !== null) {{
                        popupContent += '<tr><td style="padding-right: 10px;"><strong>Color Value ({color_field})</strong></td><td>' + feature.properties['{color_field}'] + '</td></tr>';
                        popupContent += '<tr><td style="padding-right: 10px;"><strong>Value Range</strong></td><td>{vmin:.2f} - {vmax:.2f}</td></tr>';
                    }}
            """
        else:
            popup_extra = ""
            
        on_each_feature_func = f"""
            function(feature, layer) {{
                if (feature.properties) {{
                    var popupContent = '<div style="max-height: 200px; overflow-y: auto;"><table>';
                    for (var p in feature.properties) {{
                        if (p !== '_point_color') {{  // Skip internal color property
                            popupContent += '<tr><td style="padding-right: 10px;"><strong>' + p + '</strong></td><td>' + feature.properties[p] + '</td></tr>';
                        }}
                    }}
                    {popup_extra}
                    popupContent += '</table></div>';
                    layer.bindPopup(popupContent);
                }}
            }}
        """

        js_code = f"""
            var geojsonData = {geojson_data_str};
            var options = {{
                pointToLayer: {point_to_layer_func},
                onEachFeature: {on_each_feature_func}
            }};
            var geoJsonLayer = L.geoJson(geojsonData, options);
            window.uxoMapLayers[{layer_name_js}] = geoJsonLayer;
        """

        self.map_widget.page.runJavaScript(js_code)
        
        if use_graduated_colors:
            logger.info(f"Created graduated color layer '{layer.name}' with {len(features)} points colored by '{color_field}'")
        else:
            logger.info(f"Created single-color layer '{layer.name}' with {len(features)} points")
        return True
            
    def _create_raster_layer(self, layer: UXOLayer):
        """Create raster layer with gradient heatmap from numpy array data"""
        try:
            logger.info(f"Creating raster layer: {layer.name}")
            
            # Extract numpy array from data
            data = layer.data
            if isinstance(data, dict) and 'grid' in data:
                grid_array = data['grid']
            elif isinstance(data, np.ndarray):
                grid_array = data
            else:
                logger.error(f"Raster layer '{layer.name}' requires numpy array or dict with 'grid' key, got {type(data)}")
                return False
            
            if not isinstance(grid_array, np.ndarray):
                logger.error(f"Grid data must be numpy array, got {type(grid_array)}")
                return False
            
            # Get bounds
            if not layer.bounds:
                logger.error(f"Raster layer '{layer.name}' requires bounds")
                return False
            
            # Create gradient heatmap image
            image_data_url = self._array_to_heatmap_image(grid_array, layer.name)
            if not image_data_url:
                logger.error(f"Failed to create heatmap image for '{layer.name}'")
                return False
            
            # Convert bounds to Leaflet format [[south, west], [north, east]]
            bounds = layer.bounds  # [min_x, min_y, max_x, max_y]
            leaflet_bounds = [[bounds[1], bounds[0]], [bounds[3], bounds[2]]]
            
            # Create JavaScript code for ImageOverlay
            layer_name_js = json.dumps(layer.name)
            opacity = layer.opacity if hasattr(layer, 'opacity') else 0.7
            
            js_code = f"""
                var imageUrl = '{image_data_url}';
                var imageBounds = {json.dumps(leaflet_bounds)};
                var imageOverlay = L.imageOverlay(imageUrl, imageBounds, {{
                    opacity: {opacity},
                    interactive: true
                }});
                
                // Add popup with layer info
                imageOverlay.bindPopup('<b>{layer.name}</b><br/>Raster Layer<br/>Size: {grid_array.shape[0]}x{grid_array.shape[1]}<br/>Data range: {np.min(grid_array):.2f} - {np.max(grid_array):.2f}');
                
                window.uxoMapLayers[{layer_name_js}] = imageOverlay;
            """
            
            self.map_widget.page.runJavaScript(js_code)
            logger.info(f"Created raster layer '{layer.name}' with {grid_array.shape} grid")
            return True
            
        except Exception as e:
            logger.error(f"Error creating raster layer '{layer.name}': {e}")
            import traceback
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            return False
    
    def _array_to_heatmap_image(self, array: np.ndarray, layer_name: str) -> str:
        """Convert numpy array to base64 encoded heatmap image"""
        try:
            # Preserve NaN mask for transparency
            nan_mask = np.isnan(array)
            
            # Normalize array to 0-1 range (only valid values)
            array_min = np.nanmin(array)
            array_max = np.nanmax(array)
            if array_max == array_min:
                normalized = np.zeros_like(array)
            else:
                normalized = (array - array_min) / (array_max - array_min)
            
            # Keep NaN values as NaN for now, clip valid values
            normalized = np.clip(normalized, 0.0, 1.0)
            
            # Create gradient heatmap using matplotlib-style colormap
            colored_array = self._apply_heatmap_colormap(normalized, nan_mask)
            
            # Convert to PIL Image
            try:
                from PIL import Image
            except ImportError:
                logger.error("PIL (Pillow) not available. Installing is required for raster layers.")
                return None
            
            # Convert RGBA array to PIL Image
            # Flip vertically to match geographic convention (north-up)
            image_array = np.flipud(colored_array)
            image = Image.fromarray((image_array * 255).astype(np.uint8), mode='RGBA')
            
            # Resize for smoother interpolation if the grid is small
            if image.size[0] < 512 or image.size[1] < 512:
                # Scale up with bilinear interpolation for smoother appearance
                scale_factor = max(512 // image.size[0], 512 // image.size[1], 1)
                new_size = (image.size[0] * scale_factor, image.size[1] * scale_factor)
                image = image.resize(new_size, Image.Resampling.BILINEAR)
            
            # Convert to base64 data URL
            buffer = BytesIO()
            image.save(buffer, format='PNG')
            buffer.seek(0)
            
            # Create data URL
            img_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            data_url = f"data:image/png;base64,{img_data}"
            
            logger.debug(f"Created heatmap image for '{layer_name}': {array.shape} -> {image.size}")
            return data_url
            
        except Exception as e:
            logger.error(f"Error converting array to heatmap image: {e}")
            return None
    
    def _apply_heatmap_colormap(self, normalized_array: np.ndarray, nan_mask: np.ndarray = None) -> np.ndarray:
        """Apply viridis-like colormap to normalized array (0-1 range)"""
        # Create a viridis-like gradient: dark purple -> blue -> green -> yellow
        height, width = normalized_array.shape
        colored = np.zeros((height, width, 4), dtype=np.float32)  # RGBA
        
        # Define viridis-like color stops (value, R, G, B, A)
        color_stops = [
            (0.0, 0.267, 0.004, 0.329, 0.9),   # Dark purple
            (0.2, 0.282, 0.140, 0.457, 0.9),   # Purple-blue
            (0.4, 0.253, 0.265, 0.529, 0.9),   # Blue
            (0.6, 0.163, 0.471, 0.558, 0.9),   # Blue-green
            (0.8, 0.477, 0.741, 0.408, 0.9),   # Green-yellow
            (1.0, 0.993, 0.906, 0.144, 0.9),   # Bright yellow
        ]
        
        for i in range(height):
            for j in range(width):
                # Handle NaN values (make them transparent)
                if nan_mask is not None and nan_mask[i, j]:
                    colored[i, j] = [0, 0, 0, 0]  # Fully transparent
                    continue
                    
                value = normalized_array[i, j]
                
                # Skip NaN values that weren't caught by the mask
                if np.isnan(value):
                    colored[i, j] = [0, 0, 0, 0]  # Fully transparent
                    continue
                
                # Find appropriate color stops
                if value <= color_stops[0][0]:
                    # Below first stop
                    colored[i, j] = color_stops[0][1:5]
                elif value >= color_stops[-1][0]:
                    # Above last stop
                    colored[i, j] = color_stops[-1][1:5]
                else:
                    # Interpolate between stops
                    for k in range(len(color_stops) - 1):
                        if color_stops[k][0] <= value <= color_stops[k + 1][0]:
                            # Linear interpolation
                            t = (value - color_stops[k][0]) / (color_stops[k + 1][0] - color_stops[k][0])
                            
                            r = color_stops[k][1] + t * (color_stops[k + 1][1] - color_stops[k][1])
                            g = color_stops[k][2] + t * (color_stops[k + 1][2] - color_stops[k][2])
                            b = color_stops[k][3] + t * (color_stops[k + 1][3] - color_stops[k][3])
                            a = color_stops[k][4] + t * (color_stops[k + 1][4] - color_stops[k][4])
                            
                            colored[i, j] = [r, g, b, a]
                            break
        
        return colored
        
    def _create_vector_layer_geojson(self, layer: UXOLayer):
        """Create vector layer for flight paths from DataFrame using GeoJSON"""
        if not isinstance(layer.data, pd.DataFrame):
            logger.error(f"Vector layer '{layer.name}' requires DataFrame data, got {type(layer.data)}")
            return False

        lat_col, lon_col = self._detect_coordinate_columns(layer.data)
        if not lat_col or not lon_col:
            logger.error(f"Could not detect coordinate columns for vector layer '{layer.name}'")
            return False

        if 'line_id' not in layer.data.columns:
            logger.error(f"'line_id' column is required for vector line layers like '{layer.name}'")
            return False

        features = []
        for line_id, group in layer.data.groupby('line_id'):
            coords = group[[lon_col, lat_col]].values.tolist()
            if len(coords) < 2:
                continue

            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": coords
                },
                "properties": {
                    "line_id": line_id
                }
            }
            features.append(feature)

        if not features:
            logger.warning(f"No valid lines could be processed for layer '{layer.name}'")
            return False

        geojson_data_str = json.dumps({
            "type": "FeatureCollection",
            "features": features
        })
        
        style = layer.style
        layer_name_js = json.dumps(layer.name)
        js_code = f"""
            var geojsonData = {geojson_data_str};
            var options = {{
                style: {{
                    color: '{style.line_color}',
                    weight: {style.line_width},
                    opacity: {style.line_opacity}
                }}
            }};
            var geoJsonLayer = L.geoJson(geojsonData, options);
            window.uxoMapLayers[{layer_name_js}] = geoJsonLayer;
        """

        self.map_widget.page.runJavaScript(js_code)
        
        logger.info(f"Created GeoJSON layer for '{layer.name}' with {len(features)} lines via runJavaScript")
        return True
        
    def _detect_coordinate_columns(self, data: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
        """Auto-detect latitude and longitude columns"""
        columns = data.columns.tolist()
        lat_col = None
        lon_col = None
        
        # Common latitude column names
        lat_keywords = ['lat', 'latitude', 'y', 'northing', 'north']
        lon_keywords = ['lon', 'lng', 'longitude', 'x', 'easting', 'east']
        
        for col in columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in lat_keywords):
                lat_col = col
            elif any(keyword in col_lower for keyword in lon_keywords):
                lon_col = col
                
        return lat_col, lon_col
        
    def _on_layer_removed(self, layer_name: str):
        """Handle layer removal from the JS context"""
        try:
            layer_name_js = json.dumps(layer_name)
            self.map_widget.page.runJavaScript(f"removeUxoLayer({layer_name_js});")
            logger.info(f"Removed layer '{layer_name}'")
        except Exception as e:
            logger.error(f"Failed to handle layer removal '{layer_name}': {e}")
                
    def _on_layer_visibility_changed(self, layer_name: str, is_visible: bool):
        """Handle layer visibility change"""
        try:
            layer_name_js = json.dumps(layer_name)
            if is_visible:
                # If a layer is made visible, it might not have been created in JS yet
                # (e.g., if it was added while a filter was active).
                # The safest approach is to ensure it exists before trying to show it.
                layer = self.layer_manager.get_layer(layer_name)
                if layer:
                    self._create_leaflet_layer(layer)
                    self.map_widget.page.runJavaScript(f"showUxoLayer({layer_name_js});")
                    logger.debug(f"Layer '{layer_name}' made visible")
                    self.zoom_to_visible_layers()
            else:
                self.map_widget.page.runJavaScript(f"hideUxoLayer({layer_name_js});")
                logger.debug(f"Layer '{layer_name}' hidden from map")
        except Exception as e:
            logger.error(f"Failed to change visibility for layer '{layer_name}': {e}")
            import traceback
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            
    def _on_layer_style_changed(self, layer_name: str, new_style: LayerStyle):
        """Handle layer style change by recreating the layer"""
        layer = self.layer_manager.get_layer(layer_name)
        if layer:
            was_visible = layer.is_visible
            # Re-adding the layer will apply the new style.
            # It will be hidden initially, then shown if it was visible before.
            self._on_layer_removed(layer_name)
            self._on_layer_added(layer)
            if was_visible:
                self._on_layer_visibility_changed(layer_name, True)

    def _on_layer_opacity_changed(self, layer_name: str, new_opacity: float):
        """Handle layer opacity change"""
        try:
            layer_name_js = json.dumps(layer_name)
            self.map_widget.page.runJavaScript(f"setUxoLayerOpacity({layer_name_js}, {new_opacity});")
            logger.debug(f"Set opacity for layer '{layer_name}' to {new_opacity}")
        except Exception as e:
            logger.error(f"Failed to change opacity for layer '{layer_name}': {e}")
            
    def center_default(self):
        """Center map on default location (Tarva island)"""
        try:
            if hasattr(self, 'map') and self.map is not None:
                self.map.setView([63.8167, 9.3667], 12)
                logger.info("Map centered on Tarva island")
            else:
                logger.debug("Map not yet initialized, cannot center")
        except Exception as e:
            logger.warning(f"Error centering map: {e}")
        
    def zoom_to_data(self):
        """Zoom to the extent of all data layers"""
        try:
            if not hasattr(self, 'map') or self.map is None:
                logger.debug("Map not yet initialized, cannot zoom to data")
                return
                
            bounds = self.layer_manager.get_global_bounds()
            if bounds:
                # Convert to Leaflet bounds format
                sw = [bounds[1], bounds[0]]  # [lat, lon]
                ne = [bounds[3], bounds[2]]  # [lat, lon]
                self.map.fitBounds([sw, ne])
                logger.info(f"Zoomed to data extent: {bounds}")
            else:
                logger.warning("No data bounds available")
        except Exception as e:
            logger.warning(f"Error zooming to data: {e}")
            
    def zoom_to_layer(self, layer_name: str):
        """Zoom to specific layer extent"""
        try:
            if not hasattr(self, 'map') or self.map is None:
                logger.debug("Map not yet initialized, cannot zoom to layer")
                return
                
            layer = self.layer_manager.get_layer(layer_name)
            if layer and layer.bounds:
                sw = [layer.bounds[1], layer.bounds[0]]
                ne = [layer.bounds[3], layer.bounds[2]]
                self.map.fitBounds([sw, ne])
                logger.info(f"Zoomed to layer '{layer_name}'")
        except Exception as e:
            logger.warning(f"Error zooming to layer '{layer_name}': {e}")
        
    def toggle_measure(self, checked):
        """Toggle measurement tool"""
        if checked:
            self.draw_btn.setChecked(False)
        # TODO: Implement measurement tool
        logger.info(f"Measurement tool: {'enabled' if checked else 'disabled'}")
        
    def toggle_draw(self, checked):
        """Toggle drawing tools"""
        if checked:
            self.measure_btn.setChecked(False)
        # TODO: Implement drawing tools
        logger.info(f"Drawing tools: {'enabled' if checked else 'disabled'}")
        
    def clear_drawings(self):
        """Clear all drawings from map"""
        # TODO: Implement clearing of drawings
        logger.info("Clearing drawings")
        
    def export_map(self):
        """Export current map view"""
        from qtpy.QtWidgets import QFileDialog, QMessageBox
        
        logger.info("Exporting map")
        
        # Get export file path
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Map", 
            "uxo_wizard_map.png",
            "PNG Files (*.png);;All Files (*.*)"
        )
        
        if file_path:
            try:
                # TODO: Implement proper map export
                QMessageBox.information(self, "Export", "Map export functionality coming soon!")
                logger.info(f"Map export requested to: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export map:\n{str(e)}")
                logger.error(f"Map export failed: {str(e)}")
                
    def toggle_fullscreen(self, checked):
        """Toggle fullscreen mode for the map widget"""
        if checked:
            self._original_parent = self.parent()
            self.setParent(None)
            self.showFullScreen()
            self.fullscreen_btn.setText("🔲")
            self.fullscreen_btn.setToolTip("Exit Fullscreen")
            logger.info("Map entered fullscreen mode")
        else:
            if hasattr(self, '_original_parent') and self._original_parent:
                self.showNormal()
                self.setParent(self._original_parent)
                if hasattr(self._original_parent, 'setWidget'):
                    self._original_parent.setWidget(self)
            self.fullscreen_btn.setText("🔳")
            self.fullscreen_btn.setToolTip("Toggle Fullscreen")
            logger.info("Map restored from fullscreen")
            
    def exit_fullscreen(self):
        """Exit fullscreen mode when escape is pressed"""
        if self.isFullScreen():
            self.fullscreen_btn.setChecked(False)
            self.toggle_fullscreen(False)
    
    def add_data_layer(self, name: str, data: pd.DataFrame, layer_type: str = "points"):
        """Add a data layer to the map (compatibility method for main window)"""
        logger.info(f"Adding data layer: {name} ({layer_type})")
        
        # Create UXOLayer from DataFrame
        uxo_layer = self._create_uxo_layer_from_dataframe(name, data, layer_type)
        
        if uxo_layer:
            self.add_layer_realtime(uxo_layer)
        else:
            logger.error(f"Failed to create layer from data: {name}")
            
    def _create_uxo_layer_from_dataframe(self, name: str, data: pd.DataFrame, layer_type: str):
        """Create UXOLayer from DataFrame"""
        from .layer_types import UXOLayer, LayerType, GeometryType, LayerStyle, LayerSource
        
        # Auto-detect coordinate columns
        lat_col, lon_col = self._detect_coordinate_columns(data)
        
        if not lat_col or not lon_col:
            logger.warning("No valid coordinate columns found in data")
            return None
            
        # Calculate bounds
        try:
            lats = data[lat_col].dropna()
            lons = data[lon_col].dropna()
            
            if len(lats) == 0 or len(lons) == 0:
                logger.warning("No valid coordinates in data")
                return None
                
            bounds = [
                float(lons.min()), float(lats.min()),
                float(lons.max()), float(lats.max())
            ]
        except Exception as e:
            logger.error(f"Error calculating bounds: {e}")
            bounds = None
            
        # Create metadata
        metadata = {
            "row_count": len(data),
            "columns": data.columns.tolist(),
            "lat_column": lat_col,
            "lon_column": lon_col,
            "data_type": layer_type
        }
        
        # Create UXOLayer
        uxo_layer = UXOLayer(
            name=name,
            layer_type=LayerType.POINTS if layer_type == "points" else LayerType.VECTOR,
            data=data,
            geometry_type=GeometryType.POINT if layer_type == "points" else GeometryType.MULTIPOINT,
            style=LayerStyle(),
            metadata=metadata,
            source=LayerSource.DATA_VIEWER,
            bounds=bounds
        )
        
        return uxo_layer
        
    def remove_layer(self, name: str):
        """Remove a layer from the map (compatibility method)"""
        self.layer_manager.remove_layer(name)
        
    def center_on_data(self, bounds):
        """Center map on data bounds (compatibility method)"""
        try:
            if not hasattr(self, 'map') or self.map is None:
                logger.debug("Map not yet initialized, cannot center on data")
                return
                
            if bounds and len(bounds) == 4:
                sw = [bounds[1], bounds[0]]  # [lat, lon]
                ne = [bounds[3], bounds[2]]  # [lat, lon]
                self.map.fitBounds([sw, ne])
        except Exception as e:
            logger.warning(f"Error centering on data: {e}")
            
    def set_map_center(self, lat: float, lon: float, zoom: int = 6):
        """Set map center to specific coordinates (compatibility method)"""
        try:
            if hasattr(self, 'map') and self.map is not None:
                self.map.setView([lat, lon], zoom)
            else:
                logger.debug("Map not yet initialized, cannot set center")
        except Exception as e:
            logger.warning(f"Error setting map center: {e}")
        
    def add_marker(self, lat: float, lon: float, popup_text: str = "", color: str = "blue"):
        """Add a single marker to the map (compatibility method)"""
        # Create a simple point layer for the marker
        import pandas as pd
        from .layer_types import UXOLayer, LayerType, GeometryType, LayerStyle, LayerSource
        
        df = pd.DataFrame({
            'lat': [lat],
            'lon': [lon],
            'description': [popup_text]
        })
        
        marker_layer = UXOLayer(
            name=f"Marker at {lat:.4f}, {lon:.4f}",
            layer_type=LayerType.ANNOTATION,
            data=df,
            geometry_type=GeometryType.POINT,
            style=LayerStyle(point_color=color),
            metadata={"marker": True, "popup": popup_text},
            source=LayerSource.ANNOTATION,
            bounds=[lon, lat, lon, lat]
        )
        
        self.add_layer_realtime(marker_layer)

    def showEvent(self, event):
        super().showEvent(event)
        
        # Lazy map creation on first show
        if not self._map_created and self.width() > 0 and self.height() > 0:
            logger.info("Creating map on first show")
            self._map_created = True
            QTimer.singleShot(100, self.setup_map)  # Small delay to ensure widget is fully shown
        else:
            self._kick_leaflet()
        
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._kick_leaflet()

    def _kick_leaflet(self):
        """
        Force Leaflet to re-measure the container **only** when
        the widget already has a size > 0, so map.mapObject is non-null.
        """
        if self.width() > 0 and self.height() > 0:
            QTimer.singleShot(100, self._invalidate_map_size)  # Small delay
            
    def _invalidate_map_size(self):
        """Force map to invalidate its size"""
        try:
            # Check if map exists first (for lazy initialization)
            if not hasattr(self, 'map') or self.map is None:
                logger.debug("Map not yet created, skipping size invalidation")
                return
                
            # Try pyqtlet2's direct method if available
            if hasattr(self.map, 'invalidateSize'):
                self.map.invalidateSize()
                logger.debug("Map invalidated using pyqtlet2 method")
                
            # Try JavaScript fallback
            self.map_widget.page.runJavaScript(
                "if (window.map && window.map.invalidateSize) {"
                "    window.map.invalidateSize(true);"
                "}"
            )
        except Exception as e:
            logger.warning(f"Failed to invalidate map size: {e}")

    def _init_js_layer_manager(self):
        """Initialize JavaScript layer management objects and functions"""
        js_code = """
            // Global store for all created Leaflet layers
            if (!window.uxoMapLayers) {
                window.uxoMapLayers = {};
            }
            
            // Layer group on the map to manage visibility
            if (!window.uxoMapLayerGroup) {
                // Check if map is properly initialized before adding layer group
                if (typeof map !== 'undefined' && map !== null) {
                    window.uxoMapLayerGroup = L.layerGroup().addTo(map);
                } else {
                    console.warn('Map not ready, deferring layer group creation');
                    // Don't use return at top level - just skip the initialization
                }
            }
            
            // --- Core Layer Functions ---
            
            function showUxoLayer(layerName) {
                try {
                    if (!window.uxoMapLayers || !window.uxoMapLayerGroup) {
                        console.warn('Layer management not ready for:', layerName);
                        return;
                    }
                    
                    if (window.uxoMapLayers[layerName]) {
                        if (!window.uxoMapLayerGroup.hasLayer(window.uxoMapLayers[layerName])) {
                            window.uxoMapLayerGroup.addLayer(window.uxoMapLayers[layerName]);
                            console.log('Showing layer:', layerName);
                        }
                    } else {
                        console.warn('Cannot show layer - not found:', layerName);
                    }
                } catch (error) {
                    console.error('Error showing layer:', layerName, error);
                }
            }
            
            function hideUxoLayer(layerName) {
                try {
                    if (!window.uxoMapLayers || !window.uxoMapLayerGroup) {
                        console.warn('Layer management not ready for:', layerName);
                        return;
                    }
                    
                    if (window.uxoMapLayers[layerName]) {
                        if (window.uxoMapLayerGroup.hasLayer(window.uxoMapLayers[layerName])) {
                            window.uxoMapLayerGroup.removeLayer(window.uxoMapLayers[layerName]);
                            console.log('Hiding layer:', layerName);
                        }
                    } else {
                        console.warn('Cannot hide layer - not found:', layerName);
                    }
                } catch (error) {
                    console.error('Error hiding layer:', layerName, error);
                }
            }
            
            function removeUxoLayer(layerName) {
                try {
                    hideUxoLayer(layerName);
                    if (window.uxoMapLayers && window.uxoMapLayers[layerName]) {
                        delete window.uxoMapLayers[layerName];
                        console.log('Removed layer from store:', layerName);
                    }
                } catch (error) {
                    console.error('Error removing layer:', layerName, error);
                }
            }
            
            function setUxoLayerOpacity(layerName, opacity) {
                try {
                    if (!window.uxoMapLayers) {
                        console.warn('Layer management not ready for opacity change:', layerName);
                        return;
                    }
                    
                    if (window.uxoMapLayers[layerName]) {
                        const layer = window.uxoMapLayers[layerName];
                        
                        // For layers with a setOpacity method (ImageOverlay, TileLayer)
                        if (typeof layer.setOpacity === 'function') {
                            layer.setOpacity(opacity);
                        } 
                        // For vector layers (GeoJSON)
                        else if (typeof layer.setStyle === 'function') {
                            layer.setStyle({
                                opacity: opacity,
                                fillOpacity: opacity * 0.8 // Adjust fill opacity relative to main opacity
                            });
                        }
                        // For individual markers that don't have a group setStyle
                        else if (layer.eachLayer) {
                             layer.eachLayer(function(subLayer) {
                                if (typeof subLayer.setOpacity === 'function') {
                                    subLayer.setOpacity(opacity);
                                    if (typeof subLayer.setStyle === 'function') {
                                         subLayer.setStyle({ fillOpacity: opacity * 0.8 });
                                    }
                                } else if (typeof subLayer.setStyle === 'function') {
                                    subLayer.setStyle({
                                        opacity: opacity,
                                        fillOpacity: opacity * 0.8
                                    });
                                }
                            });
                        }
                         else {
                            console.warn('Layer does not support opacity change:', layerName, layer);
                        }
                        console.log(`Set opacity for ${layerName} to ${opacity}`);
                    } else {
                        console.warn('Cannot set opacity - layer not found:', layerName);
                    }
                } catch (error) {
                    console.error('Error setting layer opacity:', layerName, error);
                }
            }
        """
        self.map_widget.page.runJavaScript(js_code)
        logger.debug("JavaScript layer manager initialized")

    def zoom_to_visible_layers(self):
        """Zoom to the extent of all visible data layers"""
        try:
            # Check if map is properly initialized
            if not hasattr(self, 'map') or self.map is None:
                logger.debug("Map not yet initialized, skipping zoom to visible layers")
                return
                
            bounds = self.layer_manager.get_visible_bounds()
            if bounds:
                # Convert to Leaflet bounds format: [[south, west], [north, east]]
                sw = [bounds[1], bounds[0]]
                ne = [bounds[3], bounds[2]]
                
                # Use safe JavaScript execution with error handling
                js_command = f"""
                if (typeof map !== 'undefined' && map && map.fitBounds) {{
                    map.fitBounds([[{sw[0]}, {sw[1]}], [{ne[0]}, {ne[1]}]]);
                    console.log('Zoomed to visible data extent');
                }} else {{
                    console.warn('Map not ready for zoom operation');
                }}
                """
                self.map_widget.page.runJavaScript(js_command)
                logger.info(f"Zoomed to visible data extent: {bounds}")
            else:
                logger.info("No visible layers with bounds to zoom to.")
        except Exception as e:
            logger.warning(f"Error zooming to visible layers: {e}")
