"""
Advanced Map Widget for UXO Wizard - pyqtlet2 based implementation

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

from .layer_types import UXOLayer, LayerType, GeometryType, LayerStyle
from .layer_manager import LayerManager


class AdvancedMapWidget(QWidget):
    """Advanced map widget with real-time layer management using pyqtlet2"""
    
    # Signals
    coordinates_clicked = Signal(float, float)  # lat, lon
    area_selected = Signal(list)  # List of coordinates
    feature_selected = Signal(str, list)  # layer_name, feature_ids
    map_ready = Signal()  # Emitted when map is fully initialized
    
    def __init__(self):
        super().__init__()
        self.layer_manager = LayerManager()
        self.leaflet_layers: Dict[str, any] = {}  # Map layer names to leaflet objects
        self.current_draw_layer = None
        self._map_created = False  # Track if map has been created
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
        self.center_btn.setToolTip("Center on Default Location (Norway)")
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
        """Initialize the pyqtlet2 map with Norwegian defaults"""
        logger.info("Setting up pyqtlet2 map with Norwegian defaults")

        # Create map centered on Norway
        self.map = L.map(self.map_widget)
        self.map.setView([64.5, 11.0], 3)  # Center of Norway, zoom 3 (less zoomed in)
        
        # Add map layers
        self._add_base_layers()
        
        # Add map controls
        self._add_map_controls()
        
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
        
    def add_layer_realtime(self, uxo_layer: UXOLayer):
        """Add layer without HTML reload"""
        logger.info(f"Adding layer in real-time: {uxo_layer.name}")
        
        # Add to layer manager
        self.layer_manager.add_layer(uxo_layer)
        
    def _on_layer_added(self, layer: UXOLayer):
        """Handle layer addition"""
        try:
            # Create appropriate Leaflet layer based on type
            leaflet_layer = self._create_leaflet_layer(layer)
            
            if leaflet_layer is not None:
                try:
                    # Add to map
                    if leaflet_layer._map is None:
                        logger.warning(f"Leaflet layer '{layer.name}' has no map reference")
                        leaflet_layer.addTo(self.map)
                    
                    # Store reference
                    self.leaflet_layers[layer.name] = leaflet_layer
                        
                    logger.info(f"Successfully added layer '{layer.name}' to map")
                    
                    # Auto-zoom to first layer
                    if len(self.leaflet_layers) == 1:
                        self.zoom_to_layer(layer.name)
                        
                except Exception as map_error:
                    logger.error(f"Failed to add layer '{layer.name}' to map: {map_error}")
            else:
                logger.warning(f"Could not create leaflet layer for '{layer.name}' - layer creation returned None")
                    
        except Exception as e:
            logger.error(f"Failed to add layer '{layer.name}': {e}")
            import traceback
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            
    def _create_leaflet_layer(self, layer: UXOLayer):
        """Create appropriate Leaflet layer from UXOLayer"""
        if layer.layer_type == LayerType.POINTS and layer.geometry_type == GeometryType.POINT:
            return self._create_point_layer(layer)
        elif layer.layer_type == LayerType.ANNOTATION and layer.geometry_type == GeometryType.POINT:
            return self._create_point_layer(layer)  # Annotations are just styled points
        elif layer.layer_type == LayerType.RASTER:
            return self._create_raster_layer(layer)
        elif layer.layer_type == LayerType.VECTOR:
            return self._create_vector_layer(layer)
        else:
            logger.warning(f"Unsupported layer type: {layer.layer_type}")
            return None
            
    def _create_point_layer(self, layer: UXOLayer):
        """Create point layer from DataFrame"""
        if not isinstance(layer.data, pd.DataFrame):
            logger.error("Point layer requires DataFrame data")
            return None
            
        # Detect coordinate columns
        lat_col, lon_col = self._detect_coordinate_columns(layer.data)
        if not lat_col or not lon_col:
            logger.error("Could not detect coordinate columns")
            return None
            
        # Create layer group for points
        layer_group = L.layerGroup()
        layer_group.addTo(self.map)
        
        # Add points
        points_added = 0
        for idx, row in layer.data.iterrows():
            try:
                lat = float(row[lat_col])
                lon = float(row[lon_col])
                
                # Create popup content
                popup_html = f"<b>Point {idx}</b><br>"
                for col in layer.data.columns[:5]:  # Show first 5 columns
                    popup_html += f"<b>{col}:</b> {row[col]}<br>"
                if len(layer.data.columns) > 5:
                    popup_html += f"<i>... and {len(layer.data.columns) - 5} more fields</i>"
                
                # Use Leaflet circleMarker for simple point visualization
                marker_style = {
                    'radius': layer.style.point_size,
                    'color': layer.style.point_color,
                    'weight': 1,
                    'opacity': layer.style.point_opacity,
                    'fillColor': layer.style.point_color,
                    'fillOpacity': layer.style.point_opacity
                }
                marker = L.circleMarker([lat, lon], marker_style)
                # Add a label to the marker (show index or a label field if available)
                label_text = None
                if layer.style.show_labels and layer.style.label_field and layer.style.label_field in row:
                    label_text = str(row[layer.style.label_field])
                else:
                    label_text = f"{layer.name} #{idx}"
                marker.bindTooltip(label_text, {"permanent": False, "direction": "top"})
                marker.bindPopup(popup_html)
                marker.addTo(layer_group)
                
                points_added += 1
                
            except (ValueError, TypeError) as e:
                logger.warning(f"Skipping invalid coordinates at row {idx}: {e}")
                continue  # Skip invalid coordinates
                
        logger.info(f"Added {points_added} points to layer '{layer.name}'")
        
        # Update layer bounds if not set
        if not layer.bounds and points_added > 0:
            lats = layer.data[lat_col].dropna()
            lons = layer.data[lon_col].dropna()
            layer.bounds = [
                float(lons.min()), float(lats.min()),
                float(lons.max()), float(lats.max())
            ]
            
        if points_added == 0:
            logger.warning(f"No valid points could be added for layer '{layer.name}'")
            return None
            
        return layer_group
        
    def _create_raster_layer(self, layer: UXOLayer):
        """Create raster layer - placeholder for future implementation"""
        logger.warning("Raster layer support not yet implemented")
        return None
        
    def _create_vector_layer(self, layer: UXOLayer):
        """Create vector layer - placeholder for future implementation"""
        logger.warning("Vector layer support not yet implemented")
        return None
        
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
        """Handle layer removal"""
        if layer_name in self.leaflet_layers:
            try:
                # Remove from map
                self.map.removeLayer(self.leaflet_layers[layer_name])
                
                # Remove reference
                del self.leaflet_layers[layer_name]
                
                logger.info(f"Removed layer '{layer_name}' from map")
                
            except Exception as e:
                logger.error(f"Failed to remove layer '{layer_name}': {e}")
                
    def _on_layer_visibility_changed(self, layer_name: str, is_visible: bool):
        """Handle layer visibility change"""
        try:
            if is_visible:
                # Restore layer
                if layer_name in self.leaflet_layers:
                    leaflet_layer = self.leaflet_layers[layer_name]
                    # If layer already attached, nothing to do
                    if getattr(leaflet_layer, "_map", None) is None:
                        leaflet_layer.addTo(self.map)
                else:
                    # If we don't have a cached leaflet layer, recreate it
                    layer = self.layer_manager.get_layer(layer_name)
                    if layer:
                        self._on_layer_added(layer)
            else:
                # Hide layer
                if layer_name in self.leaflet_layers:
                    leaflet_layer = self.leaflet_layers[layer_name]
                    self.map.removeLayer(leaflet_layer)
                    # Remove from cache so we know to recreate next time
                    del self.leaflet_layers[layer_name]

            logger.debug(f"Layer '{layer_name}' visibility set to {is_visible}")

        except Exception as e:
            logger.error(f"Failed to change visibility for layer '{layer_name}': {e}")
            
    def _on_layer_style_changed(self, layer_name: str, new_style: LayerStyle):
        """Handle layer style change"""
        # For now, recreate the layer with new style
        layer = self.layer_manager.get_layer(layer_name)
        if layer:
            self._on_layer_removed(layer_name)
            self._on_layer_added(layer)
            
    def center_default(self):
        """Center map on default location (Norway)"""
        self.map.setView([64.5, 11.0], 3)
        logger.info("Map centered on Norway")
        
    def zoom_to_data(self):
        """Zoom to the extent of all data layers"""
        bounds = self.layer_manager.get_global_bounds()
        if bounds:
            # Convert to Leaflet bounds format
            sw = [bounds[1], bounds[0]]  # [lat, lon]
            ne = [bounds[3], bounds[2]]  # [lat, lon]
            self.map.fitBounds([sw, ne])
            logger.info(f"Zoomed to data extent: {bounds}")
        else:
            logger.warning("No data bounds available")
            
    def zoom_to_layer(self, layer_name: str):
        """Zoom to specific layer extent"""
        layer = self.layer_manager.get_layer(layer_name)
        if layer and layer.bounds:
            sw = [layer.bounds[1], layer.bounds[0]]
            ne = [layer.bounds[3], layer.bounds[2]]
            self.map.fitBounds([sw, ne])
            logger.info(f"Zoomed to layer '{layer_name}'")
            
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
