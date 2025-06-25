"""
Map Widget for UXO Wizard - Interactive map display with Norwegian focus

Features:
- Primary: Kartverket (Norwegian Mapping Authority) topographic maps
- Backup: OpenStreetMap and Satellite imagery
- Interactive tools: Drawing, measuring, data plotting
- Export capabilities for professional reports
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QToolBar, QComboBox, 
    QToolButton, QLabel, QSlider
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWebEngineWidgets import QWebEngineView
import folium
from folium.plugins import Draw, MeasureControl, Fullscreen
import tempfile
import os
from loguru import logger


class MapWidget(QWidget):
    """Interactive map widget using Folium and QWebEngine"""
    
    # Signals
    coordinates_clicked = Signal(float, float)  # lat, lon
    area_selected = Signal(list)  # List of coordinates
    
    def __init__(self):
        super().__init__()
        self.temp_html = None
        self.current_map = None
        self.layers = {}
        self.setup_ui()
        self.create_default_map()
        
    def setup_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Toolbar
        toolbar = QToolBar()
        
        # Layer and navigation controls (top priority)
        self.center_btn = QToolButton()
        self.center_btn.setText("🎯")
        self.center_btn.setToolTip("Center on Default Location")
        self.center_btn.clicked.connect(self.center_default)
        toolbar.addWidget(self.center_btn)
        
        # Refresh button
        self.refresh_btn = QToolButton()
        self.refresh_btn.setText("🔄")
        self.refresh_btn.setToolTip("Refresh Map")
        self.refresh_btn.clicked.connect(self.refresh_map)
        toolbar.addWidget(self.refresh_btn)
        
        # Fullscreen button
        self.fullscreen_btn = QToolButton()
        self.fullscreen_btn.setText("🔳")
        self.fullscreen_btn.setToolTip("Toggle Fullscreen")
        self.fullscreen_btn.setCheckable(True)
        self.fullscreen_btn.clicked.connect(self.toggle_fullscreen)
        toolbar.addWidget(self.fullscreen_btn)
        
        toolbar.addSeparator()
        
        # Drawing and measurement tools (secondary)
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
        
        # Export tools (last)
        self.export_btn = QToolButton()
        self.export_btn.setText("💾")
        self.export_btn.setToolTip("Export Map")
        self.export_btn.clicked.connect(self.export_map)
        toolbar.addWidget(self.export_btn)
        
        # Web view with proper settings
        self.web_view = QWebEngineView()
        
        # Enable JavaScript and local file access
        settings = self.web_view.settings()
        from PySide6.QtWebEngineCore import QWebEngineSettings
        settings.setAttribute(QWebEngineSettings.JavascriptEnabled, True)
        settings.setAttribute(QWebEngineSettings.LocalContentCanAccessFileUrls, True)
        settings.setAttribute(QWebEngineSettings.LocalContentCanAccessRemoteUrls, True)
        
        # Layout
        layout.addWidget(toolbar)
        layout.addWidget(self.web_view)
        self.setLayout(layout)
        
        # Add escape key shortcut for exiting fullscreen
        self.escape_shortcut = QShortcut(QKeySequence(Qt.Key_Escape), self)
        self.escape_shortcut.activated.connect(self.exit_fullscreen)
        
    def create_default_map(self):
        """Create default map with Kartverket as primary layer"""
        logger.info("Creating Norwegian UXO operations map")
        
        # Create map centered on Norway
        self.current_map = folium.Map(
            location=[64.5, 11.0],  # Center of Norway
            zoom_start=5,
            control_scale=True,
            tiles=None  # Custom tiles only
        )
        
        # Add all available map layers (Kartverket last = default)
        self._add_all_map_layers()
        
        # Add UXO operations marker
        self._add_welcome_marker()
        
        # Add interactive tools
        self._add_map_tools()
        
        # Add layer switcher
        folium.LayerControl(position='topright').add_to(self.current_map)
        
        self.update_map_display()
        logger.info("Norwegian UXO map ready")
    
    def _add_all_map_layers(self):
        """Add all map layers in priority order (Kartverket last = default)"""
        # Backup layers first (will be hidden by default)
        self._add_backup_layers()
        
        # Primary: Kartverket last = default visible layer
        self._add_kartverket_primary()
    
    def _add_kartverket_primary(self):
        """Add Kartverket as the primary/default map layer"""
        try:
            folium.raster_layers.WmsTileLayer(
                url='https://wms.geonorge.no/skwms1/wms.topo4.graatone',
                name='Kartverket (Primary)',
                layers='topo4graatone_WMS',
                fmt='image/png',
                transparent=False,
                version='1.3.0',
                attr='© Kartverket - Norwegian Mapping Authority',
                overlay=False,
                control=True,
                show=True  # Explicitly visible by default
            ).add_to(self.current_map)
            logger.debug("Kartverket primary layer added as default visible layer")
        except Exception as e:
            logger.warning(f"Kartverket primary layer failed: {e}")
    
    def _add_backup_layers(self):
        """Add international backup layers (hidden by default)"""
        try:
            # Backup 1: OpenStreetMap (hidden by default)
            folium.TileLayer(
                tiles='OpenStreetMap',
                name='OpenStreetMap',
                attr='© OpenStreetMap contributors',
                control=True,
                show=False  # Hidden by default
            ).add_to(self.current_map)
            
            # Backup 2: Satellite imagery (hidden by default)
            folium.TileLayer(
                tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                attr='© Esri, Maxar, Earthstar Geographics',
                name='Satellite',
                control=True,
                show=False  # Hidden by default
            ).add_to(self.current_map)
            
            logger.debug("Backup layers added (hidden by default)")
        except Exception as e:
            logger.warning(f"Backup layers failed: {e}")
    
    def _add_welcome_marker(self):
        """Add informational marker for UXO operations"""
        folium.Marker(
            location=[64.5, 11.0],
            popup=folium.Popup("""
                <b>🗺️ UXO Wizard - Norway</b><br>
                <strong>Primary:</strong> Kartverket official mapping<br>
                <strong>Backups:</strong> OpenStreetMap, Satellite<br>
                <hr>
                <strong>Features:</strong><br>
                • Import CSV coordinates<br>
                • Mark UXO survey areas<br>
                • Measure distances/areas<br>
                • Export professional maps
            """, max_width=300),
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(self.current_map)
    
    def _add_map_tools(self):
        """Add interactive drawing and measurement tools"""
        try:
            # Add drawing tools positioned below layer control
            Draw(
                position='bottomright'  # Position below layer control
            ).add_to(self.current_map)
            
            MeasureControl(
                position='bottomleft'  # Position with drawing tools
            ).add_to(self.current_map)
            
            self.current_map.add_child(folium.LatLngPopup())
        except Exception as e:
            logger.warning(f"Some map tools unavailable: {e}")
    
    def update_map_display(self):
        """Update the web view with current map"""
        if self.current_map:
            try:
                # Save to temporary HTML file
                if self.temp_html:
                    try:
                        os.unlink(self.temp_html)
                    except:
                        pass
                        
                with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
                    self.temp_html = f.name
                    
                    # Get the HTML content and ensure CDN links work
                    html_content = self.current_map.get_root().render()
                    
                    # Create a complete HTML document with proper headers
                    full_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>UXO Wizard Map</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        html, body {{
            height: 100%;
            margin: 0;
            padding: 0;
        }}
        #map {{
            height: 100%;
            width: 100%;
        }}
    </style>
</head>
<body>
    {html_content}
</body>
</html>
"""
                    f.write(full_html)
                    
                # Load in web view with proper URL
                from PySide6.QtCore import QUrl
                self.web_view.load(QUrl.fromLocalFile(self.temp_html))
                logger.debug(f"Map updated: {self.temp_html}")
                    
            except Exception as e:
                logger.error(f"Error updating map display: {e}")
            
    def add_data_layer(self, name, data, layer_type="points"):
        """Add a data layer to the map"""
        logger.info(f"Adding layer: {name} ({layer_type})")
        
        # Create layer group
        layer_group = folium.FeatureGroup(name=name)
        
        if layer_type == "points":
            # Auto-detect coordinate columns
            lat_col, lon_col = self.detect_coordinate_columns(data)
            
            if lat_col and lon_col:
                points_added = 0
                for idx, row in data.iterrows():
                    try:
                        lat = float(row[lat_col])
                        lon = float(row[lon_col])
                        
                        # Create popup with data information
                        popup_html = f"<b>Point {idx}</b><br>"
                        for col in data.columns[:5]:  # Show first 5 columns
                            popup_html += f"{col}: {row[col]}<br>"
                        if len(data.columns) > 5:
                            popup_html += f"... and {len(data.columns) - 5} more columns"
                            
                        folium.CircleMarker(
                            location=[lat, lon],
                            radius=6,
                            popup=folium.Popup(popup_html, max_width=250),
                            color='blue',
                            fill=True,
                            fillColor='lightblue',
                            fillOpacity=0.7
                        ).add_to(layer_group)
                        points_added += 1
                        
                    except (ValueError, TypeError):
                        continue  # Skip invalid coordinates
                        
                logger.info(f"Added {points_added} points to map layer")
                
                # Auto-center map on data if it's the first layer
                if len(self.layers) == 0 and points_added > 0:
                    self.center_on_data_layer(data, lat_col, lon_col)
                    
            else:
                logger.warning("No valid coordinate columns found in data")
                
        elif layer_type == "heatmap":
            # TODO: Implement heatmap layer using folium.plugins.HeatMap
            pass
            
        elif layer_type == "contour":
            # TODO: Implement contour layer
            pass
            
        # Add to map and store reference
        layer_group.add_to(self.current_map)
        self.layers[name] = layer_group
        
        # Layer is automatically added to folium LayerControl
        
        # Update map display
        self.update_map_display()
    
    def detect_coordinate_columns(self, data):
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
    
    def center_on_data_layer(self, data, lat_col, lon_col):
        """Center map on the extent of data points"""
        try:
            lats = data[lat_col].dropna()
            lons = data[lon_col].dropna()
            
            if len(lats) > 0 and len(lons) > 0:
                # Calculate bounds
                min_lat, max_lat = lats.min(), lats.max()
                min_lon, max_lon = lons.min(), lons.max()
                
                # Calculate center and zoom
                center_lat = (min_lat + max_lat) / 2
                center_lon = (min_lon + max_lon) / 2
                
                # Fit bounds
                bounds = [[min_lat, min_lon], [max_lat, max_lon]]
                self.current_map.fit_bounds(bounds, padding=[20, 20])
                
                logger.info(f"Centered map on data: {center_lat:.4f}, {center_lon:.4f}")
                
        except Exception as e:
            logger.error(f"Error centering on data: {str(e)}")
        
    def remove_layer(self, name):
        """Remove a layer from the map"""
        if name in self.layers:
            # TODO: Implement layer removal
            del self.layers[name]
            logger.info(f"Removed layer: {name}")
            
    def center_default(self):
        """Center map on default location (Norway)"""
        if self.current_map:
            # Create new map centered on default location
            self.create_default_map()
            logger.info("Map centered on Norway")
    
    def refresh_map(self):
        """Refresh the current map display"""
        if self.current_map:
            self.update_map_display()
            logger.info("Map refreshed")
    
    def set_map_center(self, lat, lon, zoom=10):
        """Set map center to specific coordinates"""
        try:
            if self.current_map:
                # Create new map with specified center
                self.current_map = folium.Map(
                    location=[lat, lon],
                    zoom_start=zoom,
                    control_scale=True,
                    tiles=None  # Custom tiles only
                )
                
                # Add all map layers (Kartverket last)
                self._add_all_map_layers()
                
                # Re-add interactive tools
                self._add_map_tools()
                
                # Add layer control
                folium.LayerControl(position='topright').add_to(self.current_map)
                
                # Re-add existing data layers
                for name, layer in self.layers.items():
                    try:
                        layer.add_to(self.current_map)
                    except:
                        pass
                
                self.update_map_display()
                logger.info(f"Map centered on: {lat}, {lon} (zoom: {zoom})")
        except Exception as e:
            logger.error(f"Error setting map center: {e}")
        
    def change_opacity(self, value):
        """Change opacity of selected layer"""
        # Note: Opacity control removed - use folium LayerControl instead
        pass
            
    def toggle_measure(self, checked):
        """Toggle measurement tool"""
        if checked:
            self.draw_btn.setChecked(False)
        # Measurement is always available via the MeasureControl plugin
        
    def toggle_draw(self, checked):
        """Toggle drawing tools"""
        if checked:
            self.measure_btn.setChecked(False)
        # Drawing is always available via the Draw plugin
        
    def clear_drawings(self):
        """Clear all drawings from map"""
        # TODO: Implement clearing of drawings
        logger.info("Clearing drawings")
        
    def export_map(self):
        """Export current map view"""
        from PySide6.QtWidgets import QFileDialog, QMessageBox
        
        logger.info("Exporting map")
        
        if not self.current_map:
            QMessageBox.warning(self, "Export Error", "No map to export!")
            return
        
        # Get export file path
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Map",
            "uxo_wizard_map.html",
            "HTML Files (*.html);;All Files (*.*)"
        )
        
        if file_path:
            try:
                self.current_map.save(file_path)
                QMessageBox.information(self, "Export Success", f"Map exported to:\n{file_path}")
                logger.info(f"Map exported to: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export map:\n{str(e)}")
                logger.error(f"Map export failed: {str(e)}")
        
    def center_on_data(self, bounds):
        """Center map on data bounds"""
        if self.current_map and bounds:
            self.current_map.fit_bounds(bounds)
            self.update_map_display()
            
    def add_marker(self, lat, lon, popup_text="", color="blue"):
        """Add a single marker to the map"""
        folium.Marker(
            location=[lat, lon],
            popup=popup_text,
            icon=folium.Icon(color=color)
        ).add_to(self.current_map)
        self.update_map_display()
        
    def cleanup(self):
        """Clean up temporary files"""
        if self.temp_html and os.path.exists(self.temp_html):
            try:
                os.unlink(self.temp_html)
            except:
                pass 

    def toggle_fullscreen(self, checked):
        """Toggle fullscreen mode for the map widget"""
        if checked:
            # Store the current parent for restoration
            self._original_parent = self.parent()
            
            # Make this widget fullscreen
            self.setParent(None)
            self.showFullScreen()
            self.fullscreen_btn.setText("🔲")
            self.fullscreen_btn.setToolTip("Exit Fullscreen")
            logger.info("Map entered fullscreen mode")
        else:
            # Restore to original parent
            if hasattr(self, '_original_parent') and self._original_parent:
                self.showNormal()
                self.setParent(self._original_parent)
                
                # Re-add to the parent's layout if it's a dock widget
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