"""
Map Widget for UXO Wizard - Interactive map display
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QToolBar, QComboBox, 
    QToolButton, QLabel, QSlider
)
from PySide6.QtCore import Qt, Signal
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
        
        # Base map selector
        self.basemap_combo = QComboBox()
        self.basemap_combo.addItems([
            "OpenStreetMap",
            "Satellite",
            "Terrain",
            "Dark Mode"
        ])
        self.basemap_combo.currentTextChanged.connect(self.change_basemap)
        toolbar.addWidget(QLabel("Base Map:"))
        toolbar.addWidget(self.basemap_combo)
        
        toolbar.addSeparator()
        
        # Layer visibility controls
        self.layer_combo = QComboBox()
        self.layer_combo.setMinimumWidth(150)
        toolbar.addWidget(QLabel("Layers:"))
        toolbar.addWidget(self.layer_combo)
        
        # Opacity slider
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(100)
        self.opacity_slider.setMaximumWidth(100)
        self.opacity_slider.valueChanged.connect(self.change_opacity)
        toolbar.addWidget(QLabel("Opacity:"))
        toolbar.addWidget(self.opacity_slider)
        
        toolbar.addSeparator()
        
        # Tools
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
        
        # Export button
        self.export_btn = QToolButton()
        self.export_btn.setText("💾")
        self.export_btn.setToolTip("Export Map")
        self.export_btn.clicked.connect(self.export_map)
        toolbar.addWidget(self.export_btn)
        
        # Web view
        self.web_view = QWebEngineView()
        
        # Layout
        layout.addWidget(toolbar)
        layout.addWidget(self.web_view)
        self.setLayout(layout)
        
    def create_default_map(self):
        """Create a default map view"""
        # Center on a default location (can be updated based on data)
        self.current_map = folium.Map(
            location=[0, 0],
            zoom_start=2,
            control_scale=True,
            prefer_canvas=True
        )
        
        # Add plugins
        Draw(export=True).add_to(self.current_map)
        MeasureControl().add_to(self.current_map)
        Fullscreen().add_to(self.current_map)
        
        # Add coordinate display on click
        self.current_map.add_child(folium.LatLngPopup())
        
        self.update_map_display()
        
    def update_map_display(self):
        """Update the web view with current map"""
        if self.current_map:
            # Save to temporary HTML file
            if self.temp_html:
                try:
                    os.unlink(self.temp_html)
                except:
                    pass
                    
            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
                self.temp_html = f.name
                self.current_map.save(f.name)
                
            # Load in web view
            self.web_view.load(f"file:///{self.temp_html}")
            logger.debug(f"Map updated: {self.temp_html}")
            
    def add_data_layer(self, name, data, layer_type="points"):
        """Add a data layer to the map"""
        logger.info(f"Adding layer: {name} ({layer_type})")
        
        # Create layer group
        layer_group = folium.FeatureGroup(name=name)
        
        if layer_type == "points":
            # Add point markers
            for idx, row in data.iterrows():
                if 'latitude' in row and 'longitude' in row:
                    folium.CircleMarker(
                        location=[row['latitude'], row['longitude']],
                        radius=5,
                        popup=f"Point {idx}",
                        fill=True
                    ).add_to(layer_group)
                    
        elif layer_type == "heatmap":
            # TODO: Implement heatmap layer
            pass
            
        elif layer_type == "contour":
            # TODO: Implement contour layer
            pass
            
        # Add to map and store reference
        layer_group.add_to(self.current_map)
        self.layers[name] = layer_group
        
        # Update layer combo
        self.layer_combo.addItem(name)
        
        # Update map display
        self.update_map_display()
        
    def remove_layer(self, name):
        """Remove a layer from the map"""
        if name in self.layers:
            # TODO: Implement layer removal
            del self.layers[name]
            logger.info(f"Removed layer: {name}")
            
    def change_basemap(self, basemap_name):
        """Change the base map tiles"""
        logger.info(f"Changing basemap to: {basemap_name}")
        
        # Map basemap names to tile providers
        tile_providers = {
            "OpenStreetMap": "OpenStreetMap",
            "Satellite": "Stamen Terrain",
            "Terrain": "Stamen Terrain",
            "Dark Mode": "CartoDB dark_matter"
        }
        
        # TODO: Implement basemap change
        # This requires recreating the map with new tiles
        
    def change_opacity(self, value):
        """Change opacity of selected layer"""
        layer_name = self.layer_combo.currentText()
        if layer_name and layer_name in self.layers:
            # TODO: Implement opacity change
            logger.debug(f"Changing opacity of {layer_name} to {value}%")
            
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
        # TODO: Implement map export
        logger.info("Exporting map")
        
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