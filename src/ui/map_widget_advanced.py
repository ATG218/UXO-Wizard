"""
Combined Advanced Map Widget with Layer Control Panel
"""

import os
# Make sure qtpy and pyqtlet2 use the same Qt binding
os.environ["QT_API"] = "pyside6"

from qtpy.QtWidgets import QWidget, QHBoxLayout, QSplitter
from qtpy.QtCore import Qt, Signal
from typing import Optional
import pandas as pd
from loguru import logger

from .map.advanced_map_widget import AdvancedMapWidget
from .map.layer_control_panel import LayerControlPanel
from .map.layer_types import UXOLayer, LayerType, GeometryType, LayerStyle, LayerSource


class MapWidgetAdvanced(QWidget):
    """Combined map widget with integrated layer control panel"""
    
    # Re-export signals from map widget for compatibility
    coordinates_clicked = Signal(float, float)
    area_selected = Signal(list)
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.connect_signals()
        
    def setup_ui(self):
        """Initialize the UI with map and layer panel"""
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        
        # Layer control panel (left)
        self.map_widget = AdvancedMapWidget()
        self.layer_panel = LayerControlPanel(self.map_widget.layer_manager)
        self.layer_panel.setMaximumWidth(300)
        
        # Add to splitter
        splitter.addWidget(self.layer_panel)
        splitter.addWidget(self.map_widget)
        splitter.setStretchFactor(0, 0)  # Layer panel doesn't stretch
        splitter.setStretchFactor(1, 1)  # Map stretches
        
        layout.addWidget(splitter)
        self.setLayout(layout)
        
    def connect_signals(self):
        """Connect internal signals"""
        # Forward map signals
        self.map_widget.coordinates_clicked.connect(self.coordinates_clicked)
        self.map_widget.area_selected.connect(self.area_selected)
        
        # Connect layer panel to map
        self.layer_panel.zoom_to_layer.connect(self.map_widget.zoom_to_layer)
        
    def add_data_layer(self, name: str, data: pd.DataFrame, layer_type: str = "points"):
        """Add a data layer to the map (compatibility method)"""
        logger.info(f"Adding data layer: {name} ({layer_type})")
        
        # Create UXOLayer from DataFrame
        uxo_layer = self._create_uxo_layer_from_dataframe(name, data, layer_type)
        
        if uxo_layer:
            self.map_widget.add_layer_realtime(uxo_layer)
        else:
            logger.error(f"Failed to create layer from data: {name}")
            
    def _create_uxo_layer_from_dataframe(self, name: str, data: pd.DataFrame, layer_type: str) -> Optional[UXOLayer]:
        """Create UXOLayer from DataFrame"""
        # Auto-detect coordinate columns
        lat_col, lon_col = self.detect_coordinate_columns(data)
        
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
        
    def detect_coordinate_columns(self, data: pd.DataFrame):
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
        
    def remove_layer(self, name: str):
        """Remove a layer from the map"""
        self.map_widget.layer_manager.remove_layer(name)
        
    def center_on_data(self, bounds):
        """Center map on data bounds"""
        if bounds and len(bounds) == 4:
            sw = [bounds[1], bounds[0]]  # [lat, lon]
            ne = [bounds[3], bounds[2]]  # [lat, lon]
            self.map_widget.map.fitBounds([sw, ne])
            
    def set_map_center(self, lat: float, lon: float, zoom: int = 6):
        """Set map center to specific coordinates"""
        self.map_widget.map.setView([lat, lon], zoom)
        
    def add_marker(self, lat: float, lon: float, popup_text: str = "", color: str = "blue"):
        """Add a single marker to the map"""
        # Create a simple point layer for the marker
        import pandas as pd
        
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
        
        self.map_widget.add_layer_realtime(marker_layer)
        
    def cleanup(self):
        """Clean up resources"""
        # pyqtlet2 handles its own cleanup
        pass 

    def add_layer_realtime(self, uxo_layer: UXOLayer):
        """Add UXOLayer to the map in real-time (forwards to underlying map widget)"""
        logger.info(f"MapWidgetAdvanced: Adding layer '{uxo_layer.name}' to map")
        self.map_widget.add_layer_realtime(uxo_layer)
    
    def zoom_to_layer(self, layer_name: str):
        """Zoom to specific layer (forwards to underlying map widget)"""
        self.map_widget.zoom_to_layer(layer_name)
    
    def zoom_to_data(self):
        """Zoom to all data (forwards to underlying map widget)"""
        self.map_widget.zoom_to_data()
    
    def center_default(self):
        """Center on default location (forwards to underlying map widget)"""
        self.map_widget.center_default() 