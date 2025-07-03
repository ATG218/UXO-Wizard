"""
Project Schema - Data structures for .uxo project files
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import json

from ..ui.map.layer_types import UXOLayer, LayerType, GeometryType, LayerStyle


@dataclass
class LayerVisualState:
    """Visual state of a single layer on the map"""
    layer_name: str
    is_visible: bool = True
    opacity: float = 1.0
    z_index: int = 0
    # Could add more visual properties in the future:
    # style_overrides: Optional[Dict[str, Any]] = None
    

@dataclass 
class MapState:
    """Current map view state including layer visual states"""
    center_lat: float = 63.8167  # Tarva island default
    center_lon: float = 9.3667
    zoom_level: int = 12
    base_layer: str = "OpenStreetMap"
    projection: str = "EPSG:4326"
    
    # View extent
    extent_bounds: Optional[List[float]] = None  # [min_x, min_y, max_x, max_y]
    
    # Layer visual states
    layer_visual_states: List[LayerVisualState] = field(default_factory=list)


@dataclass
class DataViewerTabState:
    """State of an individual data viewer tab"""
    file_path: str
    tab_title: str
    current_column_filter: str = "All columns"
    search_text: str = ""
    selected_rows: List[int] = field(default_factory=list)
    
    
@dataclass
class DataViewerState:
    """Complete data viewer state"""
    open_tabs: List[DataViewerTabState] = field(default_factory=list)
    active_tab_index: int = 0
    has_welcome_tab: bool = True  # Whether welcome tab is shown
    

@dataclass
class ProcessingStep:
    """Individual processing operation record"""
    timestamp: datetime
    operation: str  # e.g., "magnetic_anomaly_detection"
    parameters: Dict[str, Any]
    input_layers: List[str]  # Layer names used as input
    output_layers: List[str]  # Layer names created
    success: bool
    error_message: Optional[str] = None


@dataclass
class UXOProject:
    """Complete project state representation"""
    # Project metadata
    name: str
    created: datetime = field(default_factory=datetime.now)
    modified: datetime = field(default_factory=datetime.now)
    version: str = "1.0"
    description: str = ""
    
    # Project content
    layers: List[UXOLayer] = field(default_factory=list)
    map_state: MapState = field(default_factory=MapState)
    data_viewer_state: DataViewerState = field(default_factory=DataViewerState)
    processing_history: List[ProcessingStep] = field(default_factory=list)
    
    # File organization
    working_directory: Optional[Path] = None
    
    # Layer organization
    layer_groups: Dict[str, List[str]] = field(default_factory=lambda: {
        "Survey Data": [],
        "Magnetic Processing": [],
        "GPR Processing": [],
        "Gamma Processing": [],
        "Multispectral Processing": [],
        "Annotations": [],
        "Other": []
    })
    layer_order: List[str] = field(default_factory=list)
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_modified(self):
        """Update the modified timestamp"""
        self.modified = datetime.now()
    
    def add_layer(self, layer: UXOLayer, group: Optional[str] = None):
        """Add layer to project"""
        self.layers.append(layer)
        self.layer_order.append(layer.name)
        
        # Auto-assign to group if not specified
        if group is None:
            group = self._auto_assign_group(layer)
        
        if group in self.layer_groups:
            self.layer_groups[group].append(layer.name)
        else:
            self.layer_groups["Other"].append(layer.name)
        
        self.update_modified()
    
    def remove_layer(self, layer_name: str):
        """Remove layer from project"""
        # Remove from layers list
        self.layers = [l for l in self.layers if l.name != layer_name]
        
        # Remove from order
        if layer_name in self.layer_order:
            self.layer_order.remove(layer_name)
            
        # Remove from groups
        for group_layers in self.layer_groups.values():
            if layer_name in group_layers:
                group_layers.remove(layer_name)
        
        self.update_modified()
    
    def _auto_assign_group(self, layer: UXOLayer) -> str:
        """Auto-assign layer to appropriate group"""
        if layer.processing_history:
            last_process = layer.processing_history[-1].lower()
            if "magnetic" in last_process:
                return "Magnetic Processing"
            elif "gpr" in last_process:
                return "GPR Processing"
            elif "gamma" in last_process:
                return "Gamma Processing"
            elif "multispectral" in last_process:
                return "Multispectral Processing"
                
        if layer.source.value == "data_viewer":
            return "Survey Data"
        elif layer.source.value == "annotation":
            return "Annotations"
            
        return "Other"


@dataclass
class UXOProjectMetadata:
    """Lightweight project metadata for file headers"""
    name: str
    version: str
    created: datetime
    modified: datetime
    layer_count: int
    file_size_mb: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "name": self.name,
            "version": self.version,
            "created": self.created.isoformat(),
            "modified": self.modified.isoformat(),
            "layer_count": self.layer_count,
            "file_size_mb": self.file_size_mb
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UXOProjectMetadata":
        """Create from dictionary"""
        return cls(
            name=data["name"],
            version=data["version"],
            created=datetime.fromisoformat(data["created"]),
            modified=datetime.fromisoformat(data["modified"]),
            layer_count=data["layer_count"],
            file_size_mb=data["file_size_mb"]
        ) 