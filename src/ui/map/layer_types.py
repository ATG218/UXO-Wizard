"""
Data structures for the advanced map layer system
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Union, Dict, Any, Optional, List
import pandas as pd
import numpy as np


class LayerType(Enum):
    """Types of layers supported"""
    POINTS = "points"
    RASTER = "raster"
    VECTOR = "vector"
    PROCESSED = "processed"
    ANNOTATION = "annotation"


class GeometryType(Enum):
    """Geometry types for layers"""
    POINT = "point"
    LINE = "line"
    POLYGON = "polygon"
    RASTER = "raster"
    MULTIPOINT = "multipoint"


class LayerSource(Enum):
    """Source of layer data"""
    DATA_VIEWER = "data_viewer"
    PROCESSING = "processing"
    IMPORT = "import"
    ANNOTATION = "annotation"


@dataclass
class LayerStyle:
    """Styling configuration for layers"""
    # Point style
    point_color: str = "#0066CC"
    point_size: int = 6
    point_opacity: float = 0.8
    point_symbol: str = "circle"  # circle, square, triangle, marker
    
    # Line style
    line_color: str = "#FF6600"
    line_width: int = 2
    line_opacity: float = 1.0
    line_style: str = "solid"  # solid, dashed, dotted
    
    # Fill style
    fill_color: str = "#66CC00"
    fill_opacity: float = 0.5
    
    # Labels
    show_labels: bool = False
    label_field: Optional[str] = None
    label_size: int = 10
    
    # Clustering
    enable_clustering: bool = True
    cluster_distance: int = 50
    
    # Advanced
    use_graduated_colors: bool = False
    color_field: Optional[str] = None
    color_ramp: Optional[List[str]] = None


@dataclass
class UXOLayer:
    """Unified layer representation for all data types"""
    name: str
    layer_type: LayerType
    data: Union[pd.DataFrame, np.ndarray, dict]
    geometry_type: GeometryType
    style: LayerStyle
    metadata: Dict[str, Any]
    source: LayerSource
    is_visible: bool = True
    opacity: float = 1.0
    z_index: int = 0
    
    # Coordinate system info
    crs: str = "EPSG:4326"
    bounds: Optional[List[float]] = None  # [min_x, min_y, max_x, max_y]
    
    # Processing lineage
    parent_layer: Optional[str] = None
    processing_history: List[str] = field(default_factory=list)
    
    # Norwegian-specific
    utm_zone: Optional[int] = None  # 32, 33, 34, or 35 for Norway
    
    def __post_init__(self):
        """Validate and set defaults after initialization"""
        if self.style is None:
            self.style = LayerStyle()
            
        # Auto-detect UTM zone for Norwegian data if not set
        if self.utm_zone is None and self.bounds:
            lon_center = (self.bounds[0] + self.bounds[2]) / 2
            if 6 <= lon_center < 12:
                self.utm_zone = 32
            elif 12 <= lon_center < 18:
                self.utm_zone = 33
            elif 18 <= lon_center < 24:
                self.utm_zone = 34
            elif 24 <= lon_center < 30:
                self.utm_zone = 35


# Supported coordinate systems for Norway
NORWEGIAN_CRS = {
    "EPSG:4326": "WGS84 Geographic",
    "EPSG:4258": "ETRS89 Geographic", 
    "EPSG:25832": "ETRS89 / UTM zone 32N",
    "EPSG:25833": "ETRS89 / UTM zone 33N",
    "EPSG:25834": "ETRS89 / UTM zone 34N",
    "EPSG:25835": "ETRS89 / UTM zone 35N",
    "EPSG:5972": "NN2000 height",
    "EPSG:5973": "NN54 height"
} 