# UXO Wizard Map Integration Plan
## pyqtlet2 Implementation with Advanced Layer Management

**Document Version**: 1.0  
**Date**: January 2025  
**Author**: UXO Wizard Development Team  

---

## Executive Summary

This document outlines the implementation plan for replacing the current Folium-based mapping system with pyqtlet2 to enable dynamic layer management without HTML reloads. The new system will support both point-based survey data and raster processing results with QGIS-style layer controls.

## Current Architecture Analysis

### Existing Components
- **DataViewer**: Handles CSV/Excel data loading with metadata parsing
- **MapWidget**: Currently uses Folium with HTML reload limitations  
- **ProcessingPipeline**: Generates processed datasets from magnetic/GPR/gamma/multispectral data
- **ProcessingResult**: Contains processed data with metadata and export capabilities

### Current Limitations
1. **HTML Reload Bottleneck**: Every layer addition requires full map regeneration
2. **Static Layer Management**: No real-time layer toggle/styling capabilities
3. **Limited Interactivity**: No bidirectional communication between map and data
4. **Performance Issues**: Large datasets cause significant lag during map updates

## New Architecture Design

### Core Components

#### 1. **UXOLayer** - Unified Layer Data Type
```python
@dataclass
class UXOLayer:
    """Unified layer representation for all data types"""
    name: str
    layer_type: LayerType  # POINTS, RASTER, VECTOR, PROCESSED
    data: Union[pd.DataFrame, np.ndarray, dict]
    geometry_type: GeometryType  # POINT, LINE, POLYGON, RASTER
    style: LayerStyle
    metadata: Dict[str, Any]
    source: LayerSource  # DATA_VIEWER, PROCESSING, IMPORT
    is_visible: bool = True
    opacity: float = 1.0
    z_index: int = 0
    
    # Coordinate system info
    crs: str = "EPSG:4326"
    bounds: Optional[List[float]] = None  # [min_x, min_y, max_x, max_y]
    
    # Processing lineage
    parent_layer: Optional[str] = None
    processing_history: List[str] = field(default_factory=list)
```

#### 2. **LayerManager** - Central Layer Registry
```python
class LayerManager(QObject):
    """Manages all map layers with hierarchical organization"""
    
    # Signals
    layer_added = Signal(UXOLayer)
    layer_removed = Signal(str)  # layer_name
    layer_visibility_changed = Signal(str, bool)
    layer_style_changed = Signal(str, LayerStyle)
    layer_selected = Signal(str)
    
    def __init__(self):
        self.layers: Dict[str, UXOLayer] = {}
        self.layer_groups: Dict[str, List[str]] = {
            "Survey Data": [],
            "Magnetic Processing": [],
            "GPR Processing": [],
            "Gamma Processing": [],
            "Multispectral Processing": [],
            "Annotations": []
        }
```

#### 3. **AdvancedMapWidget** - pyqtlet2 Integration
```python
class AdvancedMapWidget(QWidget):
    """Advanced map widget with real-time layer management"""
    
    def __init__(self):
        self.map_widget = MapWidget()  # pyqtlet2
        self.map = L.map(self.map_widget)
        self.layer_manager = LayerManager()
        self.leaflet_layers: Dict[str, Any] = {}  # Map layer names to leaflet objects
        
    def add_layer_realtime(self, uxo_layer: UXOLayer):
        """Add layer without HTML reload"""
        # Convert UXOLayer to appropriate Leaflet layer
        # Update layer control panel
        # Emit signals for UI updates
```

#### 4. **LayerControlPanel** - QGIS-style UI
```python
class LayerControlPanel(QWidget):
    """QGIS-style layer management panel"""
    
    def __init__(self, layer_manager: LayerManager):
        self.layer_manager = layer_manager
        self.tree_widget = QTreeWidget()
        self.setup_context_menus()
        self.setup_drag_drop()
```

## Implementation Phases

### Phase 1: Foundation (Week 1-2)
**Objective**: Replace Folium with pyqtlet2 baseline

#### Tasks:
1. **Install and configure pyqtlet2**
   ```bash
   pip install "pyqtlet2[PySide6]"
   ```

2. **Create new AdvancedMapWidget**
   - Replace current MapWidget with pyqtlet2 implementation
   - Maintain Norwegian Kartverket tile support via WMS
   - Preserve existing toolbar functionality

3. **Define data structures**
   - Implement `UXOLayer` dataclass
   - Create `LayerType` and `GeometryType` enums
   - Design `LayerStyle` configuration system

4. **Basic integration**
   - Update MainWindow to use new map widget
   - Ensure existing Norwegian tiles load correctly
   - Maintain current zoom/pan functionality

**Deliverables**:
- ✅ Working pyqtlet2 map with Norwegian tiles
- ✅ UXOLayer data structure defined
- ✅ Basic map integration in main window

### Phase 2: Data Viewer Integration (Week 2-3)
**Objective**: Enable data viewer to create map layers

#### Tasks:
1. **Enhance DataViewer layer creation**
   ```python
   def create_map_layer(self) -> Optional[UXOLayer]:
       """Create UXOLayer from current DataFrame"""
       df = self.model.get_dataframe()
       
       # Auto-detect geometry columns
       lat_col, lon_col = self.detect_coordinate_columns(df)
       
       if lat_col and lon_col:
           return UXOLayer(
               name=f"Survey Data - {self.current_file}",
               layer_type=LayerType.POINTS,
               data=df,
               geometry_type=GeometryType.POINT,
               style=self.create_default_style(),
               metadata=self.extract_metadata(),
               source=LayerSource.DATA_VIEWER
           )
   ```

2. **Coordinate system detection**
   - Auto-detect UTM zones for Norwegian data
   - Support common coordinate systems (WGS84, UTM 32N, UTM 33N)
   - Implement coordinate transformation pipeline

3. **Point visualization**
   - Create efficient point rendering for large datasets
   - Implement clustering for dense point clouds
   - Add popup information with data attributes

4. **Layer styling system**
   - Default styles for different data types
   - Color coding based on data attributes
   - Symbol size scaling options

**Deliverables**:
- ✅ DataViewer creates UXOLayer objects
- ✅ Automatic coordinate detection and transformation
- ✅ Point visualization with clustering
- ✅ Basic styling system

### Phase 3: Processing Integration (Week 3-4)
**Objective**: Process results become map layers automatically

#### Tasks:
1. **Extend ProcessingResult to create layers**
   ```python
   class ProcessingResult:
       # ... existing fields ...
       
       def to_uxo_layer(self, layer_name: str = None) -> UXOLayer:
           """Convert processing result to map layer"""
           # Determine layer type based on processing output
           # Handle both point and raster data
           # Preserve processing lineage
   ```

2. **Processing-specific layer types**
   - **Magnetic Processing**: Anomaly points, interpolated grids
   - **GPR Processing**: Profile lines, target locations
   - **Gamma Processing**: Radiation contours, hot spots
   - **Multispectral**: Classified rasters, vegetation indices

3. **Multi-format support**
   - Vector data (points, lines, polygons) as GeoJSON
   - Raster data as tile layers or ImageOverlay
   - Hybrid visualizations (contours + points)

4. **Processing lineage tracking**
   - Link processed layers to source data
   - Track parameter history
   - Enable layer dependency management

**Deliverables**:
- ✅ ProcessingResult creates appropriate UXOLayer
- ✅ Support for both vector and raster outputs
- ✅ Processing history preserved in layers
- ✅ Automatic layer generation from processing pipeline

### Phase 4: Advanced Layer Management (Week 4-5)
**Objective**: QGIS-style layer controls and interaction

#### Tasks:
1. **LayerControlPanel implementation**
   ```python
   class LayerControlPanel(QWidget):
       def __init__(self):
           self.tree_widget = QTreeWidget()
           self.setup_layer_tree()
           self.setup_context_menus()
           self.setup_drag_drop_reordering()
   ```

2. **Layer operations**
   - Toggle visibility with checkboxes
   - Opacity sliders for each layer
   - Z-order management via drag-and-drop
   - Layer grouping and organization

3. **Styling controls**
   - Color pickers for point/line styles
   - Size scaling controls
   - Classification and symbology options
   - Style presets for common UXO data types

4. **Layer interactions**
   - Click to identify features
   - Select and highlight data points
   - Cross-filtering between map and data viewer
   - Export selected features

**Deliverables**:
- ✅ Complete layer control panel
- ✅ Real-time visibility and styling controls
- ✅ Layer reordering and grouping
- ✅ Interactive feature selection

### Phase 5: Advanced Features (Week 5-6)
**Objective**: Performance optimization and specialized tools

#### Tasks:
1. **Performance optimization**
   - Implement level-of-detail for large datasets
   - Use WebGL rendering for massive point clouds
   - Efficient raster tile caching
   - Progressive loading for large files

2. **Norwegian mapping enhancements**
   - Multiple Kartverket layer options
   - Custom coordinate system support
   - Offline tile caching for field work
   - Norwegian administrative boundaries

3. **UXO-specific tools**
   - Anomaly detection visualization
   - Survey coverage analysis
   - Target probability overlays
   - Export to professional GIS formats

4. **Integration polish**
   - Smooth animations for layer transitions
   - Professional print layout options
   - Batch processing visualization
   - Project save/load with layer states

**Deliverables**:
- ✅ Optimized performance for large datasets
- ✅ Enhanced Norwegian mapping features
- ✅ UXO-specific analysis tools
- ✅ Professional-grade output options

## Technical Specifications

### Layer Type Definitions
```python
from enum import Enum

class LayerType(Enum):
    POINTS = "points"
    RASTER = "raster"
    VECTOR = "vector"
    PROCESSED = "processed"
    ANNOTATION = "annotation"

class GeometryType(Enum):
    POINT = "point"
    LINE = "line"
    POLYGON = "polygon"
    RASTER = "raster"
    MULTIPOINT = "multipoint"

class LayerSource(Enum):
    DATA_VIEWER = "data_viewer"
    PROCESSING = "processing"
    IMPORT = "import"
    ANNOTATION = "annotation"
```

### Coordinate System Support
```python
SUPPORTED_CRS = {
    "EPSG:4326": "WGS84 Geographic",
    "EPSG:25832": "ETRS89 / UTM zone 32N",
    "EPSG:25833": "ETRS89 / UTM zone 33N",
    "EPSG:25834": "ETRS89 / UTM zone 34N",
    "EPSG:25835": "ETRS89 / UTM zone 35N"
}
```

### Performance Targets
- **Layer Addition**: < 100ms for point layers up to 10k points
- **Visibility Toggle**: < 50ms response time
- **Style Changes**: < 200ms for complete re-rendering
- **Large Datasets**: Support up to 100k points with clustering
- **Memory Usage**: < 500MB additional for layer management

## Data Flow Architecture

```
DataViewer → UXOLayer → LayerManager → AdvancedMapWidget → Leaflet
     ↓                      ↓                ↓
ProcessingPipeline → ProcessingResult → UXOLayer → Real-time Display
     ↓                      ↓                ↓
LayerControlPanel ← LayerManager ← User Interactions
```

## Integration Points

### 1. DataViewer → Map Integration
```python
# In DataViewer
def plot_on_map(self):
    """Send current data to map as a layer"""
    layer = self.create_map_layer()
    if layer:
        self.layer_created.emit(layer)
        
# In MainWindow
def connect_data_to_map(self):
    self.data_viewer.layer_created.connect(
        self.map_widget.add_layer_realtime
    )
```

### 2. Processing → Map Integration
```python
# In ProcessingPipeline
def _on_processing_finished(self, result: ProcessingResult):
    # Generate output file (existing)
    output_path = self._generate_output_file(result)
    
    # NEW: Create map layer
    layer = result.to_uxo_layer(
        layer_name=f"{result.processing_script} Result"
    )
    self.layer_created.emit(layer)
```

### 3. Map → DataViewer Integration
```python
# Bidirectional communication
class AdvancedMapWidget:
    def on_feature_selected(self, layer_name: str, feature_ids: List[int]):
        """Handle map feature selection"""
        self.feature_selected.emit(layer_name, feature_ids)
        
# In MainWindow
def connect_map_to_data(self):
    self.map_widget.feature_selected.connect(
        self.data_viewer.highlight_rows
    )
```

## Risk Assessment & Mitigation

### Technical Risks

1. **pyqtlet2 Maturity**
   - *Risk*: Limited community support compared to Folium
   - *Mitigation*: Maintain fallback to QWebEngineView + custom Leaflet if needed
   - *Timeline Impact*: +1 week if major issues found

2. **Large Dataset Performance**
   - *Risk*: Point cloud rendering may be slow for >50k points
   - *Mitigation*: Implement clustering and level-of-detail early
   - *Testing*: Benchmark with realistic UXO survey datasets

3. **Norwegian Tile Integration**
   - *Risk*: Kartverket WMS compatibility with pyqtlet2
   - *Mitigation*: Test WMS integration in Phase 1, have backup tile sources
   - *Fallback*: Use standard OSM/satellite tiles if needed

### Project Risks

1. **Scope Creep**
   - *Risk*: Adding GIS features beyond core requirements
   - *Mitigation*: Stick to defined phases, defer advanced features to future versions
   - *Decision Gate*: Review after Phase 3

2. **Integration Complexity**
   - *Risk*: Breaking existing DataViewer/Processing functionality
   - *Mitigation*: Implement alongside existing system, gradual migration
   - *Testing*: Comprehensive regression testing of existing features

## Success Metrics

### Technical KPIs
- **Layer Addition Time**: < 100ms for typical datasets
- **Memory Efficiency**: < 50MB additional per active layer
- **UI Responsiveness**: < 50ms for layer visibility toggles
- **Data Integrity**: 100% preservation of coordinate accuracy

### User Experience KPIs
- **Workflow Improvement**: 80% reduction in map update time
- **Feature Adoption**: 90% of users utilize layer management features
- **Error Reduction**: 50% fewer map-related user errors
- **Task Completion**: 30% faster survey analysis workflows

### Compatibility KPIs
- **Norwegian Mapping**: 100% compatibility with Kartverket tiles
- **Data Format Support**: All existing CSV/Excel formats work seamlessly
- **Processing Integration**: All processor types create valid map layers
- **Export Quality**: Professional-grade map output maintained

## Future Enhancements (Beyond Initial Implementation)

### Advanced Visualization
- **3D Terrain Visualization**: Integrate elevation models
- **Time-series Animation**: Animated survey data over time
- **Advanced Symbology**: Graduated symbols, chart overlays
- **Custom Projections**: Support for local grid systems

### Collaborative Features
- **Shared Map Sessions**: Real-time collaboration on survey analysis
- **Annotation Sharing**: Collaborative markup and notes
- **Version Control**: Track layer changes over time
- **Cloud Synchronization**: Sync layer states across devices

### Analysis Tools
- **Spatial Analysis**: Buffer zones, overlay analysis
- **Statistical Overlays**: Density maps, hotspot analysis
- **Predictive Modeling**: ML-based anomaly prediction visualization
- **Report Generation**: Automated map-based reports

## Conclusion

This implementation plan provides a clear path from the current Folium-based system to a professional-grade mapping solution using pyqtlet2. The phased approach minimizes risk while delivering incremental value, and the modular architecture supports future enhancements.

The new system will eliminate the HTML reload bottleneck while providing QGIS-style layer management capabilities specifically tailored for UXO detection workflows. With proper implementation, this will significantly improve the user experience and enable more sophisticated analysis workflows.

---

**Next Steps**:
1. Review and approve this plan
2. Set up development environment with pyqtlet2
3. Begin Phase 1 implementation
4. Regular progress reviews at end of each phase

**Document History**:
- v1.0: Initial plan creation (January 2025) 