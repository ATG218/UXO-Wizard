# Processor Layer Output to UXOLayer System Adaptation Guide

## Executive Summary

This document explains how to adapt the current processor pipeline layer outputs to work seamlessly with the UXOLayer system for creating toggleable map layers. The system already has most components in place but needs enhanced integration between processing scripts and the map visualization system.

## Current System Architecture

### Processing Pipeline Architecture
- **Base Framework**: `src/processing/base.py` - Defines `ScriptInterface`, `BaseProcessor`, `ProcessingResult`, and `LayerOutput`
- **Layer Output Structure**: `LayerOutput` class with layer_type, data (Any), style_info, and metadata
- **Supported Layer Types**: `flight_lines`, `grid_visualization`, `points`, `processed`, `annotation`
- **Data Types**: Supports DataFrames, numpy arrays, coordinate lists, and arbitrary data structures

### UXOLayer System Architecture
- **Core Layer Class**: `src/ui/map/layer_types.py` - `UXOLayer` dataclass supporting Union[pd.DataFrame, np.ndarray, dict]
- **Layer Types**: POINTS, RASTER, VECTOR, PROCESSED, ANNOTATION with full GeometryType support
- **Styling System**: Comprehensive `LayerStyle` with point, line, fill, clustering, and graduated colors
- **Layer Management**: `src/ui/map/layer_manager.py` - Hierarchical organization with processing lineage
- **UI Controls**: `src/ui/map/layer_control_panel.py` - Professional QGIS-style interface
- **Map Integration**: `src/ui/map/advanced_map_widget.py` - pyqtlet2-based with Norwegian CRS support

### Current Implementation Status
- **✅ Point Layers**: Fully implemented with coordinate auto-detection, clustering, styling
- **⚠️ Raster Layers**: Architecture exists but `_create_raster_layer()` is placeholder
- **⚠️ Vector Layers**: Architecture exists but `_create_vector_layer()` is placeholder
- **Basic Example**: `src/ui/data_viewer.py:795-869` - Only handles simple point data from DataFrames

## Key Adaptation Requirements

### 1. Complete Raster and Vector Layer Implementation

**Current Gap**: UXOLayer architecture supports raster and vector data, but `AdvancedMapWidget` has placeholder implementations.

**Critical Missing Features**:
- **Raster Layers**: No GeoTIFF loading, numpy array to image overlay, grid interpolation
- **Vector Layers**: No line/polygon geometry from DataFrames, flight path visualization
- **Advanced Styling**: Limited data-driven styling, color ramps for continuous data

**Solution**: Implement the missing layer rendering methods:

```python
# In advanced_map_widget.py - Complete raster implementation
def _create_raster_layer(self, layer: UXOLayer):
    """Create raster layer from various data sources"""
    if isinstance(layer.data, np.ndarray):
        # Grid data - convert to image overlay
        bounds = layer.bounds  # [min_lon, min_lat, max_lon, max_lat]
        image_overlay = L.imageOverlay(
            self._array_to_data_url(layer.data, layer.style),
            [[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
            {'opacity': layer.opacity}
        )
        return image_overlay
    elif isinstance(layer.data, str) and layer.data.endswith(('.tif', '.tiff')):
        # GeoTIFF file - requires rasterio integration
        return self._load_geotiff_overlay(layer.data, layer.style)
    elif isinstance(layer.data, pd.DataFrame):
        # Point data for interpolation to grid
        return self._create_interpolated_raster(layer.data, layer.style, layer.bounds)

def _create_vector_layer(self, layer: UXOLayer):
    """Create vector layer for lines, polygons, and paths"""
    if layer.geometry_type == GeometryType.LINE:
        coords = self._extract_line_coordinates(layer.data)
        return L.polyline(coords, {
            'color': layer.style.line_color,
            'weight': layer.style.line_width,
            'opacity': layer.style.line_opacity,
            'dashArray': self._get_dash_pattern(layer.style.line_style)
        })
    elif layer.geometry_type == GeometryType.POLYGON:
        coords = self._extract_polygon_coordinates(layer.data)
        return L.polygon(coords, {
            'color': layer.style.line_color,
            'fillColor': layer.style.fill_color,
            'fillOpacity': layer.style.fill_opacity
        })
```

### 2. Enhanced LayerOutput to UXOLayer Conversion

**Current Gap**: Processing scripts generate `LayerOutput` objects, but no automatic conversion handles the full range of data types.

**Solution**: Create intelligent conversion function in `src/processing/base.py`:

```python
def layer_output_to_uxo_layer(layer_output: LayerOutput, 
                              processing_name: str,
                              input_file: str = None) -> UXOLayer:
    """Convert LayerOutput to UXOLayer with intelligent type detection"""
    
    # Map layer types to UXOLayer types
    layer_type_mapping = {
        'points': LayerType.POINTS,
        'flight_lines': LayerType.VECTOR,
        'grid_visualization': LayerType.RASTER,
        'processed': LayerType.PROCESSED,
        'annotation': LayerType.ANNOTATION
    }
    
    # Intelligent geometry type detection based on data
    geometry_type = _detect_geometry_type(layer_output.data, layer_output.layer_type)
    
    # Enhanced styling conversion
    style = _convert_layer_style(layer_output.style_info, layer_output.data, processing_name)
    
    # Calculate bounds based on data type
    bounds = _calculate_bounds(layer_output.data, geometry_type)
    
    # Create comprehensive metadata
    metadata = {
        'processing_script': processing_name,
        'source_file': input_file,
        'data_type': type(layer_output.data).__name__,
        'original_metadata': layer_output.metadata,
        'row_count': _get_data_size(layer_output.data),
        'coordinate_system': _detect_coordinate_system(layer_output.data)
    }
    
    return UXOLayer(
        name=f"{processing_name} - {layer_output.layer_type.replace('_', ' ').title()}",
        layer_type=layer_type_mapping.get(layer_output.layer_type, LayerType.PROCESSED),
        data=layer_output.data,
        geometry_type=geometry_type,
        style=style,
        metadata=metadata,
        source=LayerSource.PROCESSING,
        bounds=bounds,
        processing_history=[processing_name]
    )

def _detect_geometry_type(data: Any, layer_type: str) -> GeometryType:
    """Intelligently detect geometry type from data"""
    if isinstance(data, np.ndarray) and data.ndim == 2:
        return GeometryType.RASTER
    elif isinstance(data, pd.DataFrame):
        if layer_type == 'flight_lines' or 'track' in str(data.columns).lower():
            return GeometryType.LINE
        elif any(col in str(data.columns).lower() for col in ['lat', 'lon', 'x', 'y', 'northing', 'easting']):
            return GeometryType.POINT
    elif isinstance(data, list) and len(data) > 0:
        if isinstance(data[0], (list, tuple)) and len(data[0]) == 2:
            return GeometryType.LINE  # Coordinate pairs for line
    
    return GeometryType.POINT  # Default fallback

def _convert_layer_style(style_info: Dict, data: Any, processing_name: str) -> LayerStyle:
    """Convert basic style_info to comprehensive LayerStyle"""
    style = LayerStyle()
    
    # Processing-specific styling
    if 'magnetic' in processing_name.lower():
        style = _create_magnetic_style(data, style_info)
    elif 'gpr' in processing_name.lower():
        style = _create_gpr_style(data, style_info)
    elif 'gamma' in processing_name.lower():
        style = _create_gamma_style(data, style_info)
    
    # Override with provided style_info
    if 'color' in style_info:
        style.point_color = style_info['color']
    if 'size' in style_info:
        style.point_size = style_info['size']
    if 'opacity' in style_info:
        style.point_opacity = style_info['opacity']
    
    return style
```

### 3. Automatic Layer Registration

**Current Gap**: Processed layers aren't automatically added to the LayerManager.

**Solution**: Extend `ProcessingWorker` in `src/processing/base.py` to emit layer signals:

```python
class ProcessingWorker(QObject):
    # Existing signals...
    layer_created = pyqtSignal(object)  # Add this signal
    
    def run(self):
        # Existing processing...
        if result.success:
            # Convert and emit layers
            for layer_output in result.layer_data:
                uxo_layer = layer_output_to_uxo_layer(
                    layer_output, 
                    self.script.name,
                    self.input_file_path
                )
                self.layer_created.emit(uxo_layer)
```

### 4. Processing-Specific Layer Grouping

**Current Gap**: Processed layers need organized grouping in the layer panel.

**Solution**: Enhance `LayerManager` grouping logic:

```python
# In layer_manager.py - enhance _assign_to_group method
def _assign_to_group(self, layer: UXOLayer) -> str:
    if layer.source == LayerSource.PROCESSING:
        # Group by processing type
        processing_type = layer.processing_history[0] if layer.processing_history else "Unknown"
        if 'magnetic' in processing_type.lower():
            return "Magnetic Processing"
        elif 'gpr' in processing_type.lower():
            return "GPR Processing"
        elif 'gamma' in processing_type.lower():
            return "Gamma Processing"
        else:
            return "Data Processing"
    # ... existing logic
```

### 5. Coordinate System Integration

**Current Gap**: Processing scripts output various coordinate formats, but UXOLayer system expects consistent format.

**Solution**: Standardize coordinate handling in processing scripts:

```python
# In magbase_processing.py - enhance coordinate output
def execute(self, data: pd.DataFrame, params: Dict, progress_callback=None, input_file_path=None):
    # ... existing processing ...
    
    # Ensure coordinate columns are standardized for map integration
    if 'UTM_Easting' in processed_df.columns and 'UTM_Northing' in processed_df.columns:
        # Convert UTM to WGS84 for map display
        processed_df = self.add_wgs84_coordinates(processed_df)
    
    # Create layer outputs with proper coordinate metadata
    layer_output = LayerOutput(
        layer_type='points',
        data=processed_df,
        style={'color': 'red', 'size': 3},
        metadata={
            'coordinate_system': 'UTM',
            'utm_zone': params.get('utm_zone', 33),
            'lat_column': 'Latitude',
            'lon_column': 'Longitude'
        }
    )
```

### 6. Dynamic Styling Based on Processing Results

**Current Gap**: Processing scripts use basic styling, but UXOLayer system supports advanced styling.

**Solution**: Create processing-specific styling functions:

```python
def create_magnetic_layer_style(data: pd.DataFrame, processing_type: str) -> LayerStyle:
    """Create optimized styling for magnetic data layers"""
    
    if 'R1 [nT]' in data.columns:  # Residual magnetic field
        # Use graduated color scheme based on residual values
        return LayerStyle(
            color_column='R1 [nT]',
            color_scheme='RdBu_r',  # Red-Blue diverging
            size=4,
            opacity=0.8,
            clustering_enabled=True,
            clustering_radius=50
        )
    else:
        # Default point styling
        return LayerStyle(color='blue', size=3, opacity=0.7)
```

## Implementation Steps

### Phase 1: Complete Core Layer Support (Critical Priority)
1. **Implement raster layer rendering** in `AdvancedMapWidget._create_raster_layer()`
   - Add rasterio integration for GeoTIFF loading
   - Implement numpy array to image overlay conversion
   - Add grid interpolation from point data
   - Create color ramp styling for continuous data

2. **Implement vector layer rendering** in `AdvancedMapWidget._create_vector_layer()`
   - Add DataFrame to LineString/Polygon conversion
   - Support flight path visualization from GPS tracks
   - Implement WKT/WKB geometry column parsing

3. **Add required dependencies** to project requirements
   - `rasterio` for GeoTIFF handling
   - `scipy` for grid interpolation
   - `shapely` for geometry operations (if not already included)

### Phase 2: Enhanced Processing Integration (High Priority)
1. **Add intelligent layer conversion** function to `src/processing/base.py`
2. **Enhance ProcessingWorker** to emit layer signals with full data type support
3. **Connect processing pipeline** to LayerManager in main application
4. **Implement processing-specific styling** functions

### Phase 3: Advanced Features (Medium Priority)
1. **Enhance coordinate system handling** in processing scripts
2. **Add advanced styling options** (graduated colors, data-driven styling)
3. **Implement layer preview** in processing parameter dialogs
4. **Add layer metadata export** for processed results

### Phase 4: Optimization (Low Priority)
1. **Add caching for large raster layers**
2. **Implement progressive loading** for large datasets
3. **Add 3D visualization support** for elevation data

## Example Usage Patterns

### Magnetic Processing with Multiple Layer Types

```python
# In magbase_processing.py - Enhanced layer output
def execute(self, data: pd.DataFrame, params: Dict, progress_callback=None, input_file_path=None):
    # ... existing processing ...
    
    # 1. Point data layer - processed magnetic readings
    result.add_layer_output(
        layer_type='points',
        data=processed_df,  # DataFrame with UTM coordinates and residual fields
        style_info={'color_field': 'R1 [nT]', 'color_scheme': 'RdBu_r', 'size': 4},
        metadata={'description': 'Processed magnetic readings with residual anomalies'}
    )
    
    # 2. Vector layer - flight lines
    if 'flight_track' in processed_df.columns:
        result.add_layer_output(
            layer_type='flight_lines',
            data=flight_path_df,  # DataFrame with sequential coordinates
            style_info={'line_color': 'blue', 'line_width': 2},
            metadata={'description': 'Survey flight paths'}
        )
    
    # 3. Raster layer - interpolated magnetic field (if requested)
    if params.get('create_grid', False):
        grid_array = interpolate_magnetic_field(processed_df)
        result.add_layer_output(
            layer_type='grid_visualization',
            data=grid_array,  # numpy array
            style_info={'color_ramp': 'magnetic_field', 'transparency': 0.7},
            metadata={'description': 'Interpolated magnetic field grid', 'grid_resolution': params['grid_size']}
        )
```

### GPR Processing with Raster Output

```python
# In gpr_processing.py - Time slice visualization
def execute(self, data: pd.DataFrame, params: Dict, progress_callback=None, input_file_path=None):
    # ... process GPR data ...
    
    # Create time slice raster
    time_slice_array = create_time_slice(radargram_data, depth=params['depth'])
    
    result.add_layer_output(
        layer_type='grid_visualization',
        data=time_slice_array,
        style_info={
            'color_ramp': 'amplitude',
            'min_value': time_slice_array.min(),
            'max_value': time_slice_array.max()
        },
        metadata={'description': f'GPR time slice at {params["depth"]}m depth'}
    )
```

### Automatic Workflow After Implementation

1. **User runs processing script** (e.g., magbase_processing.py)
2. **Script generates multiple LayerOutput** objects (points, vectors, rasters)
3. **ProcessingWorker automatically converts** each to UXOLayer with intelligent type detection
4. **Layers are automatically added** to LayerManager with processing-specific grouping
5. **User sees professional layer panel** with toggleable layers organized by type
6. **All layer types render correctly** on map with optimized styling
7. **Real-time interaction** - no HTML reloads, immediate visibility changes

## Benefits of This Enhanced Adaptation

- **Complete Data Type Support**: Points, rasters, vectors, and complex geometries all render correctly
- **Professional Visualization**: GeoTIFF loading, grid interpolation, flight path rendering
- **Intelligent Processing**: Automatic data type detection and appropriate styling
- **Real-time Interaction**: pyqtlet2-based rendering without HTML reloads
- **Norwegian Standards**: Built-in UTM zone handling and coordinate system support
- **Processing Lineage**: Full metadata tracking from raw data through final visualization
- **Scalable Architecture**: Extensible to new processing types and data formats

## Critical Files Requiring Modification

### Core Map Rendering (Critical)
1. **`src/ui/map/advanced_map_widget.py`** - Implement `_create_raster_layer()` and `_create_vector_layer()`
2. **`src/ui/map/layer_types.py`** - Enhance LayerStyle for raster color ramps and vector styling
3. **Project dependencies** - Add rasterio, scipy for geospatial operations

### Processing Integration (High Priority)  
4. **`src/processing/base.py`** - Add intelligent layer conversion functions and signal emission
5. **`src/ui/map/layer_manager.py`** - Enhance processing layer grouping logic
6. **Main application file** - Connect ProcessingWorker signals to LayerManager

### Processing Scripts (Medium Priority)
7. **`src/processing/scripts/magnetic/magbase_processing.py`** - Add multiple layer outputs
8. **Other processing scripts** - Update to use enhanced layer output patterns
9. **Processing templates** - Create examples for different data types

## Major Technical Gaps to Address

1. **Raster Rendering**: Currently completely missing - requires rasterio integration
2. **Vector Rendering**: Currently completely missing - requires coordinate sequence handling  
3. **Color Ramps**: No support for continuous data visualization
4. **Grid Interpolation**: No ability to create raster from point data
5. **Advanced Styling**: Limited data-driven styling capabilities
6. **Large Dataset Handling**: No optimization for large raster/vector datasets

This adaptation will create a seamless workflow from data processing to interactive map visualization with minimal changes to existing code structure.