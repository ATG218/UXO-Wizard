# UXO Wizard User Scripts

Welcome to the UXO Wizard user scripts directory! This is where you can add your own custom processing scripts to extend the functionality of UXO Wizard without modifying the core application.

## 🚀 Quick Start

1. **Choose your processor type**: Navigate to the appropriate subfolder based on your data type:
   - `magnetic/` - Magnetic anomaly detection and processing
   - `gamma/` - Gamma radiation analysis 
   - `gpr/` - Ground Penetrating Radar processing
   - `multispectral/` - Multispectral data analysis

2. **Copy the template**: Start with `examples/template_script.py` as your base

3. **Customize**: Rename and modify the script for your specific needs

4. **Override built-ins**: Create a script with the same filename as a built-in script to replace it with your custom version

5. **Test**: Load your data in UXO Wizard and test your script - it will appear with a 🟢 green icon

## 🏗️ System Architecture

UXO Wizard uses a modular architecture that separates processors from scripts:

- **Processors** (e.g., `MagneticProcessor`) manage data types and discover scripts
- **Scripts** (your custom code) perform the actual data processing algorithms
- **Pipeline** coordinates execution, file management, and UI integration

Your external scripts integrate seamlessly with this system by implementing the `ScriptInterface`.

## 📦 Available Packages

Your scripts can use any of the following packages (already included in UXO Wizard after packaging):

### Core Data Processing & Analysis
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **scipy** - Scientific computing (statistics, interpolation, optimization)
- **scikit-learn** - Machine learning algorithms
- **scikit-image** - Image processing algorithms
- **joblib** - Parallel computing utilities
- **networkx** - Network analysis and graph algorithms
- **statsmodels** - Statistical modeling and econometrics
- **numba** - JIT compilation for numerical functions

### Visualization & Interactive Tools
- **matplotlib** - Static plots and figures
- **plotly** - Interactive visualizations and dashboards
- **seaborn** - Statistical data visualization
- **folium** - Interactive maps
- **branca** - HTML/JS generation for folium

### Geospatial & Mapping
- **geopandas** - Geospatial data processing
- **rasterio** - Geospatial raster data I/O
- **fiona** - Vector data I/O
- **shapely** - Geometric objects and operations
- **pyproj** - Cartographic projections and transformations
- **proj** - PROJ coordinate transformation library
- **gdal** - Geospatial Data Abstraction Library
- **pyogrio** - Fast vector data I/O
- **contextily** - Add basemaps to matplotlib plots
- **geopy** - Geocoding and distance calculations
- **geographiclib** - Geodesic calculations
- **xyzservices** - Source of XYZ tile services
- **mercantile** - Web mercator tile utilities
- **utm** - UTM coordinate conversion

### File I/O & Data Formats
- **openpyxl** - Excel file support (.xlsx)
- **xlrd** - Reading Excel files (.xls)
- **lxml** - XML/HTML processing
- **requests** - HTTP requests and web APIs
- **h5py** - HDF5 file format
- **netcdf4** - NetCDF file format
- **pyarrow** - Apache Arrow columnar data
- **fastparquet** - Fast parquet file format
- **fsspec** - Filesystem specification
- **cramjam** - Compression/decompression utilities
- **tifffile** - TIFF image file support

### Image Processing
- **PIL/Pillow** - Image processing library
- **cv2** - OpenCV computer vision
- **imageio** - Image I/O operations
- **imagecodecs** - Image compression codecs

### Scientific Computing
- **obspy** - Seismology and geophysics
- **pywavelets** - Wavelet transforms
- **threadpoolctl** - Thread pool control

### R Integration
- **rpy2** - Python-R interface

### System & Development Utilities
- **loguru** - Advanced logging
- **psutil** - System and process utilities
- **click** - Command-line interface creation
- **colorama** - Cross-platform colored terminal text
- **tqdm** - Progress bars
- **debugpy** - Python debugger
- **pexpect** - Spawn and control interactive programs

### Compression & Encoding
- **zstandard** - Zstandard compression
- **brotli** - Brotli compression
- **lz4** - LZ4 compression
- **snappy** - Snappy compression

### Text Processing
- **jinja2** - Template engine
- **markupsafe** - Safe string handling
- **pygments** - Syntax highlighting

### Configuration & Serialization
- **pyyaml** - YAML parser and emitter
- **toml** - TOML parser
- **tomli** - TOML parser (Python < 3.11)
- **configparser** - Configuration file parser

### Date/Time Handling
- **python-dateutil** - Extensions to datetime
- **pytz** - World timezone definitions
- **tzdata** - Timezone database
- **tzlocal** - Local timezone detection

### UXO Wizard Internal
- **src** - UXO Wizard internal modules (for importing ScriptInterface, etc.)

## 🆕 Need a New Package?

If you need a package that's not listed above:

1. **Open an issue** at: [https://github.com/ATG218/UXO-Wizard/issues](https://github.com/ATG218/UXO-Wizard/issues)
2. **Use the "Package Request" template**
3. **Provide details**:
   - Package name and version
   - Your use case and why it's needed
   - Example code showing how you'd use it
   - Any alternative packages you considered

We review all package requests and add commonly requested packages in future releases.

## 📝 Script Structure

Every external script must implement the `ScriptInterface` and follow this structure:

```python
from src.processing.base import ScriptInterface, ProcessingResult, ScriptMetadata
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Callable

class YourCustomScript(ScriptInterface):
    
    @property
    def name(self) -> str:
        return "Your Custom Script Name"
    
    @property  
    def description(self) -> str:
        return "Brief description of what your script does"
    
    def get_metadata(self) -> ScriptMetadata:
        return ScriptMetadata(
            description=self.description,
            flags=["your", "tags", "here"],
            typical_use_case="When and why to use this script",
            field_compatible=True,  # Can be run in field conditions
            estimated_runtime="< 2 minutes"
        )
    
    def get_parameters(self) -> Dict[str, Any]:
        """Define UI parameters for your script"""
        return {
            'processing_options': {
                'threshold': {
                    'value': 2.0,
                    'type': 'float',
                    'min': 0.1,
                    'max': 10.0,
                    'description': 'Processing threshold'
                },
                'method': {
                    'value': 'default',
                    'type': 'choice',
                    'choices': ['default', 'advanced', 'fast'],
                    'description': 'Processing method'
                }
            }
        }
    
    def execute(self, data: pd.DataFrame, params: Dict[str, Any], 
                progress_callback: Optional[Callable] = None, 
                input_file_path: Optional[str] = None) -> ProcessingResult:
        """Main processing logic"""
        
        if progress_callback:
            progress_callback(0, "Starting processing...")
        
        try:
            # Extract parameters safely
            threshold = params.get('processing_options', {}).get('threshold', {}).get('value', 2.0)
            
            # Process your data
            processed_data = data.copy()
            # ... your processing logic here ...
            
            if progress_callback:
                progress_callback(50, "Halfway complete...")
            
            # Create result
            result = ProcessingResult(
                success=True,
                data=processed_data,
                processor_type='your_processor_type',  # magnetic, gamma, gpr, etc.
                script_id='your_script_id'
            )
            
            # Add map layers
            result.add_layer_output(
                layer_type='points',
                data=processed_data,
                style_info={'color': '#FF6600', 'size': 5},
                metadata={'layer_name': 'Your Layer Name'}
            )
            
            if progress_callback:
                progress_callback(100, "Processing complete!")
            
            return result
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                error_message=f"Processing failed: {str(e)}"
            )

# Required: Export your script class
SCRIPT_CLASS = YourCustomScript
```

### Required Components

1. **Class Definition**: Inherit from `ScriptInterface`
2. **Properties**: `name` and `description` as `@property` methods
3. **Metadata**: Implement `get_metadata()` with script information
4. **Parameters**: Define `get_parameters()` for UI controls
5. **Execution**: Implement `execute()` with your processing logic
6. **Export**: Include `SCRIPT_CLASS = YourClassName` at the end

## 🎛️ Parameter Types

Create various UI controls for your script parameters:

```python
def get_parameters(self) -> Dict[str, Any]:
    return {
        'section_name': {
            'float_param': {
                'value': 2.5,
                'type': 'float',
                'min': 0.1,
                'max': 10.0,
                'description': 'A floating point parameter'
            },
            'choice_param': {
                'value': 'option1',
                'type': 'choice',
                'choices': ['option1', 'option2', 'option3'],
                'description': 'Choose from dropdown'
            },
            'bool_param': {
                'value': True,
                'type': 'bool',
                'description': 'A checkbox parameter'
            },
            'int_param': {
                'value': 10,
                'type': 'int',
                'min': 1,
                'max': 100,
                'description': 'An integer parameter'
            },
            'file_param': {
                'value': '',
                'type': 'file',
                'file_types': ['.csv', '.txt'],
                'description': 'Select a file'
            }
        }
    }
```

## 📊 Script Metadata System

The metadata system enhances user experience with rich script information:

```python
def get_metadata(self) -> ScriptMetadata:
    return ScriptMetadata(
        description="Brief but clear description of what this script does",
        flags=["processing", "field-use", "visualization"],  # See available flags below
        typical_use_case="Explain when and why someone would use this script",
        field_compatible=True,  # True if fast/simple enough for field use
        estimated_runtime="< 30 seconds"  # Realistic time estimate
    )
```

### Available Metadata Flags

- **`preprocessing`** - Data cleaning, corrections, initial processing
- **`analysis`** - Core analytical processing and calculations  
- **`visualization`** - Produces plots, maps, or visual outputs
- **`field-use`** - Optimized for field conditions (fast, simple interface)
- **`export`** - Data export, formatting, or conversion
- **`qc`** - Quality control, validation, or diagnostic tools
- **`advanced`** - Requires expertise or complex configuration
- **`batch`** - Suitable for processing multiple files

## 🗺️ Creating Map Layers

Your script can create layers that appear automatically on the UXO Wizard map:

```python
# Point data layer
result.add_layer_output(
    layer_type='points',
    data=your_dataframe,  # Must have latitude/longitude or UTM columns
    style_info={
        'color_field': 'value_column',  # Column to color by
        'use_graduated_colors': True,
        'color_scheme': 'magnetic',  # or 'gamma', 'gpr'
        'size': 5,
        'opacity': 0.8
    },
    metadata={
        'layer_name': 'Your Custom Layer Name',
        'description': 'Description of what this layer shows',
        'data_type': 'survey_data'  # or 'anomalies', 'processed', etc.
    }
)

# Flight path/line layer
result.add_layer_output(
    layer_type='flight_lines',
    data=path_dataframe,
    style_info={
        'line_color': '#FF6600',
        'line_width': 2,
        'line_opacity': 0.9
    },
    metadata={
        'layer_name': 'Survey Path',
        'description': 'Flight path for this survey'
    }
)

# Raster/grid layer
result.add_layer_output(
    layer_type='raster',
    data={
        'grid': numpy_array,  # 2D numpy array
        'bounds': [min_x, min_y, max_x, max_y],  # Spatial bounds
        'field_name': 'interpolated_field'
    },
    style_info={
        'use_graduated_colors': True,
        'opacity': 0.7
    },
    metadata={
        'layer_name': 'Interpolated Grid',
        'description': 'Interpolated field data'
    }
)
```

## 📈 Creating Visualizations

Include matplotlib figures that display in the UXO Wizard interface:

```python
import matplotlib.pyplot as plt

def execute(self, data, params, progress_callback=None, input_file_path=None):
    # ... your processing logic ...
    
    # Create a plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Example: time series plot
    if 'timestamp' in processed_data.columns:
        ax.plot(processed_data['timestamp'], processed_data['processed_value'])
        ax.set_xlabel('Time')
        ax.set_ylabel('Processed Value')
        ax.set_title('Processing Results Over Time')
        ax.grid(True)
    
    plt.tight_layout()
    
    # Add to result - UXO Wizard handles all file saving automatically
    result = ProcessingResult(
        success=True,
        data=processed_data,
        figure=fig  # Automatically saved as .mplplot and .png
    )
    
    return result
```

### Automatic Plot Management

When you include a `figure` in your `ProcessingResult`, UXO Wizard automatically:

1. **Displays the plot** in a new data viewer tab for immediate interaction
2. **Saves two formats**:
   - `.mplplot` - Interactive matplotlib plot that can be reopened
   - `.png` - High-resolution static image (300 DPI)
3. **Organizes files** in `project/processed/processor_type/` directory
4. **Generates unique names** with timestamps to prevent overwrites

## 🔧 Advanced Features

### Script Output Control

Control whether your script handles its own file output:

```python
class MyScript(ScriptInterface):
    @property
    def handles_own_output(self) -> bool:
        """Return True if script creates its own output files"""
        return False  # Default: let UXO Wizard handle file creation
        
    # If True, your script should create and organize its own output files
    # If False, UXO Wizard automatically creates CSV files from result.data
```

### Data Validation

Validate input data before processing:

```python
def validate_data(self, data: pd.DataFrame) -> bool:
    """Check if data is suitable for this script"""
    required_columns = ['latitude', 'longitude', 'magnetic_field']
    missing_cols = [col for col in required_columns if col not in data.columns]
    
    if missing_cols:
        raise ProcessingError(f"Missing required columns: {missing_cols}")
    
    if data.empty:
        raise ProcessingError("Input data cannot be empty")
    
    return True
```

### Progress Reporting

Keep users informed during long-running processes:

```python
def execute(self, data, params, progress_callback=None, input_file_path=None):
    if progress_callback:
        progress_callback(0, "Starting processing...")
    
    # ... processing step 1 ...
    
    if progress_callback:
        progress_callback(25, "Data validation complete")
    
    # ... processing step 2 ...
    
    if progress_callback:
        progress_callback(50, "Analysis in progress...")
    
    # ... processing step 3 ...
    
    if progress_callback:
        progress_callback(75, "Creating visualizations...")
    
    # ... final steps ...
    
    if progress_callback:
        progress_callback(100, "Processing complete!")
```

### Universal Interpolation Methods

Access UXO Wizard's built-in minimum curvature interpolation:

```python
from src.processing.base import BaseProcessor

class MyInterpolationScript(ScriptInterface):
    def __init__(self, project_manager=None):
        super().__init__(project_manager)
        # Create helper to access BaseProcessor methods
        self._interpolation_helper = _InterpolationHelper()
    
    def execute(self, data, params, progress_callback=None, input_file_path=None):
        # Extract coordinates and values
        x = data['longitude'].values
        y = data['latitude'].values  
        z = data['field_value'].values
        
        # Use built-in minimum curvature interpolation
        grid_x, grid_y, grid_z = self._interpolation_helper.minimum_curvature_interpolation(
            x, y, z,
            grid_resolution=150,
            constraint_mode='soft',  # Eliminates flight line artifacts
            progress_callback=progress_callback
        )
        
        # Create result with interpolated grid
        result = ProcessingResult(success=True, data=data)
        result.add_layer_output(
            layer_type='raster',
            data={'grid': grid_z, 'bounds': [x.min(), y.min(), x.max(), y.max()]},
            metadata={'layer_name': 'Interpolated Field'}
        )
        
        return result

# Helper class to access BaseProcessor methods
class _InterpolationHelper(BaseProcessor):
    def validate_data(self, data):
        return True
```

## 💡 Best Practices

### Safe Parameter Access
Always use `.get()` for safe parameter access with defaults:
```python
# Good - safe with defaults
resolution = params.get('interpolation_params', {}).get('grid_resolution', {}).get('value', 300)

# Bad - will crash if structure changes  
resolution = params['interpolation_params']['grid_resolution']['value']
```

### Data Handling
```python
# Always work on a copy
processed_data = data.copy()

# Validate required columns
required_columns = ['latitude', 'longitude', 'magnetic_field']
missing_cols = [col for col in required_columns if col not in data.columns]
if missing_cols:
    raise ProcessingError(f"Missing required columns: {missing_cols}")
```

### Unique Layer Names
Prevent layer overwrites with unique naming:
```python
import datetime

timestamp = datetime.datetime.now().strftime("%H:%M:%S")
if input_file_path:
    input_filename = Path(input_file_path).stem
    layer_name = f'{input_filename} - Your Processing ({timestamp})'
else:
    layer_name = f'Your Processing ({timestamp})'
```

### Error Handling
Provide specific, actionable error messages:
```python
# Good - specific and helpful
if 'magnetic_field' not in data.columns:
    available_fields = [col for col in data.columns if 'magnetic' in col.lower()]
    raise ProcessingError(f"Missing 'magnetic_field' column. Available magnetic columns: {available_fields}")

# Bad - generic and unhelpful  
if 'magnetic_field' not in data.columns:
    raise ProcessingError("Missing column")
```

## 🔍 Examples

Check out the `examples/` directory for:

- **`template_script.py`** - Basic template with all required methods
- **`magnetic_example.py`** - Advanced magnetic processing example with statistical analysis

## 🔧 Troubleshooting

### Script Not Loading
- ✅ Check that your file is in the correct processor directory (`magnetic/`, `gamma/`, etc.)
- ✅ Ensure `SCRIPT_CLASS = YourClassName` is at the bottom of your file
- ✅ Verify your class inherits from `ScriptInterface`
- ✅ Check the UXO Wizard log for import errors

### Import Errors
- ✅ Only use packages listed in the "Available Packages" section
- ✅ For new packages, open a GitHub issue with a package request

### Parameter Issues  
- ✅ Ensure parameter structure matches the examples above
- ✅ Check parameter types: `'float'`, `'int'`, `'bool'`, `'choice'`, `'file'`
- ✅ Verify `min`/`max` values are reasonable for numeric parameters

### Layer Not Appearing on Map
- ✅ Check that your data has coordinate columns (`latitude`/`longitude` or UTM)
- ✅ Verify `layer_type` is correct: `'points'`, `'raster'`, `'vector'`, `'flight_lines'`
- ✅ Ensure `result.add_layer_output()` is called with valid data
- ✅ Check that your DataFrame isn't empty

### Processing Errors
- ✅ Use try/catch blocks and return `ProcessingResult(success=False, error_message=...)`
- ✅ Test with small datasets first
- ✅ Add progress callbacks to identify where processing fails
- ✅ Check the console output for detailed error messages

## 🎨 Processor-Specific Styling

Use appropriate color schemes for your processor type:

### Magnetic Data
```python
magnetic_style = {
    'color_ramp': ["#000080", "#0000FF", "#00FFFF", "#00FF00", "#FFFF00", "#FF8000", "#FF0000"],
    'use_graduated_colors': True,
    'opacity': 0.8
}
```

### Gamma Radiation  
```python
gamma_style = {
    'color_ramp': ["#004400", "#008800", "#00CC00", "#CCCC00", "#CC8800", "#CC0000"],
    'use_graduated_colors': True,
    'opacity': 0.9
}
```

### GPR Data
```python
gpr_style = {
    'color_ramp': ["#000066", "#0066CC", "#66CCFF", "#CCCCCC", "#FFCC66", "#CC6600"],
    'use_graduated_colors': True,
    'opacity': 0.8
}
```

## 🌍 Coordinate Systems

UXO Wizard automatically handles Norwegian UTM coordinate systems:

- **UTM Zone 32N** (EPSG:25832): 6°E - 12°E
- **UTM Zone 33N** (EPSG:25833): 12°E - 18°E (default for Norway)  
- **UTM Zone 34N** (EPSG:25834): 18°E - 24°E
- **UTM Zone 35N** (EPSG:25835): 24°E - 30°E
- **WGS84** (EPSG:4326): Standard latitude/longitude

Your scripts should work with any of these coordinate systems - the framework handles detection and conversion automatically.

## 🚀 Performance Tips

### Vectorized Operations
```python
# Good - vectorized with numpy/pandas
distances = np.sqrt((data['x'] - target_x)**2 + (data['y'] - target_y)**2)

# Bad - loops  
distances = []
for _, row in data.iterrows():
    dist = np.sqrt((row['x'] - target_x)**2 + (row['y'] - target_y)**2)
    distances.append(dist)
```

### Large Dataset Handling
```python
def process_large_dataset(self, data, chunk_size=10000):
    """Process large datasets in chunks to manage memory"""
    results = []
    total_chunks = len(data) // chunk_size + 1
    
    for i, start_idx in enumerate(range(0, len(data), chunk_size)):
        if progress_callback:
            progress = (i / total_chunks) * 100
            progress_callback(progress, f"Processing chunk {i+1}/{total_chunks}")
        
        chunk = data.iloc[start_idx:start_idx + chunk_size]
        processed_chunk = self._process_chunk(chunk)
        results.append(processed_chunk)
    
    return pd.concat(results, ignore_index=True)
```

## 🤝 Contributing

Found a useful script? Consider sharing it with the community:

1. **Fork the UXO Wizard repository**
2. **Add your script** to the appropriate examples directory  
3. **Include documentation** and test data
4. **Submit a pull request**

## 📞 Support

- **Script Help**: Check existing issues or start a discussion on GitHub
- **Bug Reports**: Open an issue with your script and error details
- **Feature Requests**: Use the GitHub issue tracker
- **Documentation**: All examples include inline comments and explanations

## 🔗 External Script Validation

When you create or modify scripts, UXO Wizard automatically validates them:

- **✅ Syntax Check**: Ensures your Python code is valid
- **✅ Import Validation**: Verifies all imports are available packages
- **✅ Structure Check**: Confirms ScriptInterface implementation
- **⚠️ Security Warnings**: Flags potentially dangerous code patterns

Invalid scripts are logged with specific error messages to help you fix issues quickly.

---

**Ready to get started?** Copy `examples/template_script.py` to your processor directory, customize it for your needs, and watch your custom script appear in UXO Wizard with a 🟢 green user script indicator!

Happy scripting! 🎉
