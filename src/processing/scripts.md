# UXO Wizard: Developer's Guide to Creating Processing Scripts

## 1. Introduction

Welcome to the UXO Wizard scripting framework! This guide is designed for developers who want to extend the capabilities of the UXO Wizard application by creating custom data processing scripts. By following these best practices, you can ensure your scripts integrate seamlessly with the application's data processing pipeline, user interface, and data visualization features.

The framework is built around a modular architecture that separates high-level data type "Processors" (e.g., `MagneticProcessor`) from the specific "Scripts" that perform the actual data manipulation (e.g., `MagbaseProcessing`). This allows for a flexible and extensible system where new processing algorithms can be added without modifying the core application.

This document will walk you through the architecture, the essential components you need to implement, and provide a step-by-step guide to creating a new script, using `magbase_processing.py` as our primary example.

## 2. System Architecture Overview

The data processing workflow in UXO Wizard is managed by a few key components defined in the provided source files:

* **`ProcessingPipeline` (`pipeline.py`):** The central coordinator. It discovers and manages the different `Processors`, handles running them in background threads, and manages the flow of data from input to output, including file generation.
* **`BaseProcessor` (`base.py`):** An abstract base class for data-type-specific processors (e.g., `MagneticProcessor`, `GPRProcessor`). Its primary role is to discover and manage the processing scripts associated with its data type (e.g., the `MagneticProcessor` finds all scripts in the `src/processing/scripts/magnetic/` directory).
* **`ScriptInterface` (`base.py`):** The most important component for a script developer. This is an abstract base class that defines the contract all processing scripts must follow. Your custom script class will inherit from `ScriptInterface` and implement its abstract methods.
* **`ProcessingResult` (`base.py`):** A data class used to standardize the output of all processing scripts. It contains the processed data, success status, metadata, and any generated output files or visualization layers.
* **`magbase_processing.py`:** A concrete implementation of a `ScriptInterface` for magnetic data. It serves as an excellent, comprehensive example of how to build a script that includes file inputs, complex parameters, data validation, and advanced processing.

The general workflow is as follows:
1.  The `ProcessingPipeline` identifies the type of data loaded by the user.
2.  It selects the appropriate `Processor` (e.g., `MagneticProcessor`).
3.  The `Processor` discovers all available `ScriptInterface` implementations in its designated script directory.
4.  The user selects a script and configures its parameters through the UI.
5.  The `Processor` executes the `execute` method of the selected script.
6.  The script performs the processing and returns a `ProcessingResult` object.
7.  The `ProcessingPipeline` receives the result and handles file saving and metadata generation.

## 3. Creating a New Processing Script

Creating a new script involves creating a new Python file within the appropriate subdirectory and implementing a class that adheres to the `ScriptInterface`.

### Step 1: File and Class Setup

1.  **File Location:** Your script must be placed in the correct directory for it to be automatically discovered. The path is `src/processing/scripts/<processor_type>/`, where `<processor_type>` is the lowercase name of the processor (e.g., `magnetic`, `gpr`).
    * For a new magnetic script, create a file like `src/processing/scripts/magnetic/my_new_script.py`.

2.  **Class Definition:** Inside your new file, define a class that inherits from `ScriptInterface`.

    ```python
    # src/processing/scripts/magnetic/my_new_script.py
    from src.processing.base import ScriptInterface, ProcessingResult, ProcessingError
    import pandas as pd
    from typing import Dict, Any, Optional, Callable

    class MyNewScript(ScriptInterface):
        # ... implementation ...
        pass

    # IMPORTANT: Export the class for discovery
    SCRIPT_CLASS = MyNewScript
    ```

    The `SCRIPT_CLASS = MyNewScript` line at the end of the file is **mandatory**. The `BaseProcessor` uses this to find and instantiate your script.

### Step 2: Implement the `ScriptInterface`

You must implement all the abstract methods and properties of the `ScriptInterface`.

#### `name` and `description`

These properties provide human-readable text for the UI.

```python
@property
def name(self) -> str:
    return "My New Awesome Script"

@property
def description(self) -> str:
    return "This script does amazing things to magnetic data."
```

#### `get_parameters()`

This method defines the parameters your script needs, which the application will use to automatically generate a settings panel in the UI. The structure is a dictionary of dictionaries.

* **Key Concepts:**
    * The top-level keys (e.g., `processing_options`) create collapsible groups in the UI.
    * Each parameter has a `value` (the default), a `type` (`float`, `int`, `bool`, `choice`, `file`), and a `description`.
    * Numeric types can have `min` and `max` values.
    * `choice` types require a `choices` list.
    * `file` types can specify `file_types`.

* **Example from `magbase_processing.py`:**

    ```python
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'file_inputs': {
                'base_station_file': {
                    'value': '',
                    'type': 'file',
                    'file_types': ['.txt', '.csv'],
                    'description': 'GSM-19 base station data file for diurnal correction'
                }
            },
            'sensor_configuration': {
                'sensor1_offset_east': {
                    'value': 0.0375,
                    'type': 'float',
                    'min': -5.0,
                    'max': 5.0,
                    'description': 'Sensor 1 East offset from GPS (meters)'
                }
            },
            'processing_options': {
                'vertical_alignment': {
                    'value': False,
                    'type': 'bool',
                    'description': 'Calculate vertical gradient (M1-M2)'
                },
                'sampling_mode': {
                    'value': 'interpolate',
                    'type': 'choice',
                    'choices': ['interpolate', 'downsample'],
                    'description': 'Choose how to handle magnetometer data without GPS points'
                }
            }
        }
    ```

#### `validate_data()`

This optional method allows your script to check if the input `DataFrame` is suitable for processing *before* the `execute` method is called. If the data is not valid, you should raise a `ProcessingError` with a descriptive message.

* **Example:**

    ```python
    def validate_data(self, data: pd.DataFrame) -> bool:
        if data.empty:
            raise ProcessingError("Input data cannot be empty.")
        if 'Btotal1 [nT]' not in data.columns:
            raise ProcessingError("Missing required column: 'Btotal1 [nT]'.")
        return True
    ```

#### `execute()`

This is the core of your script where all the processing happens.

* **Signature:** `execute(self, data: pd.DataFrame, params: Dict[str, Any], progress_callback: Optional[Callable] = None, input_file_path: Optional[str] = None) -> ProcessingResult`

* **Arguments:**
    * `data`: The input pandas `DataFrame`.
    * `params`: A dictionary containing the current values of the parameters you defined in `get_parameters()`.
    * `progress_callback`: A function to report progress back to the UI. Call it with `progress_callback(percentage, "message")`.
    * `input_file_path`: The path to the original file the user loaded. Useful for generating output filenames.

* **Logic:**
    1.  **Start with a progress update:** `if progress_callback: progress_callback(0, "Starting processing...")`
    2.  **Extract parameters:** Access the parameter values from the `params` dictionary.
    3.  **Perform your data manipulation:** This is your main algorithm. Use the `data` DataFrame as input.
    4.  **Report progress frequently:** Call `progress_callback` at key milestones in your processing.
    5.  **Handle errors:** Wrap your code in a `try...except` block. If an error occurs, return a failed `ProcessingResult`.
    6.  **Return a `ProcessingResult`:** On success, populate and return a `ProcessingResult` object.

### Step 3: Returning Results

The `ProcessingResult` object is crucial for communicating the outcome of your script back to the application.

* **Basic `ProcessingResult` on Success:**

    ```python
    # At the end of your execute method
    processed_df = ... # your final DataFrame

    result = ProcessingResult(
        success=True,
        data=processed_df, # The primary data output
        processing_script=self.name
    )
    return result
    ```

* **`ProcessingResult` on Failure:**

    ```python
    # In your except block
    except Exception as e:
        return ProcessingResult(
            success=False,
            error_message=f"An unexpected error occurred: {str(e)}"
        )
    ```

* **Advanced `ProcessingResult` Features:**

    The `magbase_processing.py` script demonstrates how to add rich outputs for visualization and file generation.

    * **Adding Metadata:** Include any relevant information about the processing run. This will be saved in a `.json` sidecar file.

        ```python
        result.metadata = {
            'anomalies_found': 10,
            'utm_zone': 33,
            'parameters': params # Good practice to include the params used
        }
        ```

    * **Adding Output Files:** If your script generates files (like images, reports, etc.), you can register them. The `pipeline` will use the `file_path` from the first registered file with a common data extension (csv, xlsx, json) as the primary output filename.

        ```python
        # Assuming you have created a plot and saved it
        result.add_output_file(
            file_path="path/to/my_plot.png",
            file_type="png",
            description="A beautiful plot of the processed data."
        )
        ```

    * **Adding Visualization Layers:** The enhanced layer system now automatically converts LayerOutput objects to UXOLayer objects for real-time map visualization.

        ```python
        # Basic point layer with custom name
        result.add_layer_output(
            layer_type='points',
            data=processed_df,
            style_info={
                'color_field': 'R1 [nT]',
                'use_graduated_colors': True,
                'color_scheme': 'magnetic',
                'size': 4,
                'opacity': 0.8
            },
            metadata={
                'description': 'Processed magnetic readings with residual anomalies',
                'total_points': len(processed_df),
                'data_type': 'magnetic_residuals',
                'layer_name': 'Magnetic Residuals'  # Custom layer name
            }
        )
        
        # Flight path layer
        result.add_layer_output(
            layer_type='flight_lines',
            data=flight_path_df,
            style_info={
                'line_color': '#FF6600',
                'line_width': 2,
                'line_opacity': 0.9
            },
            metadata={
                'description': 'Survey flight path',
                'data_type': 'flight_path'
            }
        )
        
        # Anomaly highlights
        result.add_layer_output(
            layer_type='processed',
            data=anomaly_df,
            style_info={
                'color': '#FF0000',
                'size': 8,
                'opacity': 1.0,
                'show_labels': True,
                'label_field': 'anomaly_strength'
            },
            metadata={
                'description': 'Detected magnetic anomalies',
                'data_type': 'magnetic_anomalies',
                'anomaly_threshold': 2.0
            }
        )
        ```

### Returning Plots

Your script can generate and return Matplotlib plots for visualization in the UXO Wizard interface.

1.  **Import `matplotlib`**: Make sure `matplotlib.pyplot` is imported.
2.  **Create a Plot**: Generate your plot as you normally would, which creates a `Figure` object.
3.  **Return the Figure**: Assign the `Figure` object to the `figure` attribute of your `ProcessingResult`.

The application will automatically handle the returned plot, allowing the user to view it interactively or save it.

**Example:**

```python
import matplotlib.pyplot as plt
from ...base import ProcessingResult

def run(data, parameters):
    # Your data processing logic...
    processed_df = data.copy()
    processed_df['new_col'] = processed_df['some_col'] * 2

    # 1. & 2. Create a plot
    fig, ax = plt.subplots()
    ax.plot(processed_df['timestamp'], processed_df['new_col'])
    ax.set_title("My Processing Result")
    fig.tight_layout()

    # 3. Return the figure in the result
    return ProcessingResult(
        success=True,
        data=processed_df,
        figure=fig, # <-- Here is the plot
        message="Processing complete with a plot."
    )
```

## 4. Enhanced Layer Generation System

The UXO-Wizard now includes a powerful modular layer generation system that automatically converts your processing outputs into interactive map layers. This system provides universal layer creation methods that all processors inherit from the `BaseProcessor` class.

### 4.1 Automatic Layer Creation

When your script generates `LayerOutput` objects using `result.add_layer_output()`, the system automatically:

1. **Converts to UXOLayer objects**: LayerOutput → ProcessingWorker → UXOLayer
2. **Applies intelligent styling**: Based on processor type and data characteristics
3. **Detects coordinate systems**: Automatic Norwegian UTM zone detection (32-35)
4. **Groups layers logically**: By processor type and data type in LayerManager
5. **Displays in real-time**: Layers appear immediately in the map interface
6. **Uses custom names**: Specify `layer_name` in metadata for custom layer names

### 4.2 Supported Layer Types

#### Point Layers (`layer_type='points'`)
For coordinate-based data points:
```python
result.add_layer_output(
    layer_type='points',
    data=dataframe_with_coords,
    style_info={
        'color_field': 'value_column',     # Column for color mapping
        'use_graduated_colors': True,      # Enable data-driven colors
        'color_scheme': 'magnetic',        # Color scheme for processor type
        'size': 4,                         # Point size
        'opacity': 0.8,                    # Point opacity
        'enable_clustering': True          # Cluster dense points
    },
    metadata={
        'description': 'Layer description',
        'coordinate_columns': {'latitude': 'lat_col', 'longitude': 'lon_col'},
        'total_points': len(dataframe_with_coords),
        'data_type': 'descriptive_type'
    }
)
```

#### Vector Layers (`layer_type='flight_lines'` or `layer_type='vector'`)
For lines and paths:
```python
result.add_layer_output(
    layer_type='flight_lines',
    data=gps_track_dataframe,
    style_info={
        'line_color': '#FF6600',
        'line_width': 2,
        'line_opacity': 0.9,
        'show_labels': False
    },
    metadata={
        'description': 'Survey flight path',
        'data_type': 'flight_path'
    }
)
```

#### Raster Layers (`layer_type='grid_visualization'` or `layer_type='raster'`)
For gridded/interpolated data:
```python
result.add_layer_output(
    layer_type='grid_visualization',
    data=numpy_grid_array,
    style_info={
        'use_graduated_colors': True,
        'color_scheme': 'amplitude',
        'opacity': 0.7
    },
    metadata={
        'description': 'Interpolated field data',
        'grid_shape': numpy_grid_array.shape,
        'bounds': [min_x, min_y, max_x, max_y],
        'data_type': 'interpolated_grid'
    }
)
```

#### Processed Layers (`layer_type='processed'`)
For derived/analyzed data (anomalies, features):
```python
result.add_layer_output(
    layer_type='processed',
    data=anomaly_dataframe,
    style_info={
        'color': '#FF0000',
        'size': 8,
        'opacity': 1.0,
        'show_labels': True,
        'label_field': 'anomaly_id'
    },
    metadata={
        'description': 'Detected anomalies',
        'data_type': 'anomalies',
        'detection_threshold': threshold_value
    }
)
```

### 4.3 Inherited Layer Methods

All processors inherit universal layer generation methods from `BaseProcessor`. While you typically use `result.add_layer_output()`, these methods are available for advanced use cases:

```python
# In your script's execute method
def execute(self, data, params, progress_callback=None, input_file_path=None):
    # Standard processing...
    result = ProcessingResult(success=True)
    
    # Multiple layer generation approaches:
    
    # Approach 1: Standard LayerOutput (recommended)
    result.add_layer_output(
        layer_type='points',
        data=processed_data,
        style_info={'color': 'blue', 'size': 4},
        metadata={'description': 'Processed data'}
    )
    
    # Approach 2: Advanced multi-layer generation
    self._generate_comprehensive_layers(processed_data, result, params)
    
    return result

def _generate_comprehensive_layers(self, data, result, params):
    """Generate multiple related layers for comprehensive visualization"""
    
    # Main data layer
    result.add_layer_output(
        layer_type='points',
        data=data,
        style_info={
            'color_field': 'primary_value',
            'use_graduated_colors': True,
            'size': 4
        },
        metadata={'description': 'Main dataset', 'data_type': 'primary'}
    )
    
    # Filtered anomalies
    anomalies = data[data['primary_value'] > data['primary_value'].quantile(0.95)]
    if len(anomalies) > 0:
        result.add_layer_output(
            layer_type='processed',
            data=anomalies,
            style_info={'color': '#FF0000', 'size': 8},
            metadata={'description': 'High-value anomalies', 'data_type': 'anomalies'}
        )
    
    # Survey path (if GPS track available)
    if self._has_continuous_gps(data):
        flight_path = self._create_flight_path(data)
        result.add_layer_output(
            layer_type='flight_lines',
            data=flight_path,
            style_info={'line_color': '#FF6600', 'line_width': 2},
            metadata={'description': 'Survey path', 'data_type': 'flight_path'}
        )
```

### 4.4 Styling Guidelines by Processor Type

#### Magnetic Data
```python
style_info = {
    'color_scheme': 'magnetic',  # Blue → Red magnetic scale
    'color_ramp': ["#000080", "#0000FF", "#00FFFF", "#00FF00", "#FFFF00", "#FF8000", "#FF0000"],
    'size': 4,
    'opacity': 0.8
}
```

#### Gamma Radiation
```python
style_info = {
    'color_scheme': 'gamma',  # Green → Red radiation scale
    'color_ramp': ["#004400", "#008800", "#00CC00", "#CCCC00", "#CC8800", "#CC0000"],
    'size': 5,
    'opacity': 0.9
}
```

#### GPR Data
```python
style_info = {
    'color_scheme': 'gpr',  # Navy → Brown amplitude scale
    'color_ramp': ["#000066", "#0066CC", "#66CCFF", "#CCCCCC", "#FFCC66", "#CC6600"],
    'size': 4,
    'opacity': 0.8
}
```

### 4.5 Norwegian UTM Coordinate Support

The system automatically handles Norwegian UTM zones (32, 33, 34, 35):

```python
# Coordinate system detection is automatic
# Your script should work with lat/lon or UTM columns as-is

# UTM zone detection logic (handled automatically):
# Zone 32: 6°E - 12°E
# Zone 33: 12°E - 18°E  (default for Norway)
# Zone 34: 18°E - 24°E
# Zone 35: 24°E - 30°E

# The system will detect and set the appropriate CRS:
# EPSG:25832 (UTM 32N), EPSG:25833 (UTM 33N), 
# EPSG:25834 (UTM 34N), EPSG:25835 (UTM 35N)
```

### 4.6 Layer Grouping and Organization

Layers are automatically organized in the LayerManager based on:

1. **Processor type**: From `metadata['processor_type']`
2. **Processing history**: From layer processing lineage
3. **Data type**: From `metadata['data_type']`

Example layer groups:
- "Magnetic Processing" (for magnetic processor outputs)
- "Survey Data" (for raw data visualizations)
- "Annotations" (for user-added layers)

### 4.7 Performance Considerations

- **Point limits**: Keep point layers under 10,000 points for performance
- **Clustering**: Enable clustering for dense point data
- **Subsampling**: Create overview layers for large datasets
- **Memory**: Large raster layers should be optimized or tiled

## 5. Best Practices and Recommendations

### 5.1 Essential Requirements

#### **CRITICAL: Always Set Processor Metadata**
```python
# MUST include processor type in metadata to prevent output directory conflicts
result.metadata.update({
    'processor': 'magnetic',  # or 'gpr', 'gamma', etc.
    # ... other metadata
})
```

**Why this matters:** Without the processor metadata, scripts default to `processed/unknown/` directory, creating duplicate outputs and confusion. This is a common source of file organization issues.

#### **Output Directory Structure**
Scripts should create outputs in the project's `processed/` directory structure:
```
project_directory/
├── processed/
│   ├── magnetic/
│   │   ├── filename_segmented/     # Flight path segmenter
│   │   ├── filename_interpolated/  # Grid interpolator
│   │   └── filename_processed/     # Other magnetic scripts
│   ├── gpr/
│   └── gamma/
```

Use the project manager's working directory:
```python
def _create_output_directory(self, input_file_path: Optional[str]) -> Path:
    """Create output directory in project/processed/processor_type/"""
    if input_file_path:
        input_path = Path(input_file_path)
        base_filename = input_path.stem
        
        # Find project root directory
        project_dir = input_path.parent
        while project_dir != project_dir.parent:
            if (project_dir / "processed").exists() or len(list(project_dir.glob("*.uxo"))) > 0:
                break
            project_dir = project_dir.parent
        
        # Create project/processed/processor_type/filename_suffix structure
        output_dir = project_dir / "processed" / "magnetic" / f"{base_filename}_processed"
    else:
        # Fallback to temporary directory
        temp_dir = tempfile.mkdtemp(prefix="processor_")
        output_dir = Path(temp_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
```

### 5.2 Core Development Practices

#### **Data Handling**
* **Immutability:** Always create a copy of input data: `df = data.copy()`
* **Robust Parameter Access:** Use `.get()` for safe parameter access:
  ```python
  # Good
  grid_resolution = interp_params.get('grid_resolution', {}).get('value', 300)
  
  # Bad - will fail if structure changes
  grid_resolution = interp_params['grid_resolution']['value']
  ```
* **Column Validation:** Check for required columns before processing:
  ```python
  required_columns = ['Latitude [Decimal Degrees]', 'Longitude [Decimal Degrees]']
  missing_columns = [col for col in required_columns if col not in data.columns]
  if missing_columns:
      raise ProcessingError(f"Missing required columns: {missing_columns}")
  ```

#### **Progress Reporting**
Provide meaningful progress updates for user feedback:
```python
def execute(self, data, params, progress_callback=None, input_file_path=None):
    if progress_callback:
        progress_callback(0, "Starting processing...")
    
    # ... processing steps ...
    
    if progress_callback:
        progress_callback(0.3, "Interpolating grid...")
    
    # ... more processing ...
    
    if progress_callback:
        progress_callback(0.8, "Creating visualizations...")
    
    # ... final steps ...
    
    if progress_callback:
        progress_callback(1.0, "Processing complete!")
```

### 5.3 Layer Creation Best Practices

#### **Unique Layer Names**
Always create unique layer names to prevent overwrites:
```python
import datetime

# Create unique names with timestamp and input filename
timestamp = datetime.datetime.now().strftime("%H:%M:%S")
if input_file_path:
    input_filename = Path(input_file_path).stem
    layer_name = f'{input_filename} - Grid Interpolation ({timestamp})'
else:
    layer_name = f'Grid Interpolation ({timestamp})'

result.add_layer_output(
    layer_type='raster',
    data=grid_data,
    style_info={},
    metadata={'layer_name': layer_name}
)
```

#### **Toggleable Features**
Make expensive or optional features toggleable:
```python
def get_parameters(self):
    return {
        'output_parameters': {
            'include_original_points': {
                'value': False,  # Default to False for performance
                'type': 'bool',
                'description': 'Include original data points as a map layer'
            },
            'create_diagnostics': {
                'value': True,
                'type': 'bool', 
                'description': 'Generate diagnostic plots and analysis'
            }
        }
    }

def execute(self, data, params, progress_callback=None, input_file_path=None):
    # Extract toggleable parameters
    include_original = params.get('output_parameters', {}).get('include_original_points', {}).get('value', False)
    
    # Only create expensive layers if requested
    if include_original:
        if progress_callback:
            progress_callback(0.9, "Creating original data points layer...")
        # ... create original points layer ...
```

### 5.4 Error Handling and Validation

#### **Specific Error Messages**
```python
# Good - specific and actionable
if 'Btotal1 [nT]' not in data.columns:
    available_cols = [col for col in data.columns if 'nT' in col]
    raise ProcessingError(f"Missing magnetic field column 'Btotal1 [nT]'. Available magnetic columns: {available_cols}")

# Bad - generic and unhelpful
if 'Btotal1 [nT]' not in data.columns:
    raise ProcessingError("Missing required column")
```

#### **Graceful Degradation**
```python
# Try advanced features, fall back to basic if needed
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    NUMBA_AVAILABLE = False

# Use in processing
if NUMBA_AVAILABLE:
    result = self._fast_interpolation(data)
else:
    result = self._basic_interpolation(data)
```

### 5.5 Performance Optimization

#### **Efficient Data Processing**
```python
# Use vectorized operations
import numpy as np

# Good - vectorized
distances = np.sqrt((x - target_x)**2 + (y - target_y)**2)

# Bad - loops
distances = []
for i in range(len(x)):
    dist = np.sqrt((x[i] - target_x)**2 + (y[i] - target_y)**2)
    distances.append(dist)
```

#### **Memory Management**
```python
# For large datasets, process in chunks
def process_large_dataset(self, data, chunk_size=10000):
    results = []
    for i in range(0, len(data), chunk_size):
        chunk = data.iloc[i:i+chunk_size]
        processed_chunk = self._process_chunk(chunk)
        results.append(processed_chunk)
    return pd.concat(results, ignore_index=True)
```

### 5.6 Common Pitfalls to Avoid

#### **1. Missing Processor Metadata**
```python
# WRONG - will create processed/unknown/ directory
result = ProcessingResult(
    success=True,
    data=processed_data,
    metadata={'custom_field': 'value'}
)

# CORRECT - properly categorized
result = ProcessingResult(
    success=True,
    data=processed_data,
    metadata={
        'processor': 'magnetic',  # Essential!
        'custom_field': 'value'
    }
)
```

#### **2. Layer Name Conflicts**
```python
# WRONG - will overwrite previous layers
result.add_layer_output(
    layer_type='raster',
    data=grid_data,
    metadata={'layer_name': 'Grid Interpolation'}  # Static name
)

# CORRECT - unique names
timestamp = datetime.datetime.now().strftime("%H:%M:%S")
input_filename = Path(input_file_path).stem if input_file_path else 'data'
layer_name = f'{input_filename} - Grid Interpolation ({timestamp})'
```

#### **3. Unsafe Parameter Access**
```python
# WRONG - will crash if structure changes
resolution = params['interpolation_parameters']['grid_resolution']['value']

# CORRECT - safe access with defaults
resolution = params.get('interpolation_parameters', {}).get('grid_resolution', {}).get('value', 300)
```

#### **4. Poor Progress Reporting**
```python
# WRONG - no user feedback
def execute(self, data, params, progress_callback=None, input_file_path=None):
    # ... long processing with no updates ...
    result = ProcessingResult(success=True, data=processed_data)
    return result

# CORRECT - regular progress updates
def execute(self, data, params, progress_callback=None, input_file_path=None):
    if progress_callback:
        progress_callback(0, "Starting processing...")
    
    # ... processing step 1 ...
    
    if progress_callback:
        progress_callback(0.3, "Interpolating grid...")
    
    # ... processing step 2 ...
    
    if progress_callback:
        progress_callback(1.0, "Processing complete!")
```

### 5.7 Testing and Debugging

#### **Logging for Development**
```python
from loguru import logger

def execute(self, data, params, progress_callback=None, input_file_path=None):
    logger.info(f"Starting {self.name} with {len(data)} data points")
    logger.debug(f"Parameters: {params}")
    
    # ... processing ...
    
    logger.info(f"Processing complete. Generated {len(result.layer_outputs)} layers")
```

#### **Parameter Validation**
```python
def validate_parameters(self, params):
    """Validate parameters before processing"""
    grid_resolution = params.get('interpolation_parameters', {}).get('grid_resolution', {}).get('value', 300)
    
    if grid_resolution < 10 or grid_resolution > 1000:
        raise ProcessingError(f"Grid resolution {grid_resolution} is outside valid range (10-1000)")
    
    return True
```

### 5.8 Documentation and Maintenance

#### **Clear Parameter Descriptions**
```python
def get_parameters(self):
    return {
        'interpolation_parameters': {
            'grid_resolution': {
                'value': 300,
                'type': 'int',
                'min': 10,
                'max': 1000,
                'description': 'Number of grid points per axis (total grid = resolution²). Higher values = more detail but slower processing.'
            }
        }
    }
```

#### **Comprehensive Metadata**
```python
result.metadata.update({
    'processor': 'magnetic',
    'processing_method': 'minimum_curvature',
    'grid_resolution': grid_resolution,
    'field_processed': field_column,
    'data_points': len(data),
    'numba_acceleration': NUMBA_AVAILABLE,
    'processing_time': time.time() - start_time
})
```

By following these practices, you'll create robust, maintainable, and well-integrated processing scripts that provide excellent user experience and reliable results in the UXO Wizard framework.