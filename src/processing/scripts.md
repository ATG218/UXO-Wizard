# UXO Wizard: Developer's Guide to Creating Processing Scripts

## 1. Introduction

Welcome to the UXO Wizard scripting framework! This guide is designed for developers who want to extend the capabilities of the UXO Wizard application by creating custom data processing scripts. By following these best practices, you can ensure your scripts integrate seamlessly with the application's data processing pipeline, user interface, and data visualization features.

The framework is built around a modular architecture that separates high-level data type "Processors" (e.g., `MagneticProcessor`) from the specific "Scripts" that perform the actual data manipulation (e.g., `MagbaseProcessing`). This allows for a flexible and extensible system where new processing algorithms can be added without modifying the core application.

This document will walk you through the architecture, the essential components you need to implement, and provide a step-by-step guide to creating a new script, using `magbase_processing.py` as our primary example.

## 2. System Architecture Overview

The data processing workflow in UXO Wizard is managed by a few key components defined in the provided source files:

### 2.1 Core Components

The framework consists of several interconnected components:

* **`ProcessingPipeline` (`pipeline.py`):** The central coordinator. It discovers and manages the different `Processors`, handles running them in background threads, and manages the flow of data from input to output, including file generation.
* **`BaseProcessor` (`base.py`):** An abstract base class for data-type-specific processors (e.g., `MagneticProcessor`, `GPRProcessor`). Its primary role is to discover and manage the processing scripts associated with its data type (e.g., the `MagneticProcessor` finds all scripts in the `src/processing/scripts/magnetic/` directory).
* **`ScriptInterface` (`base.py`):** The most important component for a script developer. This is an abstract base class that defines the contract all processing scripts must follow. Your custom script class will inherit from `ScriptInterface` and implement its abstract methods.
* **`ProcessingResult` (`base.py`):** A data class used to standardize the output of all processing scripts. It contains the processed data, success status, metadata, matplotlib figures, and any generated output files or visualization layers.
* **Processing Scripts:** Concrete implementations of `ScriptInterface` for specific processing tasks:
  - `basic_processing.py` - Simple magnetic data processing with 2D/3D plotting
  - `grid_interpolator.py` - Advanced grid interpolation with minimum curvature
  - `magbase_processing.py` - Comprehensive magnetic base station processing

### 2.2 Current Working Framework (2025)

The current framework has been enhanced with several key features:

#### **Automatic File Management**
- **Plot Auto-saving**: All matplotlib figures are automatically saved as `.mplplot` (interactive) and `.png` (static) files
- **Directory Structure**: Files are organized as `{project_root}/processed/{processor_type}/filename_script_timestamp.ext`
- **Unique Naming**: Timestamp-based filenames prevent overwrites
- **No User Prompts**: All file operations happen automatically

#### **Enhanced UI Integration**
- **Data Viewer**: Plots automatically open in new tabs when generated
- **Lab Widget**: Browse and double-click `.mplplot` files to reopen them
- **Project Explorer**: Navigate saved plots in the file tree
- **Interactive Plots**: Full matplotlib interactivity preserved in saved plots

#### **Processing Pipeline Improvements**
- **Background Processing**: Scripts run in separate threads to prevent UI freezing
- **Progress Reporting**: Real-time progress updates during processing
- **Error Handling**: Comprehensive error reporting and graceful failure handling
- **Smart Output Management**: Avoids redundant file creation for scripts that handle their own outputs
- **Integrated Metadata**: Processing metadata is embedded directly into results instead of separate JSON files

#### **Enhanced Script Experience (2025)**
- **Rich Script Tooltips**: Hover over scripts in the dropdown to see detailed metadata including description, tags, runtime estimates, and typical use cases
- **Recent Script Prioritization**: Recently used scripts appear at the top of the dropdown for quick access
- **Smart Default Selection**: The most recently used script is automatically selected when opening the processing dialog
- **Quick Re-run**: Right-click any file in Project Explorer and select "Run Previous (script_name)" to instantly re-run the last used processing script with saved parameters
- **Session Memory**: The system remembers your last used script and parameters during the current session
- **Persistent Recent Scripts**: Recent script usage is saved to application settings and persists across restarts (limited to 5 most recent per processor type)

#### **Layer System Integration**
- **Automatic Layer Creation**: Processing outputs automatically become map layers
- **Norwegian UTM Support**: Automatic coordinate system detection (zones 32-35)
- **Custom Layer Names**: Timestamped unique layer names prevent overwrites
- **Real-time Visualization**: Layers appear immediately in the map interface

The general workflow is as follows:
1.  The `ProcessingPipeline` identifies the type of data loaded by the user.
2.  It selects the appropriate `Processor` (e.g., `MagneticProcessor`).
3.  The `Processor` discovers all available `ScriptInterface` implementations in its designated script directory.
4.  The user selects a script and configures its parameters through the UI.
5.  The `Processor` executes the `execute` method of the selected script in a background thread.
6.  The script performs the processing and returns a `ProcessingResult` object.
7.  The `ProcessingPipeline` receives the result and handles:
   - Automatic data file saving to `processed/{processor_type}/`
   - Metadata generation (JSON sidecar files)
   - **Automatic plot saving** (.mplplot and .png formats)
   - Layer creation for map visualization
   - UI updates with results

### 2.3 Script Development in 2025

When developing scripts for the current framework, you benefit from:

#### **Zero-Configuration Plot Management**
- Simply return a `matplotlib.figure.Figure` in your `ProcessingResult`
- No need to handle file saving, user prompts, or file management
- Automatic integration with the data viewer and lab widget

#### **Robust Error Handling**
- Scripts run in background threads with comprehensive error handling
- Progress callbacks provide real-time user feedback
- Graceful failure handling prevents application crashes

#### **Enhanced Developer Experience**
- Clear script examples in `basic_processing.py` and `grid_interpolator.py`
- Comprehensive parameter system with automatic UI generation
- Detailed logging and debugging support
- Automatic file organization and metadata generation

#### **Integration with UXO Wizard Ecosystem**
- Scripts automatically integrate with the map system
- Layer outputs appear in real-time on the map
- File outputs are organized in the project's processed directory
- Full integration with the lab widget for result browsing

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

Your script can generate and return Matplotlib plots for visualization in the UXO Wizard interface. The framework provides automatic plot handling with seamless integration.

#### **Automatic Plot Management**

When your script returns a `ProcessingResult` with a `figure`, the framework automatically:

1. **Displays the plot** in a new data viewer tab for immediate interaction
2. **Saves the plot** to the `processed/{processor_type}/` directory in two formats:
   - `.mplplot` - Interactive matplotlib plot (pickle format) for reopening in the data viewer
   - `.png` - Static high-resolution image (300 DPI) for reports and documentation
3. **Generates unique filenames** using the pattern: `{input_filename}_{script_name}_{timestamp}.{extension}`
4. **No user prompts** - saving happens automatically in the background

#### **Creating Plots in Your Script**

**Basic Plot Example:**
```python
import matplotlib.pyplot as plt
from src.processing.base import ProcessingResult

def execute(self, data: pd.DataFrame, params: Dict[str, Any], 
            progress_callback: Optional[Callable] = None, input_file_path: Optional[str] = None) -> ProcessingResult:
    
    # Your data processing logic...
    processed_df = data.copy()
    processed_df['anomaly_field'] = processed_df['magnetic_field'] - processed_df['magnetic_field'].mean()
    
    # Create a plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(processed_df['timestamp'], processed_df['anomaly_field'])
    ax.set_title("Magnetic Anomaly Analysis")
    ax.set_xlabel("Time")
    ax.set_ylabel("Magnetic Field Anomaly [nT]")
    ax.grid(True)
    fig.tight_layout()
    
    # Return the figure in the result
    result = ProcessingResult(
        success=True,
        data=processed_df,
        figure=fig,  # <-- Framework handles everything from here
        processing_script=self.name
    )
    
    # Essential: Include processor metadata for correct file organization
    result.metadata.update({
        'processor': 'magnetic',  # Files saved to processed/magnetic/
        'processing_method': 'anomaly_detection',
        'data_points': len(processed_df)
    })
    
    return result
```

#### **Advanced Plot Examples**

**2D and 3D Plot Generation:**
```python
def execute(self, data, params, progress_callback=None, input_file_path=None):
    # ... data processing ...
    
    # Check if 3D plot is requested
    plot_type = params.get('output_options', {}).get('plot_type', {}).get('value', '2D')
    
    if plot_type == '3D':
        # Create 3D plot
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # Use first 3 numeric columns for 3D visualization
        x_col, y_col, z_col = numeric_columns[:3]
        scatter = ax.scatter(data[x_col], data[y_col], data[z_col], 
                           c=data[z_col], cmap='viridis', s=20, alpha=0.6)
        
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_zlabel(z_col)
        ax.set_title('3D Data Visualization')
        fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
        
    else:
        # Create 2D plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(data[x_col], data[y_col])
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title('2D Data Visualization')
        ax.grid(True)
    
    fig.tight_layout()
    
    result = ProcessingResult(
        success=True,
        data=processed_data,
        figure=fig,
        processing_script=self.name
    )
    
    result.metadata.update({
        'processor': 'magnetic',
        'plot_type': plot_type,
        'visualization_columns': [x_col, y_col] + ([z_col] if plot_type == '3D' else [])
    })
    
    return result
```

#### **Making Plots Optional**

For performance or user preference, make plot generation optional:

```python
def get_parameters(self):
    return {
        'output_options': {
            'generate_plot': {
                'value': True,
                'type': 'bool',
                'description': 'Generate an interactive plot for the data viewer'
            },
            'plot_type': {
                'value': '2D',
                'type': 'choice',
                'choices': ['2D', '3D'],
                'description': 'Type of plot to generate'
            }
        }
    }

def execute(self, data, params, progress_callback=None, input_file_path=None):
    # ... data processing ...
    
    result = ProcessingResult(
        success=True,
        data=processed_data,
        processing_script=self.name
    )
    
    # Only generate plot if requested
    generate_plot = params.get('output_options', {}).get('generate_plot', {}).get('value', True)
    
    if generate_plot:
        if progress_callback:
            progress_callback(0.9, "Generating visualization...")
        
        fig, ax = plt.subplots()
        # ... create your plot ...
        
        result.figure = fig
    
    return result
```

#### **Plot File Organization**

Generated plot files are automatically saved to:
```
project_directory/
├── processed/
│   ├── magnetic/
│   │   ├── survey_data_basic_processing_20250714_125841.mplplot
│   │   ├── survey_data_basic_processing_20250714_125841.png
│   │   └── survey_data_grid_interpolator_20250714_130215.mplplot
│   ├── gpr/
│   └── gamma/
```

#### **Accessing Saved Plots**

Users can access saved plots in multiple ways:

1. **Lab Widget**: Browse and double-click `.mplplot` files in the processed directory
2. **Project Explorer**: Navigate to saved plots in the file tree
3. **Data Viewer**: Plots automatically open in new tabs when generated
4. **File System**: Access static `.png` files for reports and documentation

#### **Important Notes**

- **No user interaction required**: Plots are saved automatically without prompts
- **Unique filenames**: Timestamps prevent overwrites when running scripts multiple times
- **Interactive preservation**: `.mplplot` files maintain full matplotlib interactivity
- **Performance**: Plot generation is optional and can be toggled off for large datasets
- **Memory management**: Plots are saved immediately and don't consume memory long-term

## 4. Script Metadata System

All processing scripts must implement the `get_metadata()` method to provide rich information for users. This system enhances the user experience by providing detailed script information, tooltips, and intelligent script ordering.

### 4.1 ScriptMetadata Structure

The `ScriptMetadata` dataclass provides structured information about your script:

```python
from src.processing.base import ScriptMetadata

def get_metadata(self) -> ScriptMetadata:
    return ScriptMetadata(
        description="Brief description of what this script does",
        flags=["tag1", "tag2", "tag3"],
        typical_use_case="When and why to use this script",
        field_compatible=True,  # Can be run in field conditions
        estimated_runtime="< 30 seconds"  # User-friendly time estimate
    )
```

### 4.2 Available Flags

Use appropriate flags to help users find the right scripts:

- **`visualization`** - Produces plots, maps, or visual outputs
- **`field-use`** - Optimized for field conditions (fast, simple interface)
- **`preprocessing`** - Data cleaning, corrections, initial processing
- **`analysis`** - Core analytical processing and calculations
- **`export`** - Data export, formatting, or conversion
- **`qc`** - Quality control, validation, or diagnostic tools
- **`advanced`** - Requires expertise or complex configuration
- **`batch`** - Suitable for processing multiple files

### 4.3 Best Practices

1. **Descriptions**: Write clear, concise descriptions focusing on what the script does
2. **Use Cases**: Explain when and why someone would use this script
3. **Runtime Estimates**: Provide realistic time estimates for typical data sizes
4. **Field Compatibility**: Mark scripts as field-compatible only if they're fast and simple
5. **Advanced Flag**: Use the "advanced" flag for scripts requiring expertise or complex configuration

### 4.4 Example Implementations

#### Basic Processing Script
```python
def get_metadata(self) -> ScriptMetadata:
    return ScriptMetadata(
        description="Basic magnetic data processing with column operations and statistics",
        flags=["preprocessing", "field-use"],
        typical_use_case="Simple data processing and initial exploration",
        field_compatible=True,
        estimated_runtime="< 30 seconds"
    )
```

#### Advanced Analysis Script
```python
def get_metadata(self) -> ScriptMetadata:
    return ScriptMetadata(
        description="Detects and highlights magnetic anomalies using statistical analysis",
        flags=["analysis", "visualization", "advanced"],
        typical_use_case="Identifying potential targets after preprocessing and corrections",
        field_compatible=False,
        estimated_runtime="2-5 minutes"
    )
```

#### Field-Compatible Visualization
```python
def get_metadata(self) -> ScriptMetadata:
    return ScriptMetadata(
        description="Creates visual flight path plots and interactive maps for survey data",
        flags=["visualization", "field-use", "qc"],
        typical_use_case="Visualizing drone flight paths and data coverage for quality control",
        field_compatible=True,
        estimated_runtime="< 30 seconds"
    )
```

### 4.5 How Metadata Enhances User Experience

The metadata system provides several user experience improvements:

- **Rich Tooltips**: Users see detailed information when hovering over scripts in the dropdown
- **Intelligent Ordering**: Recently used scripts appear first, followed by alphabetical ordering
- **Smart Defaults**: The most recently used script is automatically selected
- **Quick Re-run**: Context menu allows instant re-running with saved parameters
- **Field Workflow**: Field-compatible scripts are clearly identified for field use

## 5. Enhanced Layer Generation System

The UXO-Wizard now includes a powerful modular layer generation system that automatically converts your processing outputs into interactive map layers. This system provides universal layer creation methods that all processors inherit from the `BaseProcessor` class.

### 5.1 Automatic Layer Creation

When your script generates `LayerOutput` objects using `result.add_layer_output()`, the system automatically:

1. **Converts to UXOLayer objects**: LayerOutput → ProcessingWorker → UXOLayer
2. **Applies intelligent styling**: Based on processor type and data characteristics
3. **Detects coordinate systems**: Automatic Norwegian UTM zone detection (32-35)
4. **Groups layers logically**: By processor type and data type in LayerManager
5. **Displays in real-time**: Layers appear immediately in the map interface
6. **Uses custom names**: Specify `layer_name` in metadata for custom layer names

### 5.2 Supported Layer Types

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

### 5.3 Inherited Layer Methods

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

### 5.4 Styling Guidelines by Processor Type

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

### 5.5 Norwegian UTM Coordinate Support

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

### 5.6 Layer Grouping and Organization

Layers are automatically organized in the LayerManager based on:

1. **Processor type**: From `metadata['processor_type']`
2. **Processing history**: From layer processing lineage
3. **Data type**: From `metadata['data_type']`

Example layer groups:
- "Magnetic Processing" (for magnetic processor outputs)
- "Survey Data" (for raw data visualizations)
- "Annotations" (for user-added layers)

### 5.7 Performance Considerations

- **Point limits**: Keep point layers under 10,000 points for performance
- **Clustering**: Enable clustering for dense point data
- **Subsampling**: Create overview layers for large datasets
- **Memory**: Large raster layers should be optimized or tiled

## 5. Universal Minimum Curvature Interpolation System

The UXO Wizard framework now includes a powerful minimum curvature interpolation system built into the `BaseProcessor` class. This system provides all processors with access to professional-grade interpolation methods that eliminate flight line artifacts and provide superior data quality.

### 5.1 Overview

The minimum curvature interpolation system was developed by extracting the best features from both the gamma and magnetic interpolators, creating a unified implementation that all processors can inherit and use.

#### **Key Features**
- **Soft Constraints**: Eliminates flight line artifacts using influence zones
- **Hard Constraints**: Traditional minimum curvature with exact data point fitting
- **JIT Compilation**: Numba-optimized functions for maximum performance
- **Boundary Masking**: Prevents extrapolation outside data coverage
- **Universal Access**: Available to all processors through inheritance

#### **Implementation Benefits**
- **Consistency**: All processors use the same high-quality interpolation
- **Performance**: JIT compilation provides significant speed improvements
- **Maintainability**: Single codebase to improve and debug
- **Flexibility**: Both constraint modes available based on use case

### 5.2 Using Minimum Curvature Interpolation

#### **Basic Usage**
Any script can access the minimum curvature interpolation methods through the BaseProcessor:

```python
from src.processing.base import BaseProcessor

class MyInterpolationScript(ScriptInterface):
    def __init__(self):
        # Create helper to access BaseProcessor methods
        self._base_processor = _InterpolationHelper()
    
    def execute(self, data, params, progress_callback=None, input_file_path=None):
        # Extract coordinates and values
        x = data['longitude'].values
        y = data['latitude'].values
        z = data['field_value'].values
        
        # Perform minimum curvature interpolation
        grid_x, grid_y, grid_z = self._base_processor.minimum_curvature_interpolation(
            x, y, z,
            grid_resolution=150,
            constraint_mode='soft',  # or 'hard'
            max_iterations=1000,
            tolerance=1e-6,
            progress_callback=progress_callback
        )
        
        # Apply boundary masking
        mask = self._base_processor.create_boundary_mask(x, y, grid_x, grid_y)
        grid_z = np.where(mask, grid_z, np.nan)
        
        return ProcessingResult(success=True, data=processed_data)

# Helper class for accessing BaseProcessor methods
class _InterpolationHelper(BaseProcessor):
    def validate_data(self, data):
        return True
```

#### **Advanced Usage with Parameters**
```python
def get_parameters(self):
    return {
        'interpolation_parameters': {
            'grid_resolution': {
                'value': 150,
                'type': 'int',
                'min': 20,
                'max': 500,
                'description': 'Grid resolution (points per axis)'
            },
            'constraint_mode': {
                'value': 'soft',
                'type': 'choice',
                'choices': ['soft', 'hard'],
                'description': 'Constraint mode: soft (eliminates flight line artifacts) or hard (traditional)'
            },
            'max_iterations': {
                'value': 1000,
                'type': 'int',
                'description': 'Maximum iterations for convergence'
            },
            'tolerance': {
                'value': 1e-6,
                'type': 'float',
                'description': 'Convergence tolerance'
            }
        }
    }
```

### 5.3 Constraint Modes

#### **Soft Constraints (Recommended)**
- **Purpose**: Eliminates flight line artifacts in survey data
- **Method**: Uses influence zones around data points with weighted blending
- **Benefits**: Smoother results, better for flight line data, reduces artifacts
- **Use Case**: Gamma radiation data, magnetic surveys with flight lines

```python
grid_x, grid_y, grid_z = self._base_processor.minimum_curvature_interpolation(
    x, y, z,
    constraint_mode='soft',
    influence_radius=None,  # Auto-calculated based on data density
    progress_callback=progress_callback
)
```

#### **Hard Constraints (Traditional)**
- **Purpose**: Exact fitting of data points
- **Method**: Fixes data points exactly during interpolation
- **Benefits**: Preserves original data values precisely
- **Use Case**: Precise measurement data, calibration points

```python
grid_x, grid_y, grid_z = self._base_processor.minimum_curvature_interpolation(
    x, y, z,
    constraint_mode='hard',
    progress_callback=progress_callback
)
```

### 5.4 Boundary Masking

The system includes comprehensive boundary masking to prevent unreliable extrapolation:

```python
# Create boundary mask
mask = self._base_processor.create_boundary_mask(
    x, y, grid_x, grid_y, 
    method='convex_hull'  # or 'alpha_shape'
)

# Apply mask to interpolated data
grid_z = np.where(mask, grid_z, np.nan)
```

#### **Boundary Methods**
- **convex_hull**: Creates convex hull around data points
- **alpha_shape**: Distance-based boundary detection

### 5.5 Performance Optimization

The interpolation system includes several performance optimizations:

#### **JIT Compilation**
- **Numba Support**: Automatic JIT compilation for core functions
- **Fallback Support**: Python fallback when Numba unavailable
- **Performance Gain**: 10-100x speedup on large datasets

#### **Progress Reporting**
```python
def progress_callback(percentage, message):
    print(f"{percentage:.1f}%: {message}")

grid_x, grid_y, grid_z = self._base_processor.minimum_curvature_interpolation(
    x, y, z,
    progress_callback=progress_callback
)
```

### 5.6 Integration Examples

#### **Grid Interpolator Integration**
The magnetic grid interpolator has been updated to use the new system:

```python
# Now supports both soft and hard constraints
def execute(self, data, params, progress_callback=None, input_file_path=None):
    # Extract constraint mode from parameters
    constraint_mode = params.get('interpolation_parameters', {}).get('constraint_mode', {}).get('value', 'soft')
    
    # Use enhanced interpolation method
    grid_X, grid_Y, grid_z = self.minimum_curvature_interpolation_enhanced(
        x, y, z, 
        grid_resolution=grid_resolution,
        constraint_mode=constraint_mode,
        progress_callback=progress_callback
    )
```

#### **Gamma Interpolator Compatibility**
The gamma interpolator continues to use its optimized soft constraint implementation while benefiting from the shared codebase.

### 5.7 Best Practices for Interpolation

#### **Choose the Right Constraint Mode**
- **Use soft constraints** for flight line data to eliminate artifacts
- **Use hard constraints** for precise measurement data
- **Default to soft** for most geophysical survey data

#### **Optimize Grid Resolution**
- **Start with 150-300** points per axis for most datasets
- **Increase for detailed analysis** (up to 500)
- **Decrease for performance** (down to 50)

#### **Monitor Convergence**
- **Use progress callbacks** to monitor iteration progress
- **Set reasonable tolerance** (1e-6 typical)
- **Limit iterations** to prevent infinite loops (1000 typical)

## 6. Enhanced File and Layer Output System

The UXO Wizard framework provides a comprehensive output system for both file generation and map layer creation. This system automates file management, layer generation, and provides rich metadata for all outputs.

### 6.1 File Output System

#### **Automatic File Management**
The framework automatically handles file creation, naming, and organization:

```python
def execute(self, data, params, progress_callback=None, input_file_path=None):
    result = ProcessingResult(success=True, data=processed_data)
    
    # Files are automatically saved to project/processed/processor_type/
    result.add_output_file(
        file_path=str(diagnostic_plot_path),
        file_type='png',
        description='Interpolation diagnostic plots'
    )
    
    result.add_output_file(
        file_path=str(grid_csv_path),
        file_type='csv',
        description='Interpolated grid data'
    )
    
    return result
```

#### **File Organization Structure**
```
project_directory/
├── processed/
│   ├── magnetic/
│   │   ├── survey_data_grid_interpolator_20250716_143052.png
│   │   ├── survey_data_grid_interpolator_20250716_143052.csv
│   │   └── survey_data_grid_interpolator_20250716_143052.mplplot
│   ├── gamma/
│   │   ├── radiation_data_gamma_interpolator_20250716_143122.png
│   │   └── radiation_data_gamma_interpolator_20250716_143122.csv
│   └── gpr/
```

#### **Supported File Types**
- **PNG**: High-resolution plots (300 DPI)
- **CSV**: Processed data tables
- **MPLPLOT**: Interactive matplotlib plots
- **JSON**: Metadata and analysis results
- **GEOTIFF**: Georeferenced raster data
- **HTML**: Interactive reports

### 6.2 Layer Output System

#### **Automatic Layer Generation**
The framework converts processing outputs into map layers automatically:

```python
def execute(self, data, params, progress_callback=None, input_file_path=None):
    result = ProcessingResult(success=True, data=processed_data)
    
    # Create unique layer name with timestamp
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    layer_name = f'{input_filename} - Interpolated Field ({timestamp})'
    
    # Add raster layer for interpolated data
    result.add_layer_output(
        layer_type='raster',
        data={
            'grid': grid_z,
            'bounds': [min_x, min_y, max_x, max_y],
            'field_name': field_column
        },
        style_info={
            'use_graduated_colors': True,
            'opacity': 0.7,
            'color_ramp': ["#000080", "#0000FF", "#00FFFF", "#00FF00", "#FFFF00", "#FF8000", "#FF0000"]
        },
        metadata={
            'layer_name': layer_name,
            'description': 'Interpolated magnetic field data',
            'data_type': 'interpolated_grid',
            'grid_shape': grid_z.shape,
            'processing_method': 'minimum_curvature',
            'constraint_mode': constraint_mode
        }
    )
    
    # Add point layer for original data
    result.add_layer_output(
        layer_type='points',
        data=original_data,
        style_info={
            'color_field': field_column,
            'use_graduated_colors': True,
            'size': 4,
            'opacity': 0.8
        },
        metadata={
            'layer_name': f'{input_filename} - Original Data ({timestamp})',
            'description': 'Original survey data points',
            'data_type': 'survey_data',
            'total_points': len(original_data)
        }
    )
    
    return result
```

#### **Layer Types and Usage**

**Raster Layers** - For interpolated grids:
```python
result.add_layer_output(
    layer_type='raster',
    data={
        'grid': numpy_array,           # 2D numpy array
        'bounds': [x_min, y_min, x_max, y_max],  # Spatial bounds
        'field_name': 'field_column'   # Source field name
    },
    style_info={
        'use_graduated_colors': True,
        'opacity': 0.7,
        'color_ramp': custom_colors
    },
    metadata={
        'layer_name': unique_name,
        'description': 'Layer description',
        'data_type': 'interpolated_grid',
        'grid_shape': (ny, nx),
        'processing_method': 'minimum_curvature'
    }
)
```

**Point Layers** - For coordinate data:
```python
result.add_layer_output(
    layer_type='points',
    data=dataframe_with_coords,  # DataFrame with lat/lon columns
    style_info={
        'color_field': 'value_column',
        'use_graduated_colors': True,
        'size': 4,
        'opacity': 0.8,
        'enable_clustering': True
    },
    metadata={
        'layer_name': unique_name,
        'description': 'Survey data points',
        'data_type': 'survey_data',
        'total_points': len(dataframe_with_coords),
        'coordinate_columns': {'latitude': 'lat', 'longitude': 'lon'}
    }
)
```

**Vector Layers** - For lines and paths:
```python
result.add_layer_output(
    layer_type='flight_lines',
    data=flight_path_dataframe,
    style_info={
        'line_color': '#FF6600',
        'line_width': 2,
        'line_opacity': 0.9
    },
    metadata={
        'layer_name': unique_name,
        'description': 'Survey flight path',
        'data_type': 'flight_path'
    }
)
```

### 6.3 Processor-Specific Styling

#### **Magnetic Data Styling**
```python
magnetic_style = {
    'color_ramp': ["#000080", "#0000FF", "#00FFFF", "#00FF00", "#FFFF00", "#FF8000", "#FF0000"],
    'use_graduated_colors': True,
    'opacity': 0.8
}
```

#### **Gamma Radiation Styling**
```python
gamma_style = {
    'color_ramp': ["#004400", "#008800", "#00CC00", "#CCCC00", "#CC8800", "#CC0000"],
    'use_graduated_colors': True,
    'opacity': 0.9
}
```

#### **GPR Data Styling**
```python
gpr_style = {
    'color_ramp': ["#000066", "#0066CC", "#66CCFF", "#CCCCCC", "#FFCC66", "#CC6600"],
    'use_graduated_colors': True,
    'opacity': 0.8
}
```

### 6.4 Metadata and Documentation

#### **Rich Metadata System**
```python
result.metadata.update({
    'processor': 'magnetic',
    'processing_method': 'minimum_curvature',
    'constraint_mode': constraint_mode,
    'grid_resolution': grid_resolution,
    'field_processed': field_column,
    'data_points': len(original_data),
    'valid_points': len(valid_data),
    'boundary_masking': enable_masking,
    'numba_acceleration': NUMBA_AVAILABLE,
    'processing_time': processing_time,
    'convergence_iterations': actual_iterations,
    'coordinate_system': detected_crs
})
```

#### **Layer Metadata**
```python
layer_metadata = {
    'layer_name': unique_layer_name,
    'description': detailed_description,
    'data_type': classification,
    'processing_history': [processor_type, script_name],
    'creation_timestamp': timestamp,
    'source_file': input_file_path,
    'processing_parameters': relevant_params
}
```

### 6.5 Integration with UXO Wizard

#### **Automatic Processing**
- **File Auto-saving**: All outputs automatically saved to project structure
- **Layer Auto-creation**: Layers appear immediately in map interface
- **Metadata Generation**: JSON sidecar files created automatically
- **Progress Tracking**: Real-time updates during processing

#### **User Experience**
- **No Manual File Management**: Everything handled automatically
- **Unique Naming**: Timestamps prevent overwrites
- **Immediate Visualization**: Layers appear in map as soon as processing completes
- **Interactive Plots**: Saved plots maintain full matplotlib functionality

## 7. Best Practices and Recommendations

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

## 8. Smart Output Management System

The UXO Wizard framework includes an intelligent output management system that prevents redundant file creation and optimizes the processing pipeline for different types of scripts.

### 8.1 Pipeline Output Behavior

#### **Default Pipeline Behavior**
By default, the processing pipeline automatically:
1. **Creates CSV/Excel files** from the `result.data` DataFrame
2. **Generates JSON metadata files** with processing information
3. **Saves all outputs** to `processed/{processor_type}/` directory

#### **Smart Output Detection**
The pipeline now intelligently detects when scripts handle their own file output and avoids creating redundant files. This prevents issues like:
- Duplicate CSV files in wrong locations
- Unnecessary JSON sidecar files when metadata is integrated
- File organization conflicts between pipeline and script outputs

### 8.2 Scripts That Handle Their Own Output

Some scripts create their own comprehensive output files and directories. The framework automatically detects these scripts and skips redundant pipeline file generation.

#### **Current Self-Managing Scripts**
- **`flight_path_segmenter`**: Creates its own CSV files in dedicated subdirectories
- **Advanced processing scripts**: Scripts that generate multiple related output files

#### **How Detection Works**
```python
# In ProcessingPipeline._script_handles_own_output()
def _script_handles_own_output(self, result: ProcessingResult) -> bool:
    """Check if the processing script handles its own output files"""
    scripts_with_own_output = [
        'flight_path_segmenter',
        # Add other scripts here that handle their own output
    ]
    
    script_name = result.processing_script or ''
    return script_name in scripts_with_own_output
```

### 8.3 Creating Self-Managing Scripts

If your script creates comprehensive output files, you can design it to be self-managing:

#### **Script Design Pattern**
```python
class ComprehensiveProcessor(ScriptInterface):
    @property
    def name(self) -> str:
        return "comprehensive_processor"  # This name determines self-management
    
    def execute(self, data, params, progress_callback=None, input_file_path=None):
        # Create your own output directory
        output_dir = self._create_output_directory(input_file_path)
        
        # Generate multiple related files
        self._save_processed_data(data, output_dir)
        self._save_analysis_report(analysis_results, output_dir)
        self._save_visualizations(figures, output_dir)
        
        # Create result WITHOUT setting result.data to avoid pipeline CSV creation
        result = ProcessingResult(
            success=True,
            data=None,  # Set to None to prevent pipeline CSV generation
            processing_script=self.name,
            metadata={
                'processor': 'magnetic',
                'output_directory': str(output_dir),
                'files_created': len(output_files),
                # Include all metadata directly - no separate JSON needed
                'processing_info': {
                    'timestamp': datetime.now().isoformat(),
                    'processing_time': processing_time
                }
            }
        )
        
        # Register output files for tracking
        for file_path in output_files:
            result.add_output_file(
                file_path=str(file_path),
                file_type=file_path.suffix[1:],
                description=f"Processed output: {file_path.name}"
            )
        
        return result
```

#### **Directory Structure for Self-Managing Scripts**
```
project_directory/
├── processed/
│   ├── magnetic/
│   │   ├── dataset1_comprehensive/        # Self-managed directory
│   │   │   ├── data_primary.csv
│   │   │   ├── data_secondary.csv
│   │   │   ├── analysis_report.txt
│   │   │   └── visualizations.png
│   │   ├── dataset2_basic_processing.csv  # Pipeline-managed file
│   │   └── dataset2_basic_processing.png  # Pipeline-managed file
```

### 8.4 Integration Guidelines

#### **For Self-Managing Scripts**
1. **Set `result.data = None`** to prevent pipeline CSV creation
2. **Include comprehensive metadata** in `result.metadata` instead of relying on separate JSON files
3. **Register all output files** using `result.add_output_file()`
4. **Create organized directory structures** for related outputs
5. **Add script name to the self-managing list** in the pipeline

#### **For Standard Scripts**
1. **Return processed data** in `result.data` for automatic CSV generation
2. **Let the pipeline handle** file naming and directory creation
3. **Focus on processing logic** rather than file management
4. **Use the integrated metadata system** for processing information

### 8.5 Metadata Integration

The framework now integrates processing metadata directly into the `ProcessingResult.metadata` instead of creating separate JSON files.

#### **Automatic Metadata Integration**
```python
# The pipeline automatically adds this metadata to your result:
result.metadata.update({
    'processing_info': {
        'processor_type': 'magnetic',
        'processing_script': 'your_script_name',
        'processing_time': 2.34,
        'timestamp': '2025-01-15T14:30:22',
        'success': True
    },
    'file_info': {
        'input_file': '/path/to/input.csv',
        'output_file': '/path/to/output.csv',
        'export_format': 'csv',
        'data_shape': [1000, 15]
    }
})
```

#### **Accessing Integrated Metadata**
Your script can access and extend this integrated metadata:
```python
def execute(self, data, params, progress_callback=None, input_file_path=None):
    result = ProcessingResult(success=True, data=processed_data)
    
    # The pipeline will automatically add processing_info and file_info
    # You can add your own metadata alongside:
    result.metadata.update({
        'processor': 'magnetic',
        'algorithm': 'minimum_curvature',
        'parameters_used': params,
        'quality_metrics': {
            'data_coverage': 0.95,
            'interpolation_error': 2.1
        }
    })
    
    return result
```

### 8.6 Best Practices for Output Management

#### **Choose the Right Approach**
- **Use pipeline management** for simple scripts that produce a single processed dataset
- **Use self-management** for complex scripts that create multiple related files or need specific directory structures

#### **Avoid Common Issues**
- **Don't create redundant files**: Let either the pipeline OR your script handle file creation, not both
- **Don't create separate JSON files**: Use the integrated metadata system instead
- **Don't hardcode output paths**: Use the project's `processed/` directory structure

#### **Performance Considerations**
- **Self-managing scripts** reduce pipeline overhead by avoiding redundant file operations
- **Integrated metadata** eliminates the need for separate JSON file I/O
- **Smart detection** prevents unnecessary processing steps

By following these guidelines, your scripts will integrate seamlessly with the UXO Wizard framework's intelligent output management system, providing better performance and cleaner file organization.

## 9. Script Output Control Flag

The UXO Wizard framework includes a flag-based system to control whether scripts handle their own output files or rely on the processing pipeline for automatic file generation.

### 9.1 The `handles_own_output` Property

Scripts can now declare whether they manage their own output files by implementing the `handles_own_output` property:

```python
class MyScript(ScriptInterface):
    @property
    def handles_own_output(self) -> bool:
        """Whether this script creates its own output files and doesn't need pipeline CSV generation
        
        Returns:
            True if script handles its own output files (default: False)
        """
        return True  # or False
```

### 9.2 How the Flag Works

#### **Default Behavior (handles_own_output = False)**
When a script returns `False` (the default), the processing pipeline will:
1. **Automatically create CSV/Excel files** from `result.data`
2. **Generate JSON metadata files** with processing information
3. **Save outputs** to `processed/{processor_type}/` with standard naming

#### **Self-Managing Behavior (handles_own_output = True)**
When a script returns `True`, the processing pipeline will:
1. **Skip automatic file generation** to prevent duplicates
2. **Trust the script** to create its own output files
3. **Only handle layer creation and UI updates**

### 9.3 Implementation Examples

#### **Standard Script (Pipeline-Managed)**
```python
class BasicProcessor(ScriptInterface):
    @property
    def handles_own_output(self) -> bool:
        return False  # Let pipeline handle file creation
    
    def execute(self, data, params, progress_callback=None, input_file_path=None):
        # Process data
        processed_data = data.copy()
        processed_data['new_column'] = processed_data['old_column'] * 2
        
        # Return data - pipeline will create CSV automatically
        return ProcessingResult(
            success=True,
            data=processed_data,  # Pipeline creates CSV from this
            processing_script=self.name,
            metadata={'processor': 'magnetic'}
        )
```

#### **Self-Managing Script (Script-Managed)**
```python
class FlightPathSegmenter(ScriptInterface):
    @property
    def handles_own_output(self) -> bool:
        return True  # Script creates its own files
    
    def execute(self, data, params, progress_callback=None, input_file_path=None):
        # Create output directory
        output_dir = self._create_output_directory(input_file_path)
        
        # Process and segment data
        segments = self._segment_flight_paths(data)
        
        # Save multiple CSV files in organized subdirectory
        saved_files = []
        for segment_name, segment_data in segments.items():
            file_path = output_dir / f"{segment_name}.csv"
            segment_data.to_csv(file_path, index=False)
            saved_files.append(file_path)
        
        # Return result with file references - no automatic CSV creation
        result = ProcessingResult(
            success=True,
            data=data,  # Original data for layer creation
            processing_script=self.name,
            metadata={'processor': 'magnetic'}
        )
        
        # Register the files we created
        for file_path in saved_files:
            result.add_output_file(
                file_path=str(file_path),
                file_type="csv",
                description=f"Segmented flight data: {file_path.stem}"
            )
        
        return result
```

### 9.4 Pipeline Detection Logic

The processing pipeline automatically detects the flag and adjusts behavior:

```python
# In ProcessingPipeline class
def _script_handles_own_output(self) -> bool:
    """Check if the processing script handles its own output files"""
    if hasattr(self.current_script, 'handles_own_output'):
        return self.current_script.handles_own_output
    return False  # Default to pipeline-managed

# Used in processing logic
if result.data is not None and not self._script_handles_own_output():
    # Only create automatic CSV if script doesn't handle its own output
    output_path = self._generate_output_file(result)
    result.output_file_path = output_path
```

### 9.5 When to Use Each Approach

#### **Use `handles_own_output = False` (Default) When:**
- Script produces a single processed dataset
- Standard CSV/Excel output is sufficient
- Simple file naming is acceptable
- You want automatic file management

#### **Use `handles_own_output = True` When:**
- Script creates multiple related output files
- Custom directory structure is needed
- Specialized file formats are required
- Script needs complete control over file organization

### 9.6 Migration from Legacy System

The flag-based system replaces the previous hardcoded script name list approach:

#### **Old System (Deprecated)**
```python
# Old approach - hardcoded script names
def _script_handles_own_output(self, result: ProcessingResult) -> bool:
    scripts_with_own_output = [
        'flight_path_segmenter',
        'Flight Path Segmenter',  # Had to include multiple variations
        # Maintenance nightmare as new scripts were added
    ]
    script_name = result.processing_script or ''
    return script_name in scripts_with_own_output
```

#### **New System (Current)**
```python
# New approach - script declares its own behavior
def _script_handles_own_output(self) -> bool:
    if hasattr(self.current_script, 'handles_own_output'):
        return self.current_script.handles_own_output
    return False
```

### 9.7 Benefits of the Flag System

#### **For Developers**
- **Self-Documenting**: Scripts declare their own behavior
- **No Central Maintenance**: No need to update pipeline lists
- **Type Safety**: Property-based approach with clear return type
- **Flexibility**: Each script controls its own output strategy

#### **For Framework**
- **Cleaner Code**: Eliminates hardcoded script name lists
- **Better Maintainability**: Changes isolated to individual scripts
- **Consistent Behavior**: Clear contract between scripts and pipeline
- **Future-Proof**: Easy to extend with additional flags

### 9.8 Best Practices

#### **Always Override the Property**
Even if using the default behavior, explicitly declare it for clarity:
```python
@property
def handles_own_output(self) -> bool:
    """This script uses pipeline-managed output"""
    return False
```

#### **Document Your Choice**
Include a comment explaining why the script handles or doesn't handle its own output:
```python
@property
def handles_own_output(self) -> bool:
    """This script creates segmented CSV files in organized subdirectories"""
    return True
```

#### **Consider User Experience**
Scripts that handle their own output should create well-organized, clearly named files that users can easily understand and navigate.

This flag-based system provides a clean, maintainable way for scripts to declare their output management preferences, eliminating the need for manual pipeline configuration and reducing maintenance overhead.