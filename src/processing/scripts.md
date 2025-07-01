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

    * **Adding Visualization Layers:** To integrate with a future mapping system, you can add layer data.

        ```python
        result.add_layer_output(
            layer_type="survey_track",
            data=df[['UTM_Easting', 'UTM_Northing']],
            style_info={'color': 'blue', 'weight': 2},
            metadata={'description': 'Survey flight path'}
        )
        ```

## 4. Best Practices and Recommendations

* **Immutability:** Treat the input `data` DataFrame as immutable. Create a copy before you start modifying it: `df = data.copy()`.
* **Robust Parameter Access:** When accessing parameters, use `.get()` to avoid `KeyError` if the structure changes. As seen in `magbase_processing.py`: `process_opts = params.get('processing_options', {})`.
* **Logging:** Use the `loguru` logger for detailed debugging. `from loguru import logger` is available. `logger.debug("Starting step 1")`.
* **Progress Reporting:** Provide meaningful and frequent progress updates via the `progress_callback`. This is essential for long-running processes to keep the user informed.
* **Error Handling:** Be specific in your error messages. Instead of "An error occurred," use "Failed to find required 'latitude' column."
* **Code Organization:** For complex scripts like `magbase_processing`, break down the logic into smaller, private helper methods (e.g., `_interpolate_gps_gaps`, `_calculate_residual_anomalies`) to keep the `execute` method clean and readable.
* **Performance:** For computationally intensive tasks, consider using libraries like `NumPy` for vectorized operations. For tasks that are highly parallelizable, you can use Python's `multiprocessing` library, as demonstrated in the `_calculate_residual_anomalies` function of `magbase_processing.py`.

By adhering to the `ScriptInterface` contract and following these guidelines, you can build powerful, reusable, and well-integrated processing scripts for the UXO Wizard application.
