# UXO Wizard

UXO Wizard is a powerful, data-centric desktop application designed for the analysis and visualization of geophysical data for Unexploded Ordnance (UXO) detection. It provides a comprehensive suite of tools for managing, processing, and interpreting various types of sensor data in a unified project-based environment.

## Key Features

- **Integrated Project Management:** Organize your work in projects. The `.uxo` project file bundles all your data layers, map settings, and processing history into a single, portable file.
- **Advanced Data Viewer:**
    - Handle large datasets with ease thanks to optimized chunked loading for large CSV files.
    - View and analyze various file formats, including CSV, Excel, JSON, and images.
    - Filter, sort, and search your data with a powerful and intuitive interface.
- **Interactive Geospatial Analysis:**
    - Visualize your data on an interactive map with multiple base layers, including topographic and satellite imagery.
    - Overlay various data layers, such as sensor readings, flight paths, and processing results.
    - Dynamically style and interact with map layers.
- **Flexible Processing Pipeline:**
    - Process your data using a variety of built-in algorithms for different sensor types (magnetic, GPR, gamma, etc.).
    - The processing pipeline is designed to be extensible, allowing for the addition of new custom processing scripts.
- **Rich User Interface:**
    - A modern and intuitive user interface built with Qt for Python (PySide6).
    - Dockable windows and a flexible layout allow you to customize your workspace.
    - Light and dark themes to suit your preferences.

## File Structure

```
/Users/aleksandergarbuz/Documents/UXO-Wizard/
├───env.yaml
├───main.py
├───src/
│   ├───__init__.py
│   ├───processing/
│   │   ├───__init__.py
│   │   ├───base.py
│   │   ├───gamma.py
│   │   ├───gpr.py
│   │   ├───magnetic.py
│   │   ├───multispectral.py
│   │   ├───pipeline.py
│   │   └───scripts/
│   │       ├───gamma/
│   │       ├───gpr/
│   │       └───magnetic/
│   ├───project/
│   │   ├───__init__.py
│   │   ├───project_history.py
│   │   ├───project_manager.py
│   │   ├───project_schema.py
│   │   └───project_validator.py
│   └───ui/
│       ├───__init__.py
│       ├───main_window.py
│       ├───themes.py
│       ├───assets/
│       ├───dialogs/
│       ├───map/
│       └───widgets/
```

## Core Concepts

### Projects

A project in UXO Wizard is a self-contained environment for your analysis work. When you save a project, all the data layers you have loaded, the current state of your map, and the history of your processing steps are all saved into a single `.uxo` file. This makes it easy to share your work with colleagues or to resume your work later.

### Layers

Layers are the fundamental building blocks of your analysis. A layer can be a raw data file, a processed dataset, or a set of annotations. Layers are displayed in the layer panel, where you can control their visibility, opacity, and stacking order on the map.

### Processing

The processing pipeline is where you can apply various algorithms to your data. Each processor is designed for a specific type of data (e.g., magnetic, GPR) and contains a collection of scripts that perform different tasks, such as anomaly detection, filtering, or interpolation.

## Getting Started

1.  **Open a Project:** Launch the application and open a project folder to begin.
2.  **Load Data:** Use the project explorer to browse and open your data files.
3.  **Visualize:** View your tabular data in the data viewer and your geospatial data on the map.
4.  **Process:** Use the processing tools to analyze your data and generate results.
5.  **Save Your Work:** Save your entire session, including all data layers and map settings, into a `.uxo` project file.

## Usage

### Project Explorer

The Project Explorer is your main tool for navigating your project files. You can open files by double-clicking on them, and you can perform common file operations like creating new folders, renaming files, and deleting files by right-clicking on them.

### Data Viewer

The Data Viewer is where you can view and analyze your tabular data. You can sort the data by clicking on the column headers, and you can filter the data by using the column filter dropdowns. You can also search for specific values in the data using the search bar.

### Map

The Map is where you can visualize your geospatial data. You can zoom in and out, pan around the map, and switch between different base layers. You can also control the visibility and opacity of your data layers in the layer panel.

### Processing

To process your data, you can use the processing tools in the "Processing" menu. Each processor has a set of scripts that you can run on your data. When you run a script, you will be prompted to enter any required parameters. The results of the processing will be added as new layers to your project.
