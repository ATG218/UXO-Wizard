# SGTool Python

Python adaptation of SGTool for geophysical processing, optimized for magnetic survey data analysis.

## Overview

SGTool-Python is a standalone geophysical processing toolkit that adapts the proven algorithms from the QGIS SGTool plugin for command-line and programmatic use. It specializes in processing magnetic survey data from CSV files generated by magbase and flight_path_segmenter workflows.

## Features

### Core Geophysical Filters
- **Frequency Domain**: RTP, RTE, upward/downward continuation, vertical integration
- **Spatial Filters**: High-pass, low-pass, band-pass with cosine rolloff
- **Gradient Operations**: Total Horizontal Gradient, Analytic Signal, Tilt Angle
- **Convolution Filters**: Gaussian, directional, spatial statistics
- **Multivariate Analysis**: PCA/ICA for multi-band grids

### Data Processing Pipeline
- **Batch CSV Processing**: Handle directories of magnetic survey data
- **Grid Interpolation**: Vectorized kriging and minimum curvature
- **Multi-scale Analysis**: Anomaly detection across multiple scales
- **Interactive Visualization**: Folium-based maps with toggleable layers

### Performance Optimizations
- **Vectorized Operations**: NumPy/SciPy optimized algorithms
- **JIT Compilation**: Numba acceleration for critical functions
- **Memory Efficient**: Chunked processing for large datasets
- **Parallel Processing**: Multi-core support where applicable

## Installation

### Option 1: Miniconda (Recommended)
```bash
# Create conda environment
cd sgtool-python
conda env create -f environment.yml
conda activate sgtool-py

# Install in development mode
pip install -e .
```

### Option 2: UV Package Manager
```bash
# Setup with UV
cd sgtool-python
uv sync
```

## Quick Start

### 🚀 **AUTO-PROCESSING (Recommended)**
**One command to process everything:**

```bash
# Automatic processing with all filters and interactive maps
uv run sgtool-py auto-process /path/to/csv/directory

# With custom configuration
uv run sgtool-py create-config my_config.json  # Create config file
uv run sgtool-py auto-process /path/to/csv/directory --config my_config.json

# Standalone script
uv run python scripts/auto_process_example.py
```

**This automatically:**
- ✅ Auto-detects magnetic field parameters from your data location
- ✅ Applies ALL SGTool filters (RTP, RTE, THG, Analytic Signal, Tilt Angle, etc.)
- ✅ Creates interactive Folium maps with toggleable layers
- ✅ Generates comprehensive summary report
- ✅ Saves results in multiple formats (CSV, GeoTIFF, NumPy)

### 📋 **Manual Processing**

```bash
# Process directory with specific filters
uv run sgtool-py process-directory /path/to/csv/files --filters rtp,thg,analytic_signal

# Apply single filter to existing grid
uv run sgtool-py filter input.npy --filter-name rtp --inclination 70 --declination 2

# Get dataset information
uv run sgtool-py info /path/to/csv/files

# List available filters
uv run sgtool-py list-filters
```

## Data Format

SGTool-Python expects CSV files from the magbase → flight_path_segmenter processing pipeline with fields:
- Magnetic measurements: `"R1 [nT]"`, `"R2 [nT]"`, `"Btotal1 [nT]"`
- Coordinates: UTM or geographic coordinates
- Metadata: Flight line information, timestamps

## Architecture

- `sgtool_py.core`: Core geophysical algorithms adapted from SGTool
- `sgtool_py.io`: Data I/O for CSV and raster formats
- `sgtool_py.pipeline`: Batch processing workflows
- `sgtool_py.visualization`: Interactive Folium maps and static plots
- `sgtool_py.cli`: Command-line interface

## Credits

Based on SGTool by Mark Jessell (https://github.com/swaxi/SGTool)
Adapted for Python by SINTEF for magnetic survey processing workflows.