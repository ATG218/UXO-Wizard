# Scripts Directory Documentation

This directory contains Python scripts for processing and analyzing magnetic field data collected from magnetometer surveys. The scripts form a comprehensive pipeline for data processing, visualization, and analysis.

## Core Processing Scripts

### `magbase.py`
**Main magnetic data processing pipeline** (1040 lines)
- Reads base station and MagWalk magnetometer data files
- Performs GPS interpolation and coordinate transformations (UTM)
- Calculates residual magnetic anomalies by removing base station trends
- Handles data filtering and quality control
- Core functions: `read_base_station_file()`, `read_magwalk_file()`, `interpolate_gps_gaps()`, `calculate_residual_anomalies()`

### `magbase_post.py` & `magbase_post_copy.py`
**Post-processing visualization modules**
- Generate comprehensive visualizations of processed magnetic data
- Create interactive maps and time series plots
- Handle large datasets with optimized plotting routines
- Support multiple output formats (PNG, HTML, CSV)

## Visualization and Analysis Scripts

### `magvis.py`
**Comprehensive magnetic data visualization tool** (1480 lines)
- Time series plotting with customizable parameters
- Interpolated grid generation and mapping
- Interactive cutoff selection functionality
- Support for both vertical alignment and residual data visualization
- Integration with satellite basemaps and interactive mapping

### `analyze_flight_time.py`
**Flight data analysis and characterization**
- Analyzes magnetometer data quality over flight duration
- Examines altitude changes and speed patterns
- Evaluates GPS signal quality and data continuity
- Provides insights for flight path filtering criteria

## Advanced Processing Scripts

### `vistest.py`
**Basic magnetic survey processing**
- Multi-stage spike detection using MAD (Median Absolute Deviation)
- Power spectral density (PSD) analysis with configurable parameters
- Interactive mapping with Folium integration
- Field interpolation and 2D anomaly visualization
- Satellite basemap integration for geographic context

### `vistestv2.py`
**Enhanced survey processing with Kalman filtering**
- Advanced multi-stage filtering pipeline
- Kalman harmonic filter for rotor noise removal
- Automatic rotor frequency detection
- Enhanced spike detection with multiple thresholds
- Improved 2D field interpolation algorithms

### `vistestv2_uniform.py`
**Uniform timeline processing variant**
- Specialized for datasets requiring temporal regularization
- Uniform sample interval reconstruction
- Optimized for consistent data spacing requirements
- Maintains all advanced filtering capabilities from vistestv2

### `vistestv2 copy.py`
**Development copy with artifact analysis**
- Extended artifact detection and characterization
- Additional debugging and analysis features
- Experimental filtering approaches
- Enhanced spike cataloging capabilities

## Specialized Analysis Tools

### `smearing.py`
**Advanced artifact removal testbed** (527 lines)
- Systematic testing of multiple denoising approaches
- Wavelet-based artifact removal (PyWavelets integration)
- Empirical Mode Decomposition (EMD/EEMD) support
- Robust regression techniques (Huber, RANSAC)
- Comprehensive performance comparison framework
- Methods tested:
  - Polynomial detrending (orders 1-10)
  - Savitzky-Golay filtering
  - Fourier-based filtering (FFT, notch filters)
  - Spline detrending with adaptive parameters
  - Median filtering approaches
  - Butterworth high-pass filters
  - Wavelet denoising (multiple wavelets and decomposition levels)
  - Rolling statistics (mean/median detrending)

## Common Features Across Scripts

### Data Processing Capabilities
- CSV data import with datetime parsing
- GPS coordinate handling and transformations
- Multiple magnetic field column support
- Configurable data trimming and quality control

### Filtering and Noise Removal
- Multi-stage spike detection algorithms
- Notch filtering for power line interference (50Hz)
- High-pass filtering for trend removal
- Kalman filtering for harmonic noise suppression
- Median filtering for robust trend estimation

### Visualization Features
- Time series plotting with configurable scales
- Interactive web maps using Folium
- Satellite basemap integration via Contextily
- Field interpolation and gridding
- Power spectral density analysis
- Symlog scaling for wide dynamic range data

### Output Generation
- High-resolution PNG figures (300 DPI)
- Interactive HTML maps
- CSV export of processed data and spike catalogs
- Comprehensive logging and progress reporting

## Installation and Setup

### Requirements
- **Miniconda** (recommended) or Anaconda
- Python >= 3.11

### Installation
```bash
# Clone the repository
git clone https://github.com/ATG218/UXO-Wizard.git
cd UXO-Wizard/Scripts

# Create and activate conda environment
conda env create -f environment.yml
conda activate uxo-wizard-scripts
```

### Alternative Setup (if environment.yml fails)
```bash
# Create environment manually
conda create -n uxo-wizard-scripts python=3.11
conda activate uxo-wizard-scripts

# Install core dependencies
conda install -c conda-forge numpy pandas matplotlib scipy scikit-learn
conda install -c conda-forge geopandas folium contextily pyproj pykrige
conda install -c conda-forge pywavelets utm alphashape plotly branca

# Install PyEMD via pip
pip install PyEMD
```

## Dependencies

### Core Libraries
- `numpy`, `pandas` - Data manipulation and numerical computing
- `matplotlib` - Static plotting and visualization
- `scipy` - Signal processing and scientific computing
- `geopandas`, `shapely` - Geospatial data handling

### Mapping and Visualization
- `folium` - Interactive web mapping
- `contextily` - Satellite basemap integration
- `branca` - Colormap utilities for Folium

### Advanced Signal Processing
- `PyWavelets` - Wavelet analysis and denoising
- `PyEMD` - Empirical Mode Decomposition
- `sklearn` - Machine learning and robust regression

### Coordinate Systems
- `pyproj` - Coordinate reference system transformations

## Usage Workflow

1. **Data Collection**: Use `analyze_flight_time.py` to characterize raw survey data
2. **Basic Processing**: Apply `magbase.py` for fundamental data processing and anomaly calculation
3. **Visualization**: Use `magvis.py` for comprehensive data visualization
4. **Advanced Processing**: Apply `vistestv2.py` or variants for sophisticated filtering
5. **Artifact Analysis**: Use `smearing.py` to test and optimize artifact removal techniques
6. **Post-processing**: Generate final outputs using `magbase_post.py`

The scripts are designed to work together as a complete magnetic survey processing pipeline, with each script serving specific roles in the data analysis workflow.