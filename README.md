# UXO Wizard

**A comprehensive toolkit for the analysis and visualization of drone-collected sensor data for unexploded ordnance (UXO) detection**

*🚧 Work in Progress - actively developed for research and operational use*

## Overview

UXO Wizard is a Python-based suite of tools designed for processing, analyzing, and visualizing multi-sensor drone data collected for unexploded ordnance detection and geophysical surveys. The toolkit specializes in magnetic field measurements but is extensible to other sensor modalities including multispectral imaging and gamma radiation detection.

## Key Features

🧭 **Magnetic Survey Processing**
- Advanced magnetic anomaly detection and visualization
- Multi-stage filtering and noise reduction (Kalman filters, wavelets, EMD)
- Flight path segmentation and quality assessment
- Interactive web-based visualizations with satellite basemaps

🗺️ **Geophysical Analysis** 
- Reduction to Pole (RTP) and Total Horizontal Gradient processing
- Analytic signal and tilt angle calculations
- Advanced interpolation and gridding algorithms
- Multi-scale anomaly analysis

📊 **Data Pipeline**
- Automated batch processing workflows
- GPS interpolation and coordinate transformations
- Base station correction and diurnal variation removal
- Comprehensive quality control and reporting

🌐 **Visualization & Mapping**
- Interactive Folium maps with toggleable layers
- Time series analysis and frequency domain visualization
- Export capabilities (PNG, HTML, CSV, GeoTIFF)
- Statistical analysis and spike detection

## Repository Structure

```
UXO-Wizard/
├── Scripts/              # Core magnetic data processing scripts
├── sgtool-python/        # Advanced geophysical processing toolkit
├── sgtool/              # Original QGIS plugin (reference)
└── README.md            # This file
```

### Scripts Directory
Python scripts for magnetic survey data processing and analysis:
- **Magnetic Processing**: `magbase.py`, `magvis.py` - Core data processing and visualization
- **Advanced Analysis**: `vistestv2.py`, `smearing.py` - Enhanced filtering and artifact removal
- **Utilities**: `analyze_flight_time.py`, `grid_interpolator.py` - Data analysis tools

### SGTool-Python
Standalone geophysical processing toolkit adapted from SGTool:
- Command-line interface for batch processing
- Vectorized algorithms for performance
- Interactive visualization capabilities
- Automated magnetic field parameter detection

## Quick Start

### Prerequisites
- **Miniconda** or Anaconda (recommended)
- Python >= 3.9
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ATG218/UXO-Wizard.git
   cd UXO-Wizard
   ```

2. **Set up Scripts environment**
   ```bash
   cd Scripts
   conda env create -f environment.yml
   conda activate uxo-wizard-scripts
   ```

3. **Set up SGTool-Python environment**
   ```bash
   cd ../sgtool-python
   conda env create -f environment.yml
   conda activate sgtool-py
   pip install -e .
   ```

### Basic Usage

**Process magnetic survey data:**
```bash
# Activate Scripts environment
conda activate uxo-wizard-scripts

# Run basic processing
python magbase.py your_data.csv

# Advanced filtering and visualization
python vistestv2.py your_data.csv
```

**Batch geophysical processing:**
```bash
# Activate SGTool environment
conda activate sgtool-py

# Auto-process entire directory
sgtool-py auto-process /path/to/csv/files

# Apply specific filters
sgtool-py process-directory /path/to/csv/files --filters rtp,thg,analytic_signal
```

## Data Format Support

UXO Wizard works with CSV files containing:
- **Magnetic measurements**: `"R1 [nT]"`, `"R2 [nT]"`, `"Btotal1 [nT]"`
- **GPS coordinates**: Latitude/Longitude or UTM
- **Timestamps**: For temporal analysis and base station correction
- **Flight metadata**: Altitude, heading, speed

## Contributing

This is an active research project. Contributions are welcome!

### Development Setup
```bash
# Fork the repository and clone your fork
git clone https://github.com/YOUR_USERNAME/UXO-Wizard.git
cd UXO-Wizard

# Create feature branch
git checkout -b your-feature-name

# Make your changes and commit
git add .
git commit -m "Description of your changes"

# Push to your fork and create a pull request
git push origin your-feature-name
```

### Guidelines
- Follow existing code style and documentation patterns
- Add tests for new functionality where applicable
- Update documentation for new features
- Ensure compatibility with existing data workflows

## Research Applications

UXO Wizard has been developed for and applied to:
- Maritime unexploded ordnance surveys
- Archaeological geophysical prospection  
- Environmental monitoring with magnetic sensors
- Drone-based geophysical survey workflows
- Multi-sensor data fusion research

## Credits

**Development Team:**
- Aleksander Garbuz (MIT/SINTEF) - Primary developer
- Mark Jessell - Original SGTool algorithms

**Institutional Support:**
- SINTEF - Norwegian research organization
- MIT - Research collaboration

## License

*License information to be determined - currently for research use*

## Citation

If you use UXO Wizard in your research, please cite:
```
Garbuz, A. (2025). UXO Wizard: A Python toolkit for drone-based magnetic survey analysis. 
SINTEF/MIT Research Collaboration. https://github.com/ATG218/UXO-Wizard
```

---

📧 **Contact**: For questions, issues, or collaboration opportunities, please open an issue on GitHub or contact the development team.

🔬 **Research Note**: This toolkit is under active development as part of ongoing research into autonomous geophysical survey methods and UXO detection algorithms.