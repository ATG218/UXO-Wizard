# UXO Wizard Desktop Suite - Quick Start Guide

## Overview

This is the basic PySide6 skeleton for the UXO Wizard Desktop Suite. It provides:

- Main application window with dockable panels
- Project explorer for file navigation
- Console for logging and output
- Data viewer for tabular data
- Interactive map widget
- Dark/Light theme support
- Menu system and toolbars

## Installation

1. **Clone the repository** (if not already done):
   ```bash
   git clone https://github.com/ATG218/UXO-Wizard.git
   cd UXO-Wizard
   ```

2. **Create a virtual environment**:
   ```bash
   # Using conda (recommended)
   conda create -n uxo-wizard-gui python=3.9
   conda activate uxo-wizard-gui
   
   # Or using venv
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

```bash
python main.py
```

## Features Overview

### Main Window
- **Tab-based interface**: Open multiple files/views in tabs
- **Dockable panels**: Customize your workspace by moving panels
- **Theme switching**: Toggle between dark and light themes via View menu

### Project Explorer (Left Panel)
- Navigate your file system
- Filter for relevant file types (CSV, JSON, etc.)
- Double-click to open files
- Right-click for context menu

### Console (Bottom Panel)
- View application logs
- Filter by log level
- Auto-scroll option
- Clear console button

### Data Viewer (Right Panel)
- Load CSV, Excel, or JSON files
- Sort and filter columns
- View basic statistics
- Export functionality (coming soon)

### Map Widget (Right Panel - Tab 2)
- Interactive Folium-based maps
- Multiple basemap options
- Drawing and measurement tools
- Layer management

### Status Bar
- Current status messages
- Progress bar for long operations
- Memory usage monitor
- Coordinate display (when map data loaded)

## Current State

This is a **skeleton implementation** with:
- ✅ Basic UI structure
- ✅ Dockable panel system
- ✅ Theme support
- ✅ File navigation
- ✅ Basic data viewing
- ⚠️ Processing functions (TODO)
- ⚠️ Data import/export (TODO)
- ⚠️ Magnetic processing (TODO)
- ⚠️ Database integration (TODO)

## Next Steps

According to the mission plan, the next development phases include:

1. **Data Pipeline Integration** (Month 3)
   - CSV/Excel importers
   - Coordinate system handlers
   - Data validation

2. **Magnetic Processing** (Months 4-6)
   - Anomaly detection
   - Filtering algorithms
   - Visualization tools

3. **Database Integration**
   - SQLite project storage
   - Redis caching layer

## Known Issues

- Map widget requires internet for tile loading (offline tiles coming soon)
- Some menu items show placeholder messages
- File type associations not yet implemented
- Export functions not yet implemented

## Development Tips

- Check `uxo_wizard.log` for detailed debug information
- Window state is saved between sessions
- Use Ctrl+, (Cmd+, on Mac) for preferences (coming soon)
- F1 for documentation (coming soon)

## Contributing

Feel free to:
- Report issues
- Suggest UI improvements
- Implement TODO items
- Add new features

---

For more information, see the [Mission Plan](UXO_WIZARD_MISSION_PLAN.md) 