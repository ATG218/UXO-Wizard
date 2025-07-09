"""
Main Application Window for UXO Wizard Desktop Suite
"""

import os
# Ensure consistent Qt binding
os.environ["QT_API"] = "pyside6"

from qtpy.QtWidgets import (
    QMainWindow, QApplication, QDockWidget, QMenuBar, QMenu, QToolBar,
    QStatusBar, QVBoxLayout, QWidget, QTabWidget, QSplitter,
    QMessageBox, QLabel, QProgressBar, QFileDialog, QTextEdit
)
from qtpy.QtCore import Qt, QSettings, Signal, QTimer, QDir
from qtpy.QtGui import QIcon, QKeySequence, QAction
from loguru import logger

from .widgets.project_explorer import ProjectExplorer
from .widgets.console_widget import ConsoleWidget
from .widgets.data_viewer import DataViewer
from .themes import ThemeManager



class MainWindow(QMainWindow):
    """Main application window with dockable panels and ribbon-style interface"""
    
    # Signals
    project_changed = Signal(str)
    dataset_loaded = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.theme_manager = ThemeManager()
        self.settings = QSettings("UXO-Wizard", "Desktop-Suite")
        self.open_file_docks = {}
        
        self.setup_ui()
        self.setup_menus()
        self.setup_toolbars()
        self.setup_docks()
        self.setup_statusbar()
        self.setup_connections()
        
        # Restore state or apply default layout if no state exists
        if not self.restore_state():
            logger.info("No saved state found, applying default UI layout.")
            QTimer.singleShot(50, self.apply_default_layout)
        
        logger.info("UXO Wizard Desktop Suite initialized")
        
    def setup_ui(self):
        """Initialize the main UI structure"""
        self.setWindowTitle("UXO Wizard Desktop Suite")
        self.setGeometry(50, 50, 1600, 1000)
        
        # Central widget is removed to allow docks to fill the entire window space
        self.setDockNestingEnabled(True)
        
        # Apply initial theme
        self.theme_manager.apply_theme(self, "dark")
        
    def setup_menus(self):
        """Create the menu system"""
        menubar = self.menuBar()
        
        # File Menu
        file_menu = menubar.addMenu("&File")
        
        self.new_project_action = QAction("&New Project", self)
        self.new_project_action.setShortcut(QKeySequence.New)
        self.new_project_action.triggered.connect(self.new_project)
        file_menu.addAction(self.new_project_action)
        
        self.open_project_action = QAction("&Open Project", self)
        self.open_project_action.setShortcut(QKeySequence.Open)
        self.open_project_action.triggered.connect(self.open_project)
        file_menu.addAction(self.open_project_action)
        
        file_menu.addSeparator()
        
        # Project Save/Load actions
        self.save_project_action = QAction("&Save Project (.uxo)", self)
        self.save_project_action.setShortcut(QKeySequence.Save)
        self.save_project_action.triggered.connect(self.save_project)
        file_menu.addAction(self.save_project_action)
        
        self.save_project_as_action = QAction("Save Project &As...", self)
        self.save_project_as_action.setShortcut(QKeySequence.SaveAs)
        self.save_project_as_action.triggered.connect(self.save_project_as)
        file_menu.addAction(self.save_project_as_action)
        
        self.load_project_action = QAction("&Load Project (.uxo)", self)
        self.load_project_action.setShortcut("Ctrl+L")
        self.load_project_action.triggered.connect(self.load_project)
        file_menu.addAction(self.load_project_action)
        
        file_menu.addSeparator()
        
        self.export_project_info_action = QAction("Export Project Info...", self)
        self.export_project_info_action.triggered.connect(self.export_project_info)
        file_menu.addAction(self.export_project_info_action)
        
        file_menu.addSeparator()
        
        self.exit_action = QAction("E&xit", self)
        self.exit_action.setShortcut(QKeySequence.Quit)
        self.exit_action.triggered.connect(self.close)
        file_menu.addAction(self.exit_action)
        
        # Edit Menu
        edit_menu = menubar.addMenu("&Edit")
        
        self.undo_action = QAction("&Undo", self)
        self.undo_action.setShortcut(QKeySequence.Undo)
        edit_menu.addAction(self.undo_action)
        
        self.redo_action = QAction("&Redo", self)
        self.redo_action.setShortcut(QKeySequence.Redo)
        edit_menu.addAction(self.redo_action)
        
        edit_menu.addSeparator()
        
        self.preferences_action = QAction("&Preferences", self)
        self.preferences_action.setShortcut("Ctrl+,")
        self.preferences_action.triggered.connect(self.show_preferences)
        edit_menu.addAction(self.preferences_action)
        
        # View Menu
        view_menu = menubar.addMenu("&View")
        
        self.theme_menu = view_menu.addMenu("&Theme")
        self.dark_theme_action = QAction("Dark", self)
        self.dark_theme_action.setCheckable(True)
        self.dark_theme_action.setChecked(True)
        self.dark_theme_action.triggered.connect(lambda: self.change_theme("dark"))
        self.theme_menu.addAction(self.dark_theme_action)
        
        self.light_theme_action = QAction("Light", self)
        self.light_theme_action.setCheckable(True)
        self.light_theme_action.triggered.connect(lambda: self.change_theme("light"))
        self.theme_menu.addAction(self.light_theme_action)
        
        self.cyberpunk_theme_action = QAction("🌆 Cyberpunk", self)
        self.cyberpunk_theme_action.setCheckable(True)
        self.cyberpunk_theme_action.triggered.connect(lambda: self.change_theme("cyberpunk"))
        self.theme_menu.addAction(self.cyberpunk_theme_action)
        
        view_menu.addSeparator()
        
        # Dock visibility will be added dynamically
        self.dock_menu = view_menu.addMenu("&Panels")
        
        # Processing Menu
        processing_menu = menubar.addMenu("&Processing")
        
        self.mag_menu = processing_menu.addMenu("&Magnetic")
        self.mag_anomaly_action = QAction("Anomaly Detection", self)
        self.mag_menu.addAction(self.mag_anomaly_action)
        
        self.mag_filter_action = QAction("Apply Filters", self)
        self.mag_menu.addAction(self.mag_filter_action)
        
        self.mag_rtp_action = QAction("Reduction to Pole", self)
        self.mag_menu.addAction(self.mag_rtp_action)
        
        # Tools Menu
        tools_menu = menubar.addMenu("&Tools")
        
        self.batch_process_action = QAction("&Batch Processing", self)
        tools_menu.addAction(self.batch_process_action)
        
        self.plugin_manager_action = QAction("&Plugin Manager", self)
        tools_menu.addAction(self.plugin_manager_action)
        
        # Help Menu
        help_menu = menubar.addMenu("&Help")
        
        self.documentation_action = QAction("&Documentation", self)
        self.documentation_action.setShortcut("F1")
        help_menu.addAction(self.documentation_action)
        
        self.about_action = QAction("&About", self)
        self.about_action.triggered.connect(self.show_about)
        help_menu.addAction(self.about_action)
        
        # Debug Menu (for development/testing)
        self.debug_menu = menubar.addMenu("&Debug")
        self.reset_layout_action = QAction("Reset UI Layout", self)
        self.reset_layout_action.triggered.connect(self.reset_ui_layout)
        self.debug_menu.addAction(self.reset_layout_action)

        
    def setup_toolbars(self):
        """Create the toolbar system"""
        # Main toolbar
        main_toolbar = self.addToolBar("Main")
        main_toolbar.setObjectName("MainToolBar")
        
        # Essential project actions
        main_toolbar.addAction(self.new_project_action)
        main_toolbar.addAction(self.open_project_action)
        main_toolbar.addSeparator()
        main_toolbar.addAction(self.save_project_action)
        main_toolbar.addAction(self.load_project_action)
        
        # Processing toolbar
        process_toolbar = self.addToolBar("Processing")
        process_toolbar.setObjectName("ProcessingToolBar")
        
        # Quick access to common processing tasks
        self.quick_anomaly_action = QAction("Quick Anomaly", self)
        process_toolbar.addAction(self.quick_anomaly_action)
        
        self.quick_filter_action = QAction("Quick Filter", self)
        process_toolbar.addAction(self.quick_filter_action)
        
    def setup_docks(self):
        """Create dockable panels"""
        # Create the map widget first since other widgets depend on it
        from .map.map_widget import UXOMapWidget
        self.map_widget = UXOMapWidget()
        
        # Initialize project manager
        from ..project.project_manager import ProjectManager
        self.project_manager = ProjectManager(self.map_widget.layer_manager)
        
        # Project Explorer Dock (far left)
        self.project_dock = QDockWidget("File Explorer", self)
        self.project_dock.setObjectName("ProjectExplorerDock")
        self.project_explorer = ProjectExplorer()
        self.project_dock.setWidget(self.project_explorer)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.project_dock)
        
        # Layer Control Panel Dock (next to file explorer)
        self.layers_dock = QDockWidget("Layers", self)
        self.layers_dock.setObjectName("LayerControlDock")
        from .map.layer_panel import LayerControlPanel
        self.layers_panel = LayerControlPanel(self.map_widget.layer_manager)
        self.layers_dock.setWidget(self.layers_panel)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.layers_dock)
        
        # Lab Widget Dock (will be tabified with layers)
        self.lab_dock = QDockWidget("Lab", self)
        self.lab_dock.setObjectName("LabDock")
        from .widgets.lab_widget import LabWidget
        self.lab_widget = LabWidget(self.project_root if hasattr(self, 'project_root') else None)
        self.lab_dock.setWidget(self.lab_widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.lab_dock)
        
        # Console/Log Dock (right side)
        self.console_dock = QDockWidget("Console", self)
        self.console_dock.setObjectName("ConsoleDock")
        self.console_widget = ConsoleWidget()
        self.console_dock.setWidget(self.console_widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.console_dock)

        #Map Dock (right side, tabify with console)
        self.map_dock = QDockWidget("Map", self)
        self.map_dock.setObjectName("AdvancedMapDock")
        self.map_dock.setWidget(self.map_widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.map_dock)
        
        # Data Viewer Dock (bottom, spanning full width)
        self.data_dock = QDockWidget("Data Viewer", self)
        self.data_dock.setObjectName("DataViewerDock")
        # Remove the dock title bar to reclaim vertical space
        # from qtpy.QtWidgets import QWidget as _QtEmptyWidget
        # self.data_dock.setTitleBarWidget(_QtEmptyWidget())
        self.data_dock.setFeatures(QDockWidget.DockWidgetMovable)
        self.data_viewer = DataViewer(self.project_manager)
        self.data_dock.setWidget(self.data_viewer)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.data_dock)
        
        # Set data viewer reference in project manager
        self.project_manager.set_data_viewer(self.data_viewer)
        
        # Configure the dock arrangement
        # IMPORTANT: Tabify Lab with Layers (this creates proper tabs)
        self.tabifyDockWidget(self.layers_dock, self.lab_dock)
        self.layers_dock.raise_()  # Layers on top by default
        
        # Tabify Console and Map together (right side)
        self.tabifyDockWidget(self.console_dock, self.map_dock)
        self.map_dock.raise_()  # Map on top by default
        
    def apply_default_layout(self):
        """Set up dock layout with proper proportions"""
        # Force docks back to their original areas. This is the key to a reliable layout reset.
        # This ensures that if a dock was moved or floated, it gets put back in its place.
        self.addDockWidget(Qt.LeftDockWidgetArea, self.project_dock)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.layers_dock)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.console_dock)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.data_dock)
        
        # These docks must be added to the same area as the one they tabify with
        self.addDockWidget(Qt.LeftDockWidgetArea, self.map_dock)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.lab_dock)

        # Configure corners so bottom dock spans full width
        self.setCorner(Qt.BottomLeftCorner, Qt.BottomDockWidgetArea)
        self.setCorner(Qt.BottomRightCorner, Qt.BottomDockWidgetArea)
        
        # Set proportions for the dock arrangement
        # Left side: File Explorer (narrow) + Layers/Lab (medium) 
        # Right side: Map/Console (wide)
        # Bottom: Data Viewer (spans full width)
        
        # Separate the Project Explorer from the Layers/Lab pane with a horizontal split
        self.splitDockWidget(self.project_dock, self.layers_dock, Qt.Horizontal)
        
        # Split the Layers/Lab pane from the Map/Console pane to fill the central gap
        self.splitDockWidget(self.layers_dock, self.console_dock, Qt.Horizontal)

        # Re-tabify Layers and Lab in case the split operation disturbed their tab relationship
        self.tabifyDockWidget(self.layers_dock, self.lab_dock)
        self.layers_dock.raise_()  # Ensure Layers tab is shown on top
        
        # Re-tabify Console and Map to ensure they are grouped correctly after a reset
        self.tabifyDockWidget(self.console_dock, self.map_dock)
        self.map_dock.raise_()  # Map on top by default
        
        # Set horizontal proportions: File Explorer | Layers/Lab | Map/Console
        self.resizeDocks([self.project_dock, self.layers_dock, self.console_dock], [180, 250, 1000], Qt.Horizontal)
        
        # Vertical proportion is now handled automatically by the widget's size policy
        # Set a smaller default height for the Data Viewer dock
        self.resizeDocks([self.data_dock], [120], Qt.Vertical)
        
        # Add dock toggles to View menu, ensuring not to add duplicates
        if not self.dock_menu.actions():
            self.dock_menu.addAction(self.project_dock.toggleViewAction())
            self.dock_menu.addAction(self.layers_dock.toggleViewAction())
            self.dock_menu.addAction(self.lab_dock.toggleViewAction())
            self.dock_menu.addAction(self.console_dock.toggleViewAction())
            self.dock_menu.addAction(self.map_dock.toggleViewAction())
            self.dock_menu.addAction(self.data_dock.toggleViewAction())
        
        # Ensure all docks are visible and properly arranged
        self.project_dock.show()
        self.project_dock.raise_()
        
        # Make sure both layers and lab are visible (since they're tabified)
        self.layers_dock.show()
        self.lab_dock.show()
        self.layers_dock.raise_()  # Layers tab active by default
        
        self.console_dock.show()
        self.map_dock.show()
        self.map_dock.raise_()  # Map tab active by default
        
        self.data_dock.show()
        
    def setup_statusbar(self):
        """Create the status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Status message
        self.status_label = QLabel("Ready")
        self.status_bar.addWidget(self.status_label)
        
        # Progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.hide()
        self.status_bar.addPermanentWidget(self.progress_bar)
        
        # Coordinates display
        self.coord_label = QLabel("Click map for coordinates")
        self.status_bar.addPermanentWidget(self.coord_label)
        
        # Memory usage
        self.memory_label = QLabel("Memory: 0 MB")
        self.status_bar.addPermanentWidget(self.memory_label)
        
        # Update memory usage periodically
        self.memory_timer = QTimer()
        self.memory_timer.timeout.connect(self.update_memory_usage)
        self.memory_timer.start(2000)  # Update every 2 seconds
        
    def setup_connections(self):
        """Connect signals and slots"""
        # Project explorer connections
        self.project_explorer.file_selected.connect(self.open_file)
        self.project_explorer.project_changed.connect(self.project_changed.emit)
        self.project_explorer.project_changed.connect(self.on_project_changed)
        self.project_explorer.open_project_requested.connect(self.open_project)
        
        # Tab connections are no longer needed as docks handle their own closing
        
        # Data viewer connections
        # NOTE: Map integration removed for clean processor architecture
        # self.data_viewer.data_selected.connect(self.update_map_with_data)
        
        # NEW: Connect DataViewer layer creation to map
        # This enables the "Plot on Map" button functionality
        self.data_viewer.layer_created.connect(lambda layer: (
            self.map_widget.add_layer_realtime(layer),
            self.map_dock.raise_(),  # Bring map to front
            logger.info(f"Added layer '{layer.name}' from DataViewer to map")
        ))
        
        # Advanced map connections
        self.map_widget.coordinates_clicked.connect(self.on_map_coordinates_clicked)
        
        # Layer panel connections
        self.layers_panel.zoom_to_layer.connect(self.map_widget.zoom_to_layer)
        self.layers_panel.opacity_changed.connect(self.map_widget.layer_manager.set_layer_opacity)
        
        # Lab widget connections
        self.lab_widget.file_selected.connect(self.open_file)
        self.lab_widget.script_executed.connect(self.run_processing_script)
        
        # Project manager connections
        self.project_manager.project_saved.connect(self.on_project_saved)
        self.project_manager.project_loaded.connect(self.on_project_loaded)
        self.project_manager.project_error.connect(self.on_project_error)
        self.project_manager.working_directory_restored.connect(self.on_working_directory_restored)
        
        # Layer manager connections to mark project as dirty
        self.map_widget.layer_manager.layer_added.connect(lambda: self.project_manager.mark_dirty())
        self.map_widget.layer_manager.layer_removed.connect(lambda: self.project_manager.mark_dirty())
        self.map_widget.layer_manager.layer_style_changed.connect(lambda: self.project_manager.mark_dirty())
        
    def on_project_changed(self, project_path):
        """Handle project path changes"""
        if project_path:
            # Project opened
            folder_name = project_path.split('/')[-1]
            self.setWindowTitle(f"UXO Wizard Desktop Suite - {folder_name}")
            
            # Update lab widget to point to new project's processing folder
            self.project_root = project_path
            self.lab_widget.set_project_root(project_path)
            
            # Update project manager with current working directory
            self.project_manager.set_current_working_directory(project_path)
        else:
            # Project closed
            self.setWindowTitle("UXO Wizard Desktop Suite")
            self.project_manager.set_current_working_directory(None)
            
    def on_map_coordinates_clicked(self, lat, lon):
        """Handle map coordinate clicks"""
        self.coord_label.setText(f"Lat: {lat:.6f}, Lon: {lon:.6f}")
        logger.debug(f"Map clicked at: {lat:.6f}, {lon:.6f}")
        
    def update_map_with_data(self, data):
        """Update map with data from the data viewer"""
        if hasattr(self.map_widget, 'add_data_layer') and data is not None:
            try:
                # Check if data has coordinate columns
                if hasattr(data, 'columns'):
                    coord_cols = []
                    for col in data.columns:
                        col_lower = col.lower()
                        if any(x in col_lower for x in ['lat', 'lon', 'x', 'y', 'coord']):
                            coord_cols.append(col)
                    
                    if len(coord_cols) >= 2:
                        # Add data as points layer - now using direct map widget
                        self.map_widget.add_data_layer("Survey Data", data, "points")
                        self.map_dock.raise_()  # Bring map to front
                        logger.info("Data plotted on map")
                    else:
                        logger.info("No coordinate columns found in data")
                        
            except Exception as e:
                logger.error(f"Error updating map with data: {str(e)}")
    
    def run_processing_script(self, script_path):
        """Handle execution of processing scripts from Lab widget"""
        # Open the script in the main tabbed viewing area
        self.open_file(script_path)
        
    def new_project(self):
        """Create a new project"""
        logger.info("Creating new project")
        # TODO: Implement project creation dialog
        self.status_label.setText("New project created")
        
    def open_project(self):
        """Open an existing project folder"""
        # QFileDialog already imported at top
        
        logger.info("Opening project folder")
        
        # Open folder dialog to select project directory
        folder_dialog = QFileDialog(self)
        folder_dialog.setWindowTitle("Open Project Folder")
        folder_dialog.setFileMode(QFileDialog.Directory)
        folder_dialog.setOption(QFileDialog.ShowDirsOnly, True)
        
        # Set initial directory to user's home or last used project path
        last_project_path = self.settings.value("last_project_path", QDir.homePath())
        folder_dialog.setDirectory(last_project_path)
        
        if folder_dialog.exec():
            selected_folders = folder_dialog.selectedFiles()
            if selected_folders:
                project_folder = selected_folders[0]
                
                # Set the project explorer to this folder
                self.project_explorer.set_root_path(project_folder)
                
                # Show and bring the project explorer to front
                self.project_dock.show()
                self.project_dock.raise_()
                self.project_dock.activateWindow()
                
                # Save this path for next time
                self.settings.setValue("last_project_path", project_folder)
                
                # Update status
                folder_name = project_folder.split('/')[-1]
                self.status_label.setText(f"Project opened: {folder_name}")
                
                logger.info(f"Project folder opened: {project_folder}")
        
    def show_preferences(self):
        """Show preferences dialog"""
        logger.info("Showing preferences")
        # TODO: Implement preferences dialog
        QMessageBox.information(self, "Preferences", "Preferences dialog coming soon!")
        
    def show_about(self):
        """Show about dialog"""
        about_text = """
        <h2>UXO Wizard Desktop Suite</h2>
        <p>Version 0.1.0</p>
        <p>A comprehensive toolkit for the analysis and visualization of 
        drone-collected sensor data for unexploded ordnance (UXO) detection.</p>
        <p>© 2024 UXO Wizard Team</p>
        """
        QMessageBox.about(self, "About UXO Wizard", about_text)
        
    def change_theme(self, theme_name):
        """Change the application theme"""
        self.theme_manager.apply_theme(self, theme_name)
        
        # Update theme actions
        self.dark_theme_action.setChecked(theme_name == "dark")
        self.light_theme_action.setChecked(theme_name == "light")
        self.cyberpunk_theme_action.setChecked(theme_name == "cyberpunk")
        
        logger.info(f"Changed theme to: {theme_name}")
        
    def open_file(self, filepath):
        """Open a file in appropriate viewer"""
        logger.info(f"Opening file: {filepath}")
        
        try:
            filename = filepath.split('/')[-1]
            
            # Handle data files - open ONLY in the bottom data viewer dock
            if filepath.lower().endswith(('.csv', '.xlsx', '.xls', '.json')):
                # Load data in the main data viewer dock
                self.data_viewer.load_data(filepath)
                self.data_dock.show()
                self.data_dock.raise_()  # Bring data viewer to front
                
                # Update status
                self.status_label.setText(f"Data loaded: {filename}")
                logger.info(f"Data file loaded in bottom dock: {filepath}")
                return

            # Handle other files - check if already open in a dock
            if filepath in self.open_file_docks:
                self.open_file_docks[filepath].raise_()
                return
            
            # Create appropriate viewer for non-data files
            if filepath.lower().endswith(('.txt', '.dat', '.log', '.py', '.md')):
                # Text file viewer
                widget = self.create_text_viewer(filepath)
            else:
                # Default placeholder for unsupported files
                widget = QWidget()
                layout = QVBoxLayout()
                layout.addWidget(QLabel(f"File: {filepath}\nFile type not supported for viewing"))
                widget.setLayout(layout)
            
            # Create a new dock for the file viewer
            dock = QDockWidget(filename, self)
            dock.setObjectName(f"FileDock_{filename}")
            dock.setWidget(widget)
            dock.setAttribute(Qt.WA_DeleteOnClose)
            
            # Add to main window and tabify with the map
            self.addDockWidget(Qt.RightDockWidgetArea, dock)
            self.tabifyDockWidget(self.map_dock, dock)
            dock.raise_()
            
            # Track the dock and clean up when it's destroyed
            self.open_file_docks[filepath] = dock
            dock.destroyed.connect(lambda: self.open_file_docks.pop(filepath, None))
            
            # Update status
            self.status_label.setText(f"Opened: {filename}")
            
        except Exception as e:
            logger.error(f"Error opening file {filepath}: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to open file:\n{str(e)}")
    
    def create_text_viewer(self, filepath):
        """Create a simple text viewer widget"""
        # QTextEdit and QVBoxLayout already imported at top
        
        widget = QWidget()
        layout = QVBoxLayout()
        
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                text_edit.setPlainText(content)
        except UnicodeDecodeError:
            try:
                with open(filepath, 'r', encoding='latin-1') as f:
                    content = f.read()
                    text_edit.setPlainText(content)
            except Exception as e:
                text_edit.setPlainText(f"Error reading file: {str(e)}")
        except Exception as e:
            text_edit.setPlainText(f"Error reading file: {str(e)}")
        
        layout.addWidget(text_edit)
        widget.setLayout(layout)
        
        return widget
        
    def update_memory_usage(self):
        """Update memory usage display"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        self.memory_label.setText(f"Memory: {memory_mb:.1f} MB")
        
    def show_progress(self, message, maximum=0):
        """Show progress in status bar"""
        self.status_label.setText(message)
        self.progress_bar.setMaximum(maximum)
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        
    def update_progress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)
        
    def hide_progress(self):
        """Hide progress bar"""
        self.progress_bar.hide()
        self.status_label.setText("Ready")
    
    def save_project(self):
        """Save current project"""
        try:
            if self.project_manager.current_file_path:
                # Save to existing path
                success = self.project_manager.save_project(self.project_manager.current_file_path)
                if success:
                    self.status_label.setText("Project saved")
            else:
                # No existing path, show save as dialog
                self.save_project_as()
        except Exception as e:
            logger.error(f"Error saving project: {e}")
            QMessageBox.critical(self, "Error", f"Failed to save project:\n{str(e)}")
    
    def save_project_as(self):
        """Save project with new file path"""
        try:
            file_dialog = QFileDialog(self)
            file_dialog.setWindowTitle("Save Project As")
            file_dialog.setNameFilter("UXO Project files (*.uxo);;All files (*)")
            file_dialog.setDefaultSuffix("uxo")
            file_dialog.setAcceptMode(QFileDialog.AcceptSave)
            
            # Set default filename based on current project name
            project_name = self.project_manager.get_current_project_name() or "Untitled Project"
            default_filename = f"{project_name}.uxo"
            file_dialog.selectFile(default_filename)
            
            if file_dialog.exec():
                selected_files = file_dialog.selectedFiles()
                if selected_files:
                    file_path = selected_files[0]
                    success = self.project_manager.save_project(file_path)
                    if success:
                        self.status_label.setText(f"Project saved as: {file_path}")
                        
        except Exception as e:
            logger.error(f"Error saving project as: {e}")
            QMessageBox.critical(self, "Error", f"Failed to save project:\n{str(e)}")
    
    def load_project(self):
        """Load project from .uxo file"""
        try:
            # Check for unsaved changes
            if self.project_manager.is_dirty():
                reply = QMessageBox.question(
                    self, 
                    "Unsaved Changes",
                    "Current project has unsaved changes. Do you want to save before loading?",
                    QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                    QMessageBox.Yes
                )
                
                if reply == QMessageBox.Yes:
                    self.save_project()
                elif reply == QMessageBox.Cancel:
                    return
            
            file_dialog = QFileDialog(self)
            file_dialog.setWindowTitle("Load Project")
            file_dialog.setNameFilter("UXO Project files (*.uxo);;All files (*)")
            file_dialog.setFileMode(QFileDialog.ExistingFile)
            
            if file_dialog.exec():
                selected_files = file_dialog.selectedFiles()
                if selected_files:
                    file_path = selected_files[0]
                    
                    # Validate file before loading
                    from ..project.project_validator import ProjectValidator
                    is_valid, errors = ProjectValidator.validate_uxo_file(file_path)
                    
                    if not is_valid:
                        reply = QMessageBox.question(
                            self,
                            "Invalid Project File",
                            f"Project file has validation errors:\n" + "\n".join(errors[:3]) + 
                            f"\n\nDo you want to attempt to load it anyway?",
                            QMessageBox.Yes | QMessageBox.No,
                            QMessageBox.No
                        )
                        if reply == QMessageBox.No:
                            return
                    
                    success = self.project_manager.load_project(file_path)
                    if success:
                        self.status_label.setText(f"Project loaded: {file_path}")
                        
        except Exception as e:
            logger.error(f"Error loading project: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load project:\n{str(e)}")
    
    def export_project_info(self):
        """Export project information as JSON"""
        try:
            if not self.project_manager.current_project:
                QMessageBox.warning(self, "No Project", "No project is currently open.")
                return
            
            file_dialog = QFileDialog(self)
            file_dialog.setWindowTitle("Export Project Info")
            file_dialog.setNameFilter("JSON files (*.json);;All files (*)")
            file_dialog.setDefaultSuffix("json")
            file_dialog.setAcceptMode(QFileDialog.AcceptSave)
            
            project_name = self.project_manager.get_current_project_name() or "project"
            default_filename = f"{project_name}_info.json"
            file_dialog.selectFile(default_filename)
            
            if file_dialog.exec():
                selected_files = file_dialog.selectedFiles()
                if selected_files:
                    file_path = selected_files[0]
                    success = self.project_manager.export_project_info(file_path)
                    if success:
                        self.status_label.setText(f"Project info exported: {file_path}")
                        
        except Exception as e:
            logger.error(f"Error exporting project info: {e}")
            QMessageBox.critical(self, "Error", f"Failed to export project info:\n{str(e)}")
    
    def on_project_saved(self, file_path):
        """Handle project saved signal"""
        project_name = self.project_manager.get_current_project_name() or "Project"
        self.setWindowTitle(f"UXO Wizard Desktop Suite - {project_name}")
        logger.info(f"Project saved successfully: {file_path}")
    
    def on_project_loaded(self, file_path):
        """Handle project loaded signal"""
        project_name = self.project_manager.get_current_project_name() or "Project"
        self.setWindowTitle(f"UXO Wizard Desktop Suite - {project_name}")
        
        # Show map dock and zoom to data if layers were loaded
        if self.map_widget.layer_manager.layers:
            self.map_dock.raise_()
            self.map_widget.zoom_to_data()
        
        logger.info(f"Project loaded successfully: {file_path}")
    
    def on_project_error(self, operation, error_message):
        """Handle project error signal"""
        QMessageBox.critical(self, f"Project {operation.title()} Error", error_message)
        logger.error(f"Project {operation} error: {error_message}")
    
    def on_working_directory_restored(self, working_directory):
        """Handle working directory restoration"""
        try:
            # Restore the working directory in the project explorer
            self.project_explorer.set_root_path(working_directory)
            
            # Update our internal tracking
            self.project_root = working_directory
            
            # Update lab widget
            self.lab_widget.set_project_root(working_directory)
            
            # IMPORTANT: Show and raise the project explorer dock so user can see it
            self.project_dock.show()
            self.project_dock.raise_()
            self.project_dock.activateWindow()
            
            logger.info(f"Working directory restored and UI updated: {working_directory}")
            
        except Exception as e:
            logger.error(f"Error restoring working directory: {e}")
        
    def restore_state(self):
        """Restore window state from settings. Returns True if state was restored."""
        geometry = self.settings.value("geometry")
        state = self.settings.value("windowState")

        if geometry and state:
            self.restoreGeometry(geometry)
            self.restoreState(state)
            logger.info("Restored window layout from settings.")
            return True
            
        return False
            
    def closeEvent(self, event):
        """Save state before closing"""
        # Check for unsaved project changes
        if self.project_manager.is_dirty():
            reply = QMessageBox.question(
                self, 
                "Unsaved Changes",
                "You have unsaved project changes. Do you want to save before exiting?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                QMessageBox.Yes
            )
            
            if reply == QMessageBox.Yes:
                self.save_project()
                # If save was cancelled or failed, don't exit
                if self.project_manager.is_dirty():
                    event.ignore()
                    return
            elif reply == QMessageBox.Cancel:
                event.ignore()
                return
        
        # Save application state
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())
        
        # Confirm exit
        reply = QMessageBox.question(
            self, 
            "Exit", 
            "Are you sure you want to exit?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def reset_ui_layout(self):
        """Force-apply the default dock layout."""
        logger.info("Resetting UI layout to default.")
        
        # Show all docks in case some were hidden
        for dock in [self.project_dock, self.layers_dock, self.lab_dock, 
                     self.console_dock, self.map_dock, self.data_dock]:
            dock.show()
        
        # We call the internal setup method which handles splitting and sizing
        self.apply_default_layout()
        
        QMessageBox.information(
            self, 
            "Layout Reset", 
            "The user interface layout has been reset to its default configuration."
        )


def main():
    """Main entry point"""
    app = QApplication([])
    app.setApplicationName("UXO Wizard Desktop Suite")
    app.setOrganizationName("UXO-Wizard")
    
    window = MainWindow()
    window.show()
    
    app.exec()


if __name__ == "__main__":
    main() 