"""
Main Application Window for UXO Wizard Desktop Suite
"""

from PySide6.QtWidgets import (
    QMainWindow, QApplication, QDockWidget, QMenuBar, QMenu, QToolBar,
    QStatusBar, QVBoxLayout, QWidget, QTabWidget, QSplitter,
    QMessageBox, QLabel, QProgressBar
)
from PySide6.QtCore import Qt, QSettings, Signal, QTimer
from PySide6.QtGui import QIcon, QKeySequence, QAction
from loguru import logger

from .project_explorer import ProjectExplorer
from .console_widget import ConsoleWidget
from .data_viewer import DataViewer
from .themes import ThemeManager

# Try to import map widget, fallback if WebEngine not available
try:
    from .map_widget import MapWidget
    HAS_WEB_ENGINE = True
except ImportError:
    logger.warning("PySide6-WebEngine not available. Map functionality will be limited.")
    HAS_WEB_ENGINE = False
    
    # Create a fallback map widget
    class MapWidget(QWidget):
        def __init__(self):
            super().__init__()
            layout = QVBoxLayout()
            layout.addWidget(QLabel("Map functionality requires PySide6-WebEngine\nInstall with: pip install PySide6-WebEngine"))
            self.setLayout(layout)
            
        def cleanup(self):
            pass


class MainWindow(QMainWindow):
    """Main application window with dockable panels and ribbon-style interface"""
    
    # Signals
    project_changed = Signal(str)
    dataset_loaded = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.theme_manager = ThemeManager()
        self.settings = QSettings("UXO-Wizard", "Desktop-Suite")
        
        self.setup_ui()
        self.setup_menus()
        self.setup_toolbars()
        self.setup_docks()
        self.setup_statusbar()
        self.setup_connections()
        
        self.restore_state()
        logger.info("UXO Wizard Desktop Suite initialized")
        
    def setup_ui(self):
        """Initialize the main UI structure"""
        self.setWindowTitle("UXO Wizard Desktop Suite")
        self.setGeometry(100, 100, 1400, 900)
        
        # Central widget with tab system
        self.central_tabs = QTabWidget()
        self.central_tabs.setTabsClosable(True)
        self.central_tabs.setMovable(True)
        self.setCentralWidget(self.central_tabs)
        
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
        
        self.import_data_action = QAction("&Import Data", self)
        self.import_data_action.setShortcut("Ctrl+I")
        self.import_data_action.triggered.connect(self.import_data)
        file_menu.addAction(self.import_data_action)
        
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
        
    def setup_toolbars(self):
        """Create the toolbar system"""
        # Main toolbar
        main_toolbar = self.addToolBar("Main")
        main_toolbar.setObjectName("MainToolBar")
        
        main_toolbar.addAction(self.new_project_action)
        main_toolbar.addAction(self.open_project_action)
        main_toolbar.addSeparator()
        main_toolbar.addAction(self.import_data_action)
        
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
        # Project Explorer Dock
        self.project_dock = QDockWidget("Project Explorer", self)
        self.project_dock.setObjectName("ProjectExplorerDock")
        self.project_explorer = ProjectExplorer()
        self.project_dock.setWidget(self.project_explorer)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.project_dock)
        
        # Console/Log Dock
        self.console_dock = QDockWidget("Console", self)
        self.console_dock.setObjectName("ConsoleDock")
        self.console_widget = ConsoleWidget()
        self.console_dock.setWidget(self.console_widget)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.console_dock)
        
        # Data Viewer Dock
        self.data_dock = QDockWidget("Data Viewer", self)
        self.data_dock.setObjectName("DataViewerDock")
        self.data_viewer = DataViewer()
        self.data_dock.setWidget(self.data_viewer)
        self.addDockWidget(Qt.RightDockWidgetArea, self.data_dock)
        
        # Map Preview Dock
        self.map_dock = QDockWidget("Map Preview", self)
        self.map_dock.setObjectName("MapPreviewDock")
        self.map_widget = MapWidget()
        self.map_dock.setWidget(self.map_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, self.map_dock)
        
        # Tabify some docks
        self.tabifyDockWidget(self.data_dock, self.map_dock)
        self.data_dock.raise_()
        
        # Add dock toggles to View menu
        self.dock_menu.addAction(self.project_dock.toggleViewAction())
        self.dock_menu.addAction(self.console_dock.toggleViewAction())
        self.dock_menu.addAction(self.data_dock.toggleViewAction())
        self.dock_menu.addAction(self.map_dock.toggleViewAction())
        
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
        self.coord_label = QLabel("No data")
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
        
        # Tab connections
        self.central_tabs.tabCloseRequested.connect(self.close_tab)
        
    def new_project(self):
        """Create a new project"""
        logger.info("Creating new project")
        # TODO: Implement project creation dialog
        self.status_label.setText("New project created")
        
    def open_project(self):
        """Open an existing project"""
        logger.info("Opening project")
        # TODO: Implement project opening dialog
        self.status_label.setText("Project opened")
        
    def import_data(self):
        """Import data into the current project"""
        logger.info("Importing data")
        # TODO: Implement data import dialog
        self.status_label.setText("Importing data...")
        
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
        
        logger.info(f"Changed theme to: {theme_name}")
        
    def open_file(self, filepath):
        """Open a file in a new tab"""
        logger.info(f"Opening file: {filepath}")
        # TODO: Implement file opening based on type
        
        # For now, create a placeholder widget
        widget = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(QLabel(f"File: {filepath}"))
        widget.setLayout(layout)
        
        # Add to tabs
        filename = filepath.split('/')[-1]
        self.central_tabs.addTab(widget, filename)
        
    def close_tab(self, index):
        """Close a tab"""
        self.central_tabs.removeTab(index)
        
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
        
    def restore_state(self):
        """Restore window state from settings"""
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
            
        state = self.settings.value("windowState")
        if state:
            self.restoreState(state)
            
    def closeEvent(self, event):
        """Save state before closing"""
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