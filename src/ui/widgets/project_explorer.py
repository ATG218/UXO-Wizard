"""
Project Explorer Widget for UXO Wizard
"""

from PySide6.QtWidgets import (
    QTreeView, QVBoxLayout, QWidget, QToolButton, 
    QHBoxLayout, QMenu, QFileSystemModel, QHeaderView,
    QStackedWidget, QLabel, QPushButton
)
from PySide6.QtCore import Qt, Signal, QDir, QModelIndex, QSettings
from PySide6.QtGui import QIcon, QAction, QFont
from loguru import logger


class ProjectWelcomeWidget(QWidget):
    """Welcome widget shown when no project is open"""
    
    # Signals
    open_project_requested = Signal()
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        
    def setup_ui(self):
        """Initialize the welcome UI"""
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Add stretch to center content
        layout.addStretch()
        
        # Icon/Logo (you can replace with actual icon later)
        icon_label = QLabel("📁")
        icon_label.setAlignment(Qt.AlignCenter)
        icon_label.setStyleSheet("font-size: 48px; margin-bottom: 10px;")
        layout.addWidget(icon_label)
        
        # Welcome text
        welcome_label = QLabel("Welcome to UXO Wizard")
        welcome_label.setAlignment(Qt.AlignCenter)
        welcome_font = QFont()
        welcome_font.setPointSize(16)
        welcome_font.setBold(True)
        welcome_label.setFont(welcome_font)
        welcome_label.setStyleSheet("color: #888888; margin-bottom: 10px;")
        layout.addWidget(welcome_label)
        
        # Subtitle
        subtitle_label = QLabel("Open a project folder to get started")
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setStyleSheet("color: #666666; margin-bottom: 20px;")
        layout.addWidget(subtitle_label)
        
        # Open Project button
        self.open_project_btn = QPushButton("Open Project")
        self.open_project_btn.setMinimumHeight(40)
        self.open_project_btn.setMinimumWidth(150)
        self.open_project_btn.clicked.connect(self.open_project_requested.emit)
        self.open_project_btn.setStyleSheet("""
            QPushButton {
                background-color: #0d7377;
                color: white;
                border: none;
                border-radius: 6px;
                font-size: 14px;
                font-weight: bold;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #14a085;
            }
            QPushButton:pressed {
                background-color: #0a5d61;
            }
        """)
        
        # Center the button
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(self.open_project_btn)
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        # Add stretch to center content
        layout.addStretch()
        
        self.setLayout(layout)


class ProjectExplorer(QWidget):
    """Tree-based project explorer for navigating files and datasets"""
    
    # Signals
    file_selected = Signal(str)
    project_changed = Signal(str)
    open_project_requested = Signal()
    
    def __init__(self):
        super().__init__()
        self.current_project_path = None
        self.settings = QSettings("UXO-Wizard", "Desktop-Suite")
        self.setup_ui()
        
        # Restore last project after UI is set up
        self.restore_last_project()
        
    def setup_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create stacked widget to switch between welcome and tree view
        self.stacked_widget = QStackedWidget()
        
        # Create welcome widget
        self.welcome_widget = ProjectWelcomeWidget()
        self.welcome_widget.open_project_requested.connect(self.open_project_requested.emit)
        self.stacked_widget.addWidget(self.welcome_widget)
        
        # Create tree view container
        tree_container = QWidget()
        tree_layout = QVBoxLayout()
        tree_layout.setContentsMargins(0, 0, 0, 0)
        
        # Toolbar
        toolbar_layout = QHBoxLayout()
        
        self.refresh_btn = QToolButton()
        self.refresh_btn.setText("↻")
        self.refresh_btn.setToolTip("Refresh")
        self.refresh_btn.clicked.connect(self.refresh_view)
        toolbar_layout.addWidget(self.refresh_btn)
        
        self.collapse_btn = QToolButton()
        self.collapse_btn.setText("−")
        self.collapse_btn.setToolTip("Collapse All")
        self.collapse_btn.clicked.connect(self.collapse_all)
        toolbar_layout.addWidget(self.collapse_btn)
        
        self.expand_btn = QToolButton()
        self.expand_btn.setText("+")
        self.expand_btn.setToolTip("Expand All")
        self.expand_btn.clicked.connect(self.expand_all)
        toolbar_layout.addWidget(self.expand_btn)
        
        self.home_btn = QToolButton()
        self.home_btn.setText("🏠")
        self.home_btn.setToolTip("Go to Home Directory")
        self.home_btn.clicked.connect(self.go_home)
        toolbar_layout.addWidget(self.home_btn)
        
        # Add "Close Project" button to toolbar
        self.close_project_btn = QToolButton()
        self.close_project_btn.setText("✕")
        self.close_project_btn.setToolTip("Close Project")
        self.close_project_btn.clicked.connect(self.close_project)
        toolbar_layout.addWidget(self.close_project_btn)
        
        toolbar_layout.addStretch()
        
        # Tree view
        self.tree_view = QTreeView()
        self.tree_view.setAlternatingRowColors(True)
        self.tree_view.setAnimated(True)
        self.tree_view.setSortingEnabled(True)
        
        # File system model
        self.model = QFileSystemModel()
        self.model.setRootPath(QDir.rootPath())
        
        # Set filters for relevant file types
        self.model.setNameFilters([
            "*.csv", "*.txt", "*.dat", "*.xlsx", "*.xls",
            "*.json", "*.geojson", "*.shp", "*.tif", "*.tiff",
            "*.png", "*.jpg", "*.jpeg", "*.uxo"  # UXO project files
        ])
        self.model.setNameFilterDisables(False)  # Hide non-matching files
        
        self.tree_view.setModel(self.model)
        
        # Configure columns
        self.tree_view.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tree_view.hideColumn(1)  # Hide size column
        self.tree_view.hideColumn(2)  # Hide type column
        self.tree_view.hideColumn(3)  # Hide date column
        
        # Context menu
        self.tree_view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree_view.customContextMenuRequested.connect(self.show_context_menu)
        
        # Selection handling
        self.tree_view.doubleClicked.connect(self.handle_double_click)
        self.tree_view.clicked.connect(self.handle_click)
        
        # Add to tree container
        tree_layout.addLayout(toolbar_layout)
        tree_layout.addWidget(self.tree_view)
        tree_container.setLayout(tree_layout)
        
        # Add tree container to stacked widget
        self.stacked_widget.addWidget(tree_container)
        
        # Add stacked widget to main layout
        layout.addWidget(self.stacked_widget)
        self.setLayout(layout)
        
        # Start with welcome widget
        self.show_welcome()
        
    def show_welcome(self):
        """Show the welcome widget"""
        self.stacked_widget.setCurrentIndex(0)
        self.current_project_path = None
        self.project_changed.emit(None)  # Emit None to indicate no project
        logger.info("Showing project explorer welcome screen")
        
    def show_tree_view(self):
        """Show the tree view"""
        self.stacked_widget.setCurrentIndex(1)
        
    def close_project(self):
        """Close the current project and return to welcome screen"""
        self.show_welcome()
        # Clear saved project path
        self.settings.remove("last_project_path")
        logger.info("Project closed")
        
    def set_root_path(self, path):
        """Set the root path for the explorer"""
        if path and QDir(path).exists():
            self.current_project_path = path
            index = self.model.index(path)
            self.tree_view.setRootIndex(index)
            self.show_tree_view()  # Switch to tree view when project is opened
            self.project_changed.emit(path)
            
            # Save project path to settings
            self.settings.setValue("last_project_path", path)
            logger.info(f"Project explorer root set to: {path}")
        else:
            logger.warning(f"Invalid path: {path}")
            self.show_welcome()
        
    def get_selected_path(self):
        """Get the currently selected file path"""
        indexes = self.tree_view.selectedIndexes()
        if indexes:
            return self.model.filePath(indexes[0])
        return None
        
    def handle_double_click(self, index: QModelIndex):
        """Handle double-click on file"""
        if not self.model.isDir(index):
            file_path = self.model.filePath(index)
            self.file_selected.emit(file_path)
            logger.info(f"File double-clicked: {file_path}")
            
    def handle_click(self, index: QModelIndex):
        """Handle single click on item"""
        file_path = self.model.filePath(index)
        logger.debug(f"Item clicked: {file_path}")
        
    def show_context_menu(self, position):
        """Show context menu for selected item"""
        indexes = self.tree_view.selectedIndexes()
        if not indexes:
            return
            
        index = indexes[0]
        file_path = self.model.filePath(index)
        is_dir = self.model.isDir(index)
        
        menu = QMenu()
        
        if not is_dir:
            open_action = QAction("Open", self)
            open_action.triggered.connect(lambda: self.file_selected.emit(file_path))
            menu.addAction(open_action)
            
            menu.addSeparator()
            
            process_action = QAction("Process...", self)
            menu.addAction(process_action)
            
        else:
            set_root_action = QAction("Set as Root", self)
            set_root_action.triggered.connect(lambda: self.set_root_path(file_path))
            menu.addAction(set_root_action)
            
            menu.addSeparator()
            
            new_folder_action = QAction("New Folder", self)
            menu.addAction(new_folder_action)
            
        menu.addSeparator()
        
        rename_action = QAction("Rename", self)
        menu.addAction(rename_action)
        
        delete_action = QAction("Delete", self)
        menu.addAction(delete_action)
        
        menu.addSeparator()
        
        properties_action = QAction("Properties", self)
        menu.addAction(properties_action)
        
        menu.exec_(self.tree_view.mapToGlobal(position))
        
    def refresh_view(self):
        """Refresh the file view"""
        if self.current_project_path:
            # Force model to update
            root = self.model.rootPath()
            self.model.setRootPath("")
            self.model.setRootPath(root)
            logger.info("Project explorer refreshed")
        
    def collapse_all(self):
        """Collapse all tree items"""
        self.tree_view.collapseAll()
        
    def expand_all(self):
        """Expand all tree items"""
        self.tree_view.expandAll()
        
    def set_name_filters(self, filters):
        """Update the file name filters"""
        self.model.setNameFilters(filters)
        
    def show_hidden_files(self, show):
        """Toggle showing hidden files"""
        if show:
            self.model.setFilter(self.model.filter() | QDir.Hidden)
        else:
            self.model.setFilter(self.model.filter() & ~QDir.Hidden) 
    
    def go_home(self):
        """Navigate to the home directory"""
        self.set_root_path(QDir.homePath())
        
    def restore_last_project(self):
        """Restore the last opened project from settings"""
        last_project_path = self.settings.value("last_project_path")
        
        if last_project_path and QDir(last_project_path).exists():
            logger.info(f"Restoring last project: {last_project_path}")
            self.set_root_path(last_project_path)
        else:
            if last_project_path:
                logger.warning(f"Last project path no longer exists: {last_project_path}")
                # Clean up invalid path from settings
                self.settings.remove("last_project_path")
            self.show_welcome() 