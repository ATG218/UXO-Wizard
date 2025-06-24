"""
Project Explorer Widget for UXO Wizard
"""

from PySide6.QtWidgets import (
    QTreeView, QVBoxLayout, QWidget, QToolButton, 
    QHBoxLayout, QMenu, QFileSystemModel, QHeaderView
)
from PySide6.QtCore import Qt, Signal, QDir, QModelIndex
from PySide6.QtGui import QIcon, QAction
from loguru import logger


class ProjectExplorer(QWidget):
    """Tree-based project explorer for navigating files and datasets"""
    
    # Signals
    file_selected = Signal(str)
    project_changed = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.current_project_path = None
        self.setup_ui()
        
    def setup_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
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
            "*.png", "*.jpg", "*.jpeg", "*.uxo"  # Custom project files
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
        
        # Layout
        layout.addLayout(toolbar_layout)
        layout.addWidget(self.tree_view)
        self.setLayout(layout)
        
        # Set initial directory (home for now)
        self.set_root_path(QDir.homePath())
        
    def set_root_path(self, path):
        """Set the root path for the explorer"""
        self.current_project_path = path
        index = self.model.index(path)
        self.tree_view.setRootIndex(index)
        self.project_changed.emit(path)
        logger.info(f"Project explorer root set to: {path}")
        
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