"""
Lab Widget - Processing Output Explorer
"""

import os
import sys
import subprocess
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTreeView, QLabel, QPushButton,
    QFileSystemModel, QMenu, QMessageBox, QToolButton, QFrame, QSplitter,
    QTextEdit, QTabWidget, QStackedWidget, QDialog
)
from qtpy.QtCore import Qt, QDir, QFileInfo, Signal, QModelIndex, QTimer, QSettings, QItemSelectionModel
from qtpy.QtGui import QFont, QIcon, QDesktopServices
from .processing.processing_dialog import ProcessingDialog
from loguru import logger


class LabWidget(QWidget):
    """Lab widget for exploring processing outputs and scripts"""
    
    # Signals
    file_selected = Signal(str)  # file_path
    script_executed = Signal(str)  # script_path
    layer_created = Signal(object)  # Emits UXOLayer for map integration
    plot_generated = Signal(object, str)  # Figure, title - for opening plots in data viewer
    processing_complete = Signal(object)  # ProcessingResult - emitted when processing completes
    
    def __init__(self, project_root: str = None, project_manager=None):
        super().__init__()
        self.settings = QSettings("UXO-Wizard", "Desktop-Suite")
        self.project_manager = project_manager
        
        # Initialize with provided project_root or restore from settings
        if project_root:
            self.project_root = project_root
        else:
            # Try to restore from settings (use same key as Project Explorer for sync)
            self.project_root = self.settings.value("last_project_path", os.getcwd())
            
        self.processed_path = os.path.join(self.project_root, "processed")
        self.has_processed_folder = False
        
        self.setup_ui()
        
        # Restore last project after UI is set up (if no project_root was provided)
        if not project_root:
            self.restore_last_project()
        else:
            self.check_processed_folder()
        
        # Auto-refresh timer to check for processed folder
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.check_processed_folder)
        self.refresh_timer.start(3000)  # Check every 3 seconds
        
    def setup_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Header
        self.create_header(layout)
        
        # Toolbar
        self.create_toolbar(layout)
        
        # Stacked widget to switch between placeholder and content
        self.stacked_widget = QStackedWidget()
        
        # Placeholder widget (shown when no processed folder)
        self.placeholder_widget = self.create_placeholder_widget()
        self.stacked_widget.addWidget(self.placeholder_widget)
        
        # Main content area (shown when processed folder exists)
        self.content_widget = self.create_content_area()
        self.stacked_widget.addWidget(self.content_widget)
        
        layout.addWidget(self.stacked_widget)
        self.setLayout(layout)
        
    def create_header(self, layout):
        """Create the header section"""
        header = QFrame()
        header.setFixedHeight(30)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(10, 5, 10, 5)
        
        # Title
        title_label = QLabel("🧪 Lab")
        title_font = QFont()
        title_font.setBold(True) 
        title_font.setPointSize(10)
        title_label.setFont(title_font)
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        
        # Processing folder indicator
        self.path_label = QLabel("Waiting for processed data...")
        self.path_label.setFont(QFont("", 8))
        header_layout.addWidget(self.path_label)
        
        layout.addWidget(header)
        
    def create_toolbar(self, layout):
        """Create the toolbar"""
        toolbar = QFrame()
        toolbar.setFixedHeight(35)
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(5, 3, 5, 3)
        toolbar_layout.setSpacing(2)
        
        # Refresh button
        self.refresh_btn = self.create_toolbar_button("⟳", "Refresh")
        self.refresh_btn.clicked.connect(self.check_processed_folder)
        toolbar_layout.addWidget(self.refresh_btn)
        
        # Open folder button
        self.open_folder_btn = self.create_toolbar_button("□", "Open in Folder")
        self.open_folder_btn.clicked.connect(self.open_processed_folder)
        self.open_folder_btn.setEnabled(False)  # Disabled until processed folder exists
        toolbar_layout.addWidget(self.open_folder_btn)
        
        # Clear outputs button
        self.clear_btn = self.create_toolbar_button("×", "Clear Outputs")
        self.clear_btn.clicked.connect(self.clear_outputs)
        self.clear_btn.setEnabled(False)  # Disabled until processed folder exists
        toolbar_layout.addWidget(self.clear_btn)
        
        toolbar_layout.addStretch()
        
        # Info label
        self.info_label = QLabel("No processed data")
        self.info_label.setFont(QFont("", 8))
        toolbar_layout.addWidget(self.info_label)
        
        layout.addWidget(toolbar)
        
    def create_toolbar_button(self, text: str, tooltip: str) -> QToolButton:
        """Create a toolbar button"""
        btn = QToolButton()
        btn.setText(text)
        btn.setToolTip(tooltip)
        btn.setFixedSize(32, 24)  # Wider to accommodate text
        font = QFont()
        font.setPointSize(8)
        font.setBold(True)
        btn.setFont(font)
        return btn
        
    def create_placeholder_widget(self) -> QWidget:
        """Create the placeholder widget shown when no processed folder exists"""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(20)
        
        # Icon/emoji
        icon_label = QLabel("🔬")
        icon_label.setAlignment(Qt.AlignCenter)
        icon_font = QFont()
        icon_font.setPointSize(48)
        icon_label.setFont(icon_font)
        layout.addWidget(icon_label)
        
        # Main message
        message_label = QLabel("Please process files first to gain access to lab")
        message_label.setAlignment(Qt.AlignCenter)
        message_font = QFont()
        message_font.setPointSize(12)
        message_font.setBold(True)
        message_label.setFont(message_font)
        layout.addWidget(message_label)
        
        # Secondary message
        help_label = QLabel("Process sensor data files to create outputs that will appear here")
        help_label.setAlignment(Qt.AlignCenter)
        help_font = QFont()
        help_font.setPointSize(10)
        help_label.setFont(help_font)
        help_label.setStyleSheet("color: #888888;")
        layout.addWidget(help_label)
        
        # Status
        self.status_label = QLabel("Waiting for 'processed' folder...")
        self.status_label.setAlignment(Qt.AlignCenter)
        status_font = QFont()
        status_font.setPointSize(9)
        status_font.setItalic(True)
        self.status_label.setFont(status_font)
        self.status_label.setStyleSheet("color: #666666;")
        layout.addWidget(self.status_label)
        
        widget.setLayout(layout)
        return widget
        
    def create_content_area(self) -> QWidget:
        """Create the main content area (file browser)"""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Split between file tree and preview
        splitter = QSplitter(Qt.Vertical)
        
        # File tree
        self.tree_view = QTreeView()
        self.tree_view.setHeaderHidden(False)
        self.tree_view.setAlternatingRowColors(True)
        self.tree_view.setSelectionMode(QTreeView.ExtendedSelection)
        self.tree_view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree_view.setSortingEnabled(True)
        
        # Connect signals
        self.tree_view.doubleClicked.connect(self.on_file_double_click)
        self.tree_view.clicked.connect(self.on_file_click)
        self.tree_view.customContextMenuRequested.connect(self.show_context_menu)
        
        splitter.addWidget(self.tree_view)
        
        # Preview/info panel
        self.info_panel = QTabWidget()
        self.info_panel.setMaximumHeight(120)  # Reduced height
        self.info_panel.setMinimumHeight(120)   # Set minimum to keep it compact
        
        # File info tab
        self.file_info = QTextEdit()
        self.file_info.setReadOnly(True)
        self.file_info.setMaximumHeight(100)
        self.info_panel.addTab(self.file_info, "File Info")
        
        # Recent activity tab  
        self.activity_log = QTextEdit()
        self.activity_log.setReadOnly(True)
        self.activity_log.setMaximumHeight(100)
        self.info_panel.addTab(self.activity_log, "Recent Activity")
        
        splitter.addWidget(self.info_panel)
        splitter.setStretchFactor(0, 3)  # Tree takes much more space
        splitter.setStretchFactor(1, 1)  # Info panel takes less space
        
        # Set initial splitter sizes to push info panel to bottom
        splitter.setSizes([300, 120])
        
        layout.addWidget(splitter)
        widget.setLayout(layout)
        return widget
        
    def check_processed_folder(self):
        """Check if processed folder exists and switch views accordingly"""
        processed_exists = os.path.exists(self.processed_path) and os.path.isdir(self.processed_path)
        
        if processed_exists and not self.has_processed_folder:
            # Processed folder just appeared
            self.has_processed_folder = True
            self.setup_file_model()
            self.stacked_widget.setCurrentWidget(self.content_widget)
            self.path_label.setText(f"📁 {os.path.basename(self.processed_path)}")
            self.info_label.setText("Ready")
            
            # Enable toolbar buttons
            self.open_folder_btn.setEnabled(True)
            self.clear_btn.setEnabled(True)
            
            self.log_activity("Lab activated - processed folder found!")
            logger.info(f"Lab widget activated: {self.processed_path}")
            
        elif not processed_exists and self.has_processed_folder:
            # Processed folder disappeared
            self.has_processed_folder = False
            self.stacked_widget.setCurrentWidget(self.placeholder_widget)
            self.path_label.setText("Waiting for processed data...")
            self.info_label.setText("No processed data")
            
            # Disable toolbar buttons
            self.open_folder_btn.setEnabled(False)
            self.clear_btn.setEnabled(False)
            
            logger.info("Lab widget deactivated - processed folder not found")
            
        elif processed_exists and self.has_processed_folder:
            # Refresh the view if already active
            self.refresh_view()
            
        # Update status in placeholder
        if not processed_exists:
            project_name = os.path.basename(self.project_root) if self.project_root else "Unknown"
            self.status_label.setText(f"Looking for processed folder in project: {project_name}")
        
    def setup_file_model(self):
        """Setup the file system model"""
        if not os.path.exists(self.processed_path):
            return
            
        self.file_model = QFileSystemModel()
        self.file_model.setRootPath(self.processed_path)
        
        # Set up filters to show relevant files
        self.file_model.setNameFilters([
            "*.py", "*.csv", "*.txt", "*.json", "*.png", "*.jpg", "*.tiff", 
            "*.dat", "*.xyz", "*.las", "*.laz", "*.shp", "*.geojson", "*.mplplot",
            "*.npz", "*.sgy", "*.SGY", "*.log"
        ])
        self.file_model.setNameFilterDisables(False)
        
        # Apply model to tree view
        self.tree_view.setModel(self.file_model)
        self.tree_view.setRootIndex(self.file_model.index(self.processed_path))
        
        # Configure columns
        self.tree_view.setColumnWidth(0, 200)  # Name
        self.tree_view.setColumnWidth(1, 80)   # Size
        self.tree_view.setColumnWidth(2, 100)  # Type
        self.tree_view.setColumnWidth(3, 120)  # Date Modified
        
        # Sort by date modified by default
        self.tree_view.sortByColumn(3, Qt.DescendingOrder)
        
        # Update path label
        self.path_label.setText(f"📁 {os.path.basename(self.processed_path)}")
        
        logger.info(f"Lab widget initialized for: {self.processed_path}")
        
    def on_file_click(self, index: QModelIndex):
        """Handle file selection - show file info only"""
        file_path = self.file_model.filePath(index)
        file_info = QFileInfo(file_path)
        
        if file_info.isFile():
            self.show_file_info(file_path)
            # Don't emit signals on single click - just show info
            
    def on_file_double_click(self, index: QModelIndex):
        """Handle file double-click - open in data viewer or appropriate application"""
        file_path = self.file_model.filePath(index)
        file_info = QFileInfo(file_path)
        
        if file_info.isFile():
            self.open_file(file_path)
            # Log the action for user feedback
            filename = os.path.basename(file_path)
            if file_path.lower().endswith('.py'):
                self.log_activity(f"Opened script '{filename}' for viewing")
            elif file_path.lower().endswith(('.csv', '.txt', '.dat', '.json', '.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.tif', '.mplplot')):
                self.log_activity(f"Opened '{filename}' in data viewer")
            else:
                self.log_activity(f"Opened '{filename}' with system default")
            
    def show_file_info(self, file_path: str):
        """Show information about the selected file including metadata"""
        file_info = QFileInfo(file_path)
        
        info_text = f"""
<b>File:</b> {file_info.fileName()}<br>
<b>Path:</b> {file_path}<br>
<b>Size:</b> {self.format_file_size(file_info.size())}<br>
<b>Modified:</b> {file_info.lastModified().toString()}<br>
<b>Type:</b> {file_info.suffix() or 'Folder'}<br>
"""
        
        # Add metadata based on file type
        metadata = self.extract_file_metadata(file_path)
        if metadata:
            info_text += "<br><b>Metadata:</b><br>"
            for key, value in metadata.items():
                info_text += f"<b>{key}:</b> {value}<br>"
        
        # Add specific info based on file type
        if file_path.endswith('.py'):
            info_text += "<br><i>💡 Double-click to view script</i>"
        elif file_path.endswith(('.csv', '.txt', '.dat', '.json')):
            info_text += "<br><i>📊 Double-click to open in data viewer</i>"
        elif file_path.endswith(('.png', '.jpg', '.tiff')):
            info_text += "<br><i>🖼️ Double-click to open image</i>"
        else:
            info_text += "<br><i>📄 Double-click to open file</i>"
            
        self.file_info.setHtml(info_text)
        
    def extract_file_metadata(self, file_path: str) -> dict:
        """Extract metadata from different file types"""
        metadata = {}
        
        try:
            if file_path.endswith('.csv'):
                metadata.update(self._extract_csv_metadata(file_path))
            elif file_path.endswith('.json'):
                metadata.update(self._extract_json_metadata(file_path))
            elif file_path.endswith('.txt') or file_path.endswith('.dat'):
                metadata.update(self._extract_text_metadata(file_path))
            elif file_path.endswith('.py'):
                metadata.update(self._extract_python_metadata(file_path))
            elif file_path.endswith(('.png', '.jpg', '.jpeg', '.tiff')):
                metadata.update(self._extract_image_metadata(file_path))
                
        except Exception as e:
            logger.warning(f"Failed to extract metadata from {file_path}: {e}")
            metadata["Error"] = "Could not read metadata"
            
        return metadata
        
    def _extract_csv_metadata(self, file_path: str) -> dict:
        """Extract metadata from CSV files"""
        import pandas as pd
        metadata = {}
        
        try:
            # Read just the header and a few rows for quick analysis
            df = pd.read_csv(file_path, nrows=100)
            metadata["Columns"] = len(df.columns)
            metadata["Sample Rows"] = len(df)
            
            # Try to get total rows by reading the file more efficiently
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    total_rows = sum(1 for line in f) - 1  # Subtract header
                metadata["Total Rows"] = f"{total_rows:,}"
            except:
                metadata["Total Rows"] = "Unknown"
                
            # Show first few column names
            col_names = list(df.columns[:5])
            if len(df.columns) > 5:
                col_names.append("...")
            metadata["Columns Names"] = ", ".join(col_names)
            
            # Detect numeric vs text columns
            numeric_cols = len(df.select_dtypes(include=['number']).columns)
            text_cols = len(df.select_dtypes(include=['object']).columns)
            metadata["Numeric Cols"] = numeric_cols
            metadata["Text Cols"] = text_cols
            
        except Exception as e:
            metadata["Error"] = f"Could not parse CSV: {str(e)[:50]}..."
            
        return metadata
        
    def _extract_json_metadata(self, file_path: str) -> dict:
        """Extract metadata from JSON files"""
        import json
        metadata = {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if isinstance(data, dict):
                metadata["Type"] = "Object"
                metadata["Keys"] = len(data.keys())
                # Show first few keys
                key_names = list(data.keys())[:5]
                if len(data.keys()) > 5:
                    key_names.append("...")
                metadata["Key Names"] = ", ".join(key_names)
            elif isinstance(data, list):
                metadata["Type"] = "Array"
                metadata["Items"] = len(data)
                if data and isinstance(data[0], dict):
                    metadata["Item Type"] = "Objects"
                    if data:
                        metadata["Object Keys"] = len(data[0].keys())
                else:
                    metadata["Item Type"] = "Values"
                    
        except Exception as e:
            metadata["Error"] = f"Could not parse JSON: {str(e)[:50]}..."
            
        return metadata
        
    def _extract_text_metadata(self, file_path: str) -> dict:
        """Extract metadata from text/dat files"""
        metadata = {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            metadata["Lines"] = len(lines)
            
            # Try to detect if it's structured data
            if lines:
                first_line = lines[0].strip()
                # Check if it might be delimited data
                for delimiter in [',', '\t', ';', '|']:
                    if delimiter in first_line:
                        parts = first_line.split(delimiter)
                        if len(parts) > 1:
                            metadata["Possible Format"] = f"Delimited ({delimiter})"
                            metadata["Columns"] = len(parts)
                            break
                            
                # Check for numbers in first line (might be numeric data)
                import re
                numbers = re.findall(r'-?\d+\.?\d*', first_line)
                if len(numbers) > 2:
                    metadata["Numeric Data"] = "Likely"
                    
        except UnicodeDecodeError:
            try:
                # Try binary mode to get basic info
                with open(file_path, 'rb') as f:
                    content = f.read(1024)  # Read first 1KB
                metadata["Format"] = "Binary data"
                metadata["Sample Size"] = "1KB preview"
            except:
                metadata["Error"] = "Could not read file"
        except Exception as e:
            metadata["Error"] = f"Read error: {str(e)[:50]}..."
            
        return metadata
        
    def _extract_python_metadata(self, file_path: str) -> dict:
        """Extract metadata from Python files"""
        import ast
        metadata = {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
                
            metadata["Lines"] = len(lines)
            
            # Try to parse AST for more details
            try:
                tree = ast.parse(content)
                
                functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
                classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
                imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
                
                metadata["Functions"] = len(functions)
                metadata["Classes"] = len(classes)
                metadata["Imports"] = len(imports)
                
                if functions:
                    func_names = [f.name for f in functions[:3]]
                    if len(functions) > 3:
                        func_names.append("...")
                    metadata["Function Names"] = ", ".join(func_names)
                    
            except SyntaxError:
                metadata["Status"] = "Syntax errors present"
            except Exception:
                metadata["Status"] = "Basic text file"
                
        except Exception as e:
            metadata["Error"] = f"Could not parse: {str(e)[:50]}..."
            
        return metadata
        
    def _extract_image_metadata(self, file_path: str) -> dict:
        """Extract metadata from image files"""
        metadata = {}
        
        try:
            # Try using PIL if available
            try:
                from PIL import Image
                with Image.open(file_path) as img:
                    metadata["Dimensions"] = f"{img.width} × {img.height}"
                    metadata["Mode"] = img.mode
                    metadata["Format"] = img.format
                    
                    # Try to get EXIF data
                    if hasattr(img, '_getexif') and img._getexif():
                        exif = img._getexif()
                        if exif:
                            metadata["EXIF Data"] = "Available"
                    
            except ImportError:
                # Fallback: just basic file info
                metadata["Type"] = "Image file"
                metadata["Note"] = "Install PIL for detailed info"
                
        except Exception as e:
            metadata["Error"] = f"Could not read image: {str(e)[:50]}..."
            
        return metadata
        
    def format_file_size(self, size: int) -> str:
        """Format file size in human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"
        
    def open_file(self, file_path: str):
        """Open file with appropriate handler"""
        try:
            if file_path.lower().endswith('.py'):
                # For Python scripts, emit signal to open in main app
                self.script_executed.emit(file_path)
                self.log_activity(f"Opened script: {os.path.basename(file_path)}")
                
            elif file_path.lower().endswith(('.csv', '.txt', '.dat', '.json', '.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.tif', '.mplplot')):
                # For data and image files, emit signal to open in data viewer
                self.file_selected.emit(file_path)
                self.log_activity(f"Opened file in data viewer: {os.path.basename(file_path)}")
                
            else:
                # For other files, open with system default
                QDesktopServices.openUrl(f"file://{file_path}")
                self.log_activity(f"Opened file: {os.path.basename(file_path)}")
                
        except Exception as e:
            logger.error(f"Failed to open file {file_path}: {e}")
            QMessageBox.warning(self, "Error", f"Failed to open file:\n{str(e)}")
            
    def show_context_menu(self, position):
        """Show context menu for file operations"""
        index = self.tree_view.indexAt(position)
        if not index.isValid():
            return
        
        # Get all selected indexes
        selected_indexes = self.tree_view.selectedIndexes()
        # Filter to only get one index per row (column 0)
        selected_indexes = [idx for idx in selected_indexes if idx.column() == 0]
        
        if not selected_indexes:
            return
        
        # If clicked item is not in selection, select it
        if index not in selected_indexes:
            self.tree_view.selectionModel().clearSelection()
            self.tree_view.selectionModel().select(index, QItemSelectionModel.Select)
            selected_indexes = [index]
        
        selected_paths = [self.file_model.filePath(idx) for idx in selected_indexes]
        selected_files = [f for f in selected_paths if QFileInfo(f).isFile()]
        selected_dirs = [f for f in selected_paths if QFileInfo(f).isDir()]
        
        if not selected_paths:
            return
        
        menu = QMenu(self)
        
        if len(selected_paths) == 1 and len(selected_files) == 1:
            # Single file actions
            file_path = selected_files[0]
            
            if file_path.endswith(('.csv', '.txt', '.dat', '.json')):
                open_action = menu.addAction("📊 Open in Data Viewer")
            elif file_path.endswith('.py'):
                open_action = menu.addAction("💡 View Script")
            elif file_path.endswith(('.png', '.jpg', '.tiff')):
                open_action = menu.addAction("🖼️ Open Image")
            else:
                open_action = menu.addAction("📄 Open File")
            open_action.triggered.connect(lambda: self.open_file(file_path))
            
            # Add process option for data files
            if file_path.endswith(('.csv', '.txt', '.dat', '.json', '.npz')):
                process_action = menu.addAction("⚡ Process...")
                process_action.triggered.connect(lambda: self.process_file(file_path))
            
            menu.addSeparator()
            
            if file_path.endswith('.py'):
                run_action = menu.addAction("▶️ Execute Script")
                run_action.triggered.connect(lambda: self.script_executed.emit(file_path))
                
                
            menu.addSeparator()
            
            reveal_action = menu.addAction("📁 Show in Folder")
            reveal_action.triggered.connect(lambda: self.reveal_in_folder(file_path))
            
            delete_action = menu.addAction("🗑️ Delete")
            delete_action.triggered.connect(lambda: self.delete_items([file_path]))
        elif len(selected_paths) == 1 and len(selected_dirs) == 1:
            # Single folder actions
            folder_path = selected_dirs[0]
            
            reveal_action = menu.addAction("📁 Show in Folder")
            reveal_action.triggered.connect(lambda: self.reveal_in_folder(folder_path))
            
            delete_action = menu.addAction("🗑️ Delete Folder")
            delete_action.triggered.connect(lambda: self.delete_items([folder_path]))
        else:
            # Multiple items selected - only show delete option
            total_items = len(selected_files) + len(selected_dirs)
            if len(selected_files) > 0 and len(selected_dirs) > 0:
                item_desc = f"{len(selected_files)} files & {len(selected_dirs)} folders"
            elif len(selected_files) > 0:
                item_desc = f"{len(selected_files)} files"
            else:
                item_desc = f"{len(selected_dirs)} folders"
                
            delete_action = menu.addAction(f"🗑️ Delete {item_desc}")
            delete_action.triggered.connect(lambda: self.delete_items(selected_paths))
            
        menu.exec(self.tree_view.mapToGlobal(position))
        
    def reveal_in_folder(self, file_path: str):
        """Reveal file in system file manager"""
        try:
            if os.name == 'nt':  # Windows
                os.startfile(os.path.dirname(file_path))
            elif os.name == 'posix':  # macOS/Linux
                if sys.platform == 'darwin':  # macOS
                    subprocess.run(['open', '-R', file_path])
                else:  # Linux
                    subprocess.run(['xdg-open', os.path.dirname(file_path)])
        except Exception as e:
            logger.error(f"Failed to reveal file: {e}")
            
    def delete_files(self, file_paths: list):
        """Delete one or more files with confirmation"""
        if len(file_paths) == 1:
            file_path = file_paths[0]
            reply = QMessageBox.question(
                self, 
                "Delete File",
                f"Are you sure you want to delete:\n{os.path.basename(file_path)}?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
        else:
            file_names = "\n".join([f"• {os.path.basename(f)}" for f in file_paths[:10]])  # Show max 10 names
            if len(file_paths) > 10:
                file_names += f"\n... and {len(file_paths) - 10} more files"
            
            reply = QMessageBox.question(
                self, 
                "Delete Files",
                f"Are you sure you want to delete {len(file_paths)} files?\n\n{file_names}",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
        
        if reply == QMessageBox.Yes:
            deleted_count = 0
            failed_files = []
            
            for file_path in file_paths:
                try:
                    os.remove(file_path)
                    deleted_count += 1
                    self.log_activity(f"Deleted: {os.path.basename(file_path)}")
                except Exception as e:
                    logger.error(f"Failed to delete file {file_path}: {e}")
                    failed_files.append(os.path.basename(file_path))
            
            self.refresh_view()
            
            # Show result summary
            if failed_files:
                QMessageBox.warning(
                    self, 
                    "Delete Complete with Errors",
                    f"Successfully deleted {deleted_count} files.\n"
                    f"Failed to delete {len(failed_files)} files:\n" +
                    "\n".join([f"• {f}" for f in failed_files[:5]]) +
                    (f"\n... and {len(failed_files) - 5} more" if len(failed_files) > 5 else "")
                )
            elif len(file_paths) > 1:
                self.log_activity(f"Successfully deleted {deleted_count} files")
    
    def delete_items(self, item_paths: list):
        """Delete one or more files and/or folders with confirmation"""
        if len(item_paths) == 1:
            item_path = item_paths[0]
            item_name = os.path.basename(item_path)
            is_dir = os.path.isdir(item_path)
            item_type = "folder" if is_dir else "file"
            
            reply = QMessageBox.question(
                self, 
                f"Delete {item_type.title()}",
                f"Are you sure you want to delete {item_type}:\n{item_name}?" +
                ("\n\nThis will delete all contents of the folder." if is_dir else ""),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
        else:
            files = [f for f in item_paths if os.path.isfile(f)]
            dirs = [f for f in item_paths if os.path.isdir(f)]
            
            item_names = []
            if files:
                item_names.extend([f"📄 {os.path.basename(f)}" for f in files[:5]])
            if dirs:
                item_names.extend([f"📁 {os.path.basename(d)}" for d in dirs[:5]])
            
            total_shown = len(files[:5]) + len(dirs[:5])
            remaining = len(item_paths) - total_shown
            
            if remaining > 0:
                item_names.append(f"... and {remaining} more items")
            
            item_list = "\n".join(item_names)
            
            reply = QMessageBox.question(
                self, 
                "Delete Items",
                f"Are you sure you want to delete {len(item_paths)} items?\n\n{item_list}" +
                ("\n\nThis will delete all contents of any folders." if dirs else ""),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
        
        if reply == QMessageBox.Yes:
            deleted_count = 0
            failed_items = []
            
            for item_path in item_paths:
                try:
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                        self.log_activity(f"Deleted folder: {os.path.basename(item_path)}")
                    else:
                        os.remove(item_path)
                        self.log_activity(f"Deleted file: {os.path.basename(item_path)}")
                    deleted_count += 1
                except Exception as e:
                    logger.error(f"Failed to delete {item_path}: {e}")
                    failed_items.append(os.path.basename(item_path))
            
            self.refresh_view()
            
            # Show result summary
            if failed_items:
                QMessageBox.warning(
                    self, 
                    "Delete Complete with Errors",
                    f"Successfully deleted {deleted_count} items.\n"
                    f"Failed to delete {len(failed_items)} items:\n" +
                    "\n".join([f"• {f}" for f in failed_items[:5]]) +
                    (f"\n... and {len(failed_items) - 5} more" if len(failed_items) > 5 else "")
                )
            elif len(item_paths) > 1:
                self.log_activity(f"Successfully deleted {deleted_count} items")
    
    def delete_file(self, file_path: str):
        """Delete a single file with confirmation (legacy method)"""
        self.delete_files([file_path])
                
    def refresh_view(self):
        """Refresh the file view"""
        self.file_model.directoryLoaded.connect(self.on_directory_loaded)
        
        # Count files
        total_files = 0
        if os.path.exists(self.processed_path):
            for root, dirs, files in os.walk(self.processed_path):
                total_files += len(files)
                
        self.info_label.setText(f"{total_files} files")
        
    def on_directory_loaded(self):
        """Handle directory loading completion"""
        self.info_label.setText("Ready")
        
    def open_processed_folder(self):
        """Open processed folder in system file manager"""
        try:
            if os.path.exists(self.processed_path):
                QDesktopServices.openUrl(f"file://{self.processed_path}")
                self.log_activity("Opened processed folder")
            else:
                QMessageBox.information(self, "Info", "Processed folder does not exist yet.")
        except Exception as e:
            logger.error(f"Failed to open folder: {e}")
            
    def clear_outputs(self):
        """Clear processing outputs (with confirmation)"""
        if not self.has_processed_folder:
            return
            
        reply = QMessageBox.question(
            self,
            "Clear Outputs",
            "This will delete all processing output files.\nAre you sure?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                # Define output file patterns to clear
                output_patterns = ['*.png', '*.jpg', '*.tiff', '*.csv', '*.txt', '*.dat']
                cleared_count = 0
                
                for root, dirs, files in os.walk(self.processed_path):
                    for file in files:
                        for pattern in output_patterns:
                            if file.endswith(pattern.replace('*', '')):
                                file_path = os.path.join(root, file)
                                try:
                                    os.remove(file_path)
                                    cleared_count += 1
                                except:
                                    pass
                                    
                self.log_activity(f"Cleared {cleared_count} output files")
                self.refresh_view()
                
            except Exception as e:
                logger.error(f"Failed to clear outputs: {e}")
                QMessageBox.critical(self, "Error", f"Failed to clear outputs:\n{str(e)}")
                
    def log_activity(self, message: str):
        """Log activity to the activity tab"""
        from datetime import datetime
        from qtpy.QtGui import QTextCursor
        timestamp = datetime.now().strftime("%H:%M:%S")
        activity_text = f"[{timestamp}] {message}\n"
        
        # Append to activity log
        current_text = self.activity_log.toPlainText()
        lines = current_text.split('\n')
        
        # Keep only last 50 lines
        if len(lines) > 50:
            lines = lines[-49:]
            
        lines.append(activity_text.strip())
        self.activity_log.setPlainText('\n'.join(lines))
        
        # Scroll to bottom
        cursor = self.activity_log.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.activity_log.setTextCursor(cursor)
        
    def set_project_root(self, project_root: str):
        """Update the project root path"""
        if project_root and os.path.exists(project_root):
            self.project_root = project_root
            self.processed_path = os.path.join(project_root, "processed")
            self.has_processed_folder = False
            
            # Save project path to settings (sync with Project Explorer)
            self.settings.setValue("last_project_path", project_root)
            
            # Reset to placeholder view and check for processed folder
            self.stacked_widget.setCurrentWidget(self.placeholder_widget)
            self.path_label.setText("Waiting for processed data...")
            self.info_label.setText("No processed data")
            
            # Disable toolbar buttons
            self.open_folder_btn.setEnabled(False)
            self.clear_btn.setEnabled(False)
            
            # Check if processed folder exists in new project
            self.check_processed_folder()
            
            if self.has_processed_folder:
                self.log_activity(f"Switched to project: {os.path.basename(project_root)}")
            else:
                logger.info(f"Switched to project: {os.path.basename(project_root)} (no processed folder yet)")
        else:
            logger.warning(f"Invalid project root path: {project_root}")
            
    def restore_last_project(self):
        """Restore the last opened project from settings"""
        last_project_path = self.settings.value("last_project_path")
        
        if last_project_path and os.path.exists(last_project_path):
            logger.info(f"Lab widget restoring last project: {last_project_path}")
            # Don't call set_project_root to avoid saving again, just update directly
            self.project_root = last_project_path
            self.processed_path = os.path.join(last_project_path, "processed")
            self.check_processed_folder()
            
            if self.has_processed_folder:
                self.log_activity(f"Restored project: {os.path.basename(last_project_path)}")
        else:
            if last_project_path:
                logger.warning(f"Lab widget: last project path no longer exists: {last_project_path}")
                # Clean up invalid path from settings
                self.settings.remove("last_project_path")
            # Keep placeholder view for invalid/missing projects
            logger.info("Lab widget: showing placeholder - no valid project to restore")
    
    def process_file(self, file_path: str):
        """Process the selected file using the processing dialog"""
        try:
            # Load the file as a DataFrame
            df = self.load_dataframe(file_path)
            if df is None:
                raise ValueError("Unsupported file type for processing")
                
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load file for processing: {str(e)}")
            return
            
        dialog = ProcessingDialog(df, self, input_file_path=file_path, project_manager=self.project_manager)
        
        # Forward layer creation signals to enable map integration
        dialog.layer_created.connect(self.layer_created.emit)
        # Forward plot generation signals to enable data viewer integration  
        dialog.plot_generated.connect(self.plot_generated.emit)
        # Forward processing completion signals for history viewer refresh
        dialog.processing_complete.connect(self.processing_complete.emit)
        
        if dialog.exec() == QDialog.Accepted:
            result = dialog.get_result()
            if result and result.success:
                message = "Processing completed successfully!"
                if result.output_file_path:
                    message += f"\nOutput saved to: {result.output_file_path}"
                QMessageBox.information(self, "Processing Complete", message)
                self.log_activity(f"Processed file: {os.path.basename(file_path)}")
                self.refresh_view()  # Refresh to show new output files
            else:
                error_msg = result.error_message if result else "Unknown error"
                QMessageBox.critical(self, "Processing Failed", f"Processing failed: {error_msg}")
                self.log_activity(f"Processing failed for: {os.path.basename(file_path)}")
    
    def load_dataframe(self, file_path: str) -> pd.DataFrame:
        """Load a file as a pandas DataFrame"""
        try:
            if file_path.endswith('.csv'):
                return pd.read_csv(file_path)
            elif file_path.endswith(('.txt', '.dat')):
                # Try different separators
                for sep in [',', '\t', ' ', ';']:
                    try:
                        df = pd.read_csv(file_path, sep=sep)
                        if len(df.columns) > 1:
                            return df
                    except:
                        continue
                # Fallback to default
                return pd.read_csv(file_path)
            elif file_path.endswith('.json'):
                return pd.read_json(file_path)
            elif file_path.endswith('.npz'):
                try:
                    with np.load(file_path) as npz_file:
                        main_array_key = max(npz_file.files, key=lambda k: npz_file[k].size)
                        df = pd.DataFrame(npz_file[main_array_key])
                        return df
                except Exception as e:
                    logger.error(f"Failed to load NPZ file {file_path}: {e}")
                    raise ValueError(f"Could not read .npz file: {e}")
            else:
                return None
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            return None 