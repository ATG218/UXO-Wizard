"""
Project Explorer Widget for UXO Wizard
"""

from PySide6.QtWidgets import (
    QTreeView, QVBoxLayout, QWidget, QToolButton, 
    QHBoxLayout, QMenu, QFileSystemModel, QHeaderView,
    QStackedWidget, QLabel, QPushButton,
    QInputDialog, QMessageBox, QFormLayout, QDialog, QDialogButtonBox
)
from PySide6.QtCore import Qt, Signal, QDir, QModelIndex, QSettings, QFileInfo, QItemSelectionModel, QSortFilterProxyModel
from PySide6.QtGui import QIcon, QAction, QFont
from loguru import logger
import os
import shutil
import pandas as pd
from .processing.processing_dialog import ProcessingDialog
from .processing.processing_widget import ProcessingWidget
import numpy as np


class ProjectExplorerProxyModel(QSortFilterProxyModel):
    """Proxy model that filters out internal project directories"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.hidden_dirs = {'_project'}  # Directories to hide
    
    def filterAcceptsRow(self, source_row, source_parent):
        """Override to filter out hidden directories"""
        # Get the source model
        source_model = self.sourceModel()
        if not source_model:
            return True
            
        # Get the index and file info for this row
        index = source_model.index(source_row, 0, source_parent)
        if not index.isValid():
            return True
            
        file_info = source_model.fileInfo(index)
        
        # Hide _project directory
        if file_info.isDir() and file_info.fileName() in self.hidden_dirs:
            return False
            
        return True


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
    layer_created = Signal(object)  # Emits UXOLayer for map integration
    plot_generated = Signal(object, str)  # Figure, title - for opening plots in data viewer
    
    def __init__(self, project_manager=None, auto_restore=True):
        super().__init__()
        self.clipboard_paths = []
        self.is_cut = False
        self.clicked_path = None
        self.current_project_path = None
        self.project_manager = project_manager
        self.settings = QSettings("UXO-Wizard", "Desktop-Suite")
        self.setup_ui()
        
        # Restore last project after UI is set up (unless disabled)
        if auto_restore:
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
        self.tree_view.setSelectionMode(QTreeView.ExtendedSelection)
        self.tree_view.setDragEnabled(True)
        self.tree_view.setAcceptDrops(True)
        self.tree_view.setDropIndicatorShown(True)
        self.tree_view.setAlternatingRowColors(True)
        self.tree_view.setAnimated(True)
        self.tree_view.setSortingEnabled(True)
        
        # File system model with proxy for filtering
        self.source_model = QFileSystemModel()
        self.source_model.setRootPath(QDir.rootPath())
        
        # Set filters for relevant file types
        self.source_model.setNameFilters([
            "*.csv", "*.txt", "*.dat", "*.xlsx", "*.xls",
            "*.json", "*.geojson", "*.shp", "*.tif", "*.tiff",
            "*.png", "*.jpg", "*.jpeg", "*.uxo", "*.mplplot", # UXO project and plot files
            "*.npz", "*.sgy", "*.SGY", "*.log" # GPR data, output and log files
        ])
        self.source_model.setNameFilterDisables(False)  # Hide non-matching files
        
        # Proxy model to filter out _project directory
        self.model = ProjectExplorerProxyModel()
        self.model.setSourceModel(self.source_model)
        
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
            # Get index from source model first, then map to proxy
            source_index = self.source_model.index(path)
            proxy_index = self.model.mapFromSource(source_index)
            self.tree_view.setRootIndex(proxy_index)
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
            # Map proxy index back to source model to get file path
            source_index = self.model.mapToSource(indexes[0])
            return self.source_model.filePath(source_index)
        return None
        
    def handle_double_click(self, index: QModelIndex):
        """Handle double-click on file"""
        source_index = self.model.mapToSource(index)
        if not self.source_model.isDir(source_index):
            file_path = self.source_model.filePath(source_index)
            self.file_selected.emit(file_path)
            logger.info(f"File double-clicked: {file_path}")
            
    def handle_click(self, index: QModelIndex):
        """Handle single click on item"""
        source_index = self.model.mapToSource(index)
        file_path = self.source_model.filePath(source_index)
        logger.debug(f"Item clicked: {file_path}")
        
    def show_context_menu(self, position):
        """Show context menu for selected item"""
        clicked_index = self.tree_view.indexAt(position)
        if not clicked_index.isValid():
            self.clicked_path = self.current_project_path
        else:
            source_index = self.model.mapToSource(clicked_index)
            self.clicked_path = self.source_model.filePath(source_index)
        
        indexes = self.tree_view.selectedIndexes()
        if not indexes and clicked_index.isValid():
            self.tree_view.selectionModel().select(clicked_index, QItemSelectionModel.Select)
            indexes = self.tree_view.selectedIndexes()
        
        if not indexes:
            return
        
        selected_indexes = [idx for idx in indexes if idx.column() == 0]
        if not selected_indexes:
            return
        
        selected_paths = [self.source_model.filePath(self.model.mapToSource(idx)) for idx in selected_indexes]
        is_dirs = [self.source_model.isDir(self.model.mapToSource(idx)) for idx in selected_indexes]
        all_dirs = all(is_dirs)
        
        menu = QMenu()
        
        if not all_dirs:
            open_action = QAction("Open", self)
            open_action.triggered.connect(lambda: [self.file_selected.emit(p) for p in selected_paths if not os.path.isdir(p)])
            menu.addAction(open_action)
            
            process_action = QAction("Process...", self)
            process_action.triggered.connect(self.process_selected)
            if len(selected_paths) != 1 or os.path.isdir(selected_paths[0]):
                process_action.setEnabled(False)
            menu.addAction(process_action)
            
            # NEW: Add "Run Previous" option if available
            last_used = ProcessingWidget.get_last_used_script()
            
            if last_used and len(selected_paths) == 1 and not os.path.isdir(selected_paths[0]):
                script_name = last_used["script"]
                processor_type = last_used["processor_type"].title()
                
                run_previous_action = QAction(f"Run Previous ({script_name})", self)
                run_previous_action.triggered.connect(lambda: self.run_previous_processing(selected_paths[0]))
                run_previous_action.setToolTip(f"Re-run {processor_type} processor with {script_name} using last parameters")
                menu.addAction(run_previous_action)
            
            menu.addSeparator()
        
        if all_dirs and len(selected_paths) == 1:
            set_root_action = QAction("Set as Root", self)
            set_root_action.triggered.connect(lambda: self.set_root_path(selected_paths[0]))
            menu.addAction(set_root_action)
        
        new_folder_action = QAction("New Folder", self)
        new_folder_action.triggered.connect(self.create_new_folder)
        menu.addAction(new_folder_action)
        
        menu.addSeparator()
        
        copy_action = QAction("Copy", self)
        copy_action.triggered.connect(self.copy_items)
        menu.addAction(copy_action)
        
        cut_action = QAction("Cut", self)
        cut_action.triggered.connect(self.cut_items)
        menu.addAction(cut_action)
        
        if self.clipboard_paths:
            paste_action = QAction("Paste", self)
            paste_action.triggered.connect(self.paste_items)
            menu.addAction(paste_action)
        
        menu.addSeparator()
        
        rename_action = QAction("Rename", self)
        rename_action.triggered.connect(self.rename_item)
        if len(selected_paths) != 1:
            rename_action.setEnabled(False)
        menu.addAction(rename_action)
        
        delete_action = QAction("Delete", self)
        delete_action.triggered.connect(self.delete_item)
        menu.addAction(delete_action)
        
        menu.addSeparator()
        
        properties_action = QAction("Properties", self)
        properties_action.triggered.connect(self.show_properties)
        if len(selected_paths) != 1:
            properties_action.setEnabled(False)
        menu.addAction(properties_action)
        
        menu.exec_(self.tree_view.viewport().mapToGlobal(position))
        
    def refresh_view(self):
        """Refresh the file view"""
        if self.current_project_path:
            # Force source model to update
            root = self.source_model.rootPath()
            self.source_model.setRootPath("")
            self.source_model.setRootPath(root)
            logger.info("Project explorer refreshed")
        
    def collapse_all(self):
        """Collapse all tree items"""
        self.tree_view.collapseAll()
        
    def expand_all(self):
        """Expand all tree items"""
        self.tree_view.expandAll()
        
    def set_name_filters(self, filters):
        """Update the file name filters"""
        self.source_model.setNameFilters(filters)
        
    def show_hidden_files(self, show):
        """Toggle showing hidden files"""
        if show:
            self.source_model.setFilter(self.source_model.filter() | QDir.Hidden)
        else:
            self.source_model.setFilter(self.source_model.filter() & ~QDir.Hidden) 
    
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

    def get_selected_paths(self):
        """Get list of unique selected paths"""
        indexes = self.tree_view.selectedIndexes()
        paths = set()
        for index in indexes:
            if index.column() == 0:
                source_index = self.model.mapToSource(index)
                paths.add(self.source_model.filePath(source_index))
        return list(paths)
    
    def create_new_folder(self):
        """Create a new folder in the clicked or root directory"""
        parent_dir = self.current_project_path
        if self.clicked_path and os.path.isdir(self.clicked_path):
            parent_dir = self.clicked_path
        name, ok = QInputDialog.getText(self, "New Folder", "Folder name:")
        if ok and name:
            new_path = os.path.join(parent_dir, name)
            try:
                os.mkdir(new_path)
                logger.info(f"Created folder: {new_path}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to create folder: {str(e)}")
    
    def rename_item(self):
        """Rename the selected item (single selection only)"""
        selected = self.get_selected_paths()
        if len(selected) != 1:
            return
        path = selected[0]
        base = os.path.basename(path)
        name, ok = QInputDialog.getText(self, "Rename", "New name:", text=base)
        if ok and name:
            new_path = os.path.join(os.path.dirname(path), name)
            try:
                os.rename(path, new_path)
                logger.info(f"Renamed {path} to {new_path}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to rename: {str(e)}")
    
    def delete_item(self):
        """Delete selected items"""
        selected = self.get_selected_paths()
        if not selected:
            return
        msg = f"Are you sure you want to delete {len(selected)} item(s)?"
        if QMessageBox.question(self, "Confirm Delete", msg) != QMessageBox.Yes:
            return
        for path in selected:
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
                logger.info(f"Deleted: {path}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to delete {path}: {str(e)}")
    
    def show_properties(self):
        """Show properties for single selected item"""
        selected = self.get_selected_paths()
        if len(selected) != 1:
            return
        path = selected[0]
        info = QFileInfo(path)
        dialog = QDialog(self)
        dialog.setWindowTitle("Properties")
        layout = QFormLayout()
        layout.addRow("Name:", QLabel(info.fileName()))
        layout.addRow("Path:", QLabel(info.absoluteFilePath()))
        layout.addRow("Type:", QLabel("Folder" if info.isDir() else "File"))
        layout.addRow("Size:", QLabel(f"{info.size()} bytes"))
        layout.addRow("Modified:", QLabel(info.lastModified().toString()))
        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(dialog.accept)
        layout.addRow(buttons)
        dialog.setLayout(layout)
        dialog.exec_()
    
    def copy_items(self):
        """Copy selected paths to clipboard"""
        self.clipboard_paths = self.get_selected_paths()
        self.is_cut = False
        logger.info(f"Copied {len(self.clipboard_paths)} items")
    
    def cut_items(self):
        """Cut selected paths to clipboard"""
        self.clipboard_paths = self.get_selected_paths()
        self.is_cut = True
        logger.info(f"Cut {len(self.clipboard_paths)} items")
    
    def paste_items(self):
        """Paste clipboard items to clicked or root directory"""
        if not self.clipboard_paths:
            return
        target_dir = self.current_project_path
        if self.clicked_path:
            if os.path.isdir(self.clicked_path):
                target_dir = self.clicked_path
            else:
                target_dir = os.path.dirname(self.clicked_path)
        for src in self.clipboard_paths:
            dest = os.path.join(target_dir, os.path.basename(src))
            try:
                if self.is_cut:
                    shutil.move(src, dest)
                else:
                    if os.path.isdir(src):
                        shutil.copytree(src, dest)
                    else:
                        shutil.copy(src, dest)
                logger.info(f"Pasted {src} to {dest}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to paste {src}: {str(e)}")
        if self.is_cut:
            self.clipboard_paths = []

    def _detect_csv_delimiter(self, filepath, encoding='utf-8', data_start_line=0):
        """Simple delimiter detection by counting occurrences"""
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                lines = f.readlines()
            
            if data_start_line > 0 and data_start_line < len(lines):
                sample_lines = lines[data_start_line:data_start_line + 5]
            else:
                sample_lines = lines[:10]
            
            delimiter_counts = {';': 0, ',': 0, '\t': 0, '|': 0}
            
            for line in sample_lines:
                for delimiter in delimiter_counts:
                    delimiter_counts[delimiter] += line.count(delimiter)
            
            best_delimiter = max(delimiter_counts, key=delimiter_counts.get)
            if delimiter_counts[best_delimiter] > 0:
                logger.debug(f"Detected delimiter '{best_delimiter}'")
                return best_delimiter
            
            return ';'
        except Exception as e:
            logger.warning(f"Error detecting delimiter: {e}")
            return ';'

    def _parse_file_header(self, filepath, encoding='utf-8'):
        """Parse multi-line header and find where tabular data starts"""
        metadata = {}
        data_start_line = 0
        header_line = None
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                lines = f.readlines()
            
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                delimiter_count = max(line.count(';'), line.count(','), line.count('\t'), line.count('|'))
                if delimiter_count >= 3:
                    delim = max([';', ',', '\t', '|'], key=line.count)
                    tokens = [t.strip().lower() for t in line.split(delim)]
                    if tokens and 'timestamp' in tokens[0]:
                        header_line = i
                        data_start_line = i + 1
                        break
            return metadata, data_start_line, header_line
        except Exception as e:
            logger.warning(f"Error parsing header: {e}")
            return {}, 0, None

    def _load_csv_enhanced(self, filepath):
        """Load CSV with smart parsing"""
        encoding = 'utf-8'
        metadata, data_start_line, header_line = self._parse_file_header(filepath, encoding)
        delimiter = self._detect_csv_delimiter(filepath, encoding, data_start_line)
        if header_line is not None and header_line >= 0:
            skip_rows = header_line
        else:
            skip_rows = data_start_line if data_start_line > 0 else None
        if skip_rows == 0:
            skip_rows = None
        try:
            df = pd.read_csv(filepath, delimiter=delimiter, encoding=encoding, skiprows=skip_rows, header=0)
            df.columns = df.columns.str.strip()
            if hasattr(df, 'attrs'):
                df.attrs['file_metadata'] = metadata
                df.attrs['source_file'] = filepath
            return df
        except Exception as e:
            logger.warning(f"Primary load failed: {e}")
            fallback_options = [(';','latin-1'), (',','utf-8'), (',','latin-1')]
            for delim, enc in fallback_options:
                try:
                    df = pd.read_csv(filepath, delimiter=delim, encoding=enc, skiprows=skip_rows, header=0)
                    df.columns = df.columns.str.strip()
                    if hasattr(df, 'attrs'):
                        df.attrs['file_metadata'] = metadata
                        df.attrs['source_file'] = filepath
                    return df
                except:
                    continue
            raise Exception("Failed to load CSV")

    def load_dataframe(self, filepath):
        """Load dataframe from file"""
        file_lower = filepath.lower()
        if file_lower.endswith('.csv'):
            return self._load_csv_enhanced(filepath)
        elif file_lower.endswith(('.xlsx', '.xls')):
            return pd.read_excel(filepath)
        elif file_lower.endswith('.json'):
            return pd.read_json(filepath)
        elif file_lower.endswith('.npz'):
            try:
                with np.load(filepath) as npz_file:
                    # Assume the largest array is the main data
                    main_array_key = max(npz_file.files, key=lambda k: npz_file[k].size)
                    df = pd.DataFrame(npz_file[main_array_key])
                    # Add other arrays as metadata if needed
                    if hasattr(df, 'attrs'):
                        df.attrs['source_file'] = filepath
                        df.attrs['npz_arrays'] = npz_file.files
                    return df
            except Exception as e:
                logger.error(f"Failed to load NPZ file {filepath}: {e}")
                raise ValueError(f"Could not read .npz file: {e}")
        else:
            return None

    def process_selected(self):
        """Process the selected file"""
        selected = self.get_selected_paths()
        if len(selected) != 1:
            QMessageBox.warning(self, "Selection Error", "Please select exactly one file to process.")
            return
        path = selected[0]
        if os.path.isdir(path):
            QMessageBox.warning(self, "Selection Error", "Cannot process directories.")
            return
        try:
            df = None
            # For file types that are not directly loaded as dataframes (like .npz),
            # we can pass an empty dataframe and rely on the script to handle the file path.
            if path.lower().endswith('.npz'):
                df = pd.DataFrame()
            else:
                df = self.load_dataframe(path)

            if df is None:
                raise ValueError("Unsupported file type for processing")
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load file for processing: {str(e)}")
            return
        dialog = ProcessingDialog(df, self, input_file_path=path, project_manager=self.project_manager)
        
        # Forward layer creation signals to enable map integration
        dialog.layer_created.connect(self.layer_created.emit)
        # Forward plot generation signals to enable data viewer integration
        dialog.plot_generated.connect(self.plot_generated.emit)
        
        if dialog.exec() == QDialog.Accepted:
            result = dialog.get_result()
            if result and result.success:
                message = "Processing completed successfully!"
                if result.output_file_path:
                    message += f"\nOutput saved to: {result.output_file_path}"
                QMessageBox.information(self, "Success", message)
    
    def run_previous_processing(self, file_path):
        """Run the last used processing script with saved parameters"""
        last_used = ProcessingWidget.get_last_used_script()
        if not last_used:
            QMessageBox.warning(self, "No Previous Script", "No previous processing script found.")
            return
        
        try:
            # Load the data file
            df = self.load_dataframe(file_path)
            if df is None:
                raise ValueError("Unsupported file type for processing")
                
            # Debug: Print data columns to understand what we have
            logger.info(f"Loaded data columns: {list(df.columns)}")
            logger.info(f"Data shape: {df.shape}")
            logger.info(f"First few rows:\n{df.head()}")
            
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load file for processing: {str(e)}")
            return
        
        try:
            # Create processing dialog
            dialog = ProcessingDialog(df, self, input_file_path=file_path, project_manager=self.project_manager)
            
            # Set up the processor and script from last used
            processor_type = last_used["processor_type"]
            script_name = last_used["script"] 
            parameters = last_used["parameters"]
            
            logger.info(f"Running previous processing:")
            logger.info(f"  Processor type: {processor_type}")
            logger.info(f"  Script name: {script_name}")
            logger.info(f"  Parameters keys: {list(parameters.keys()) if parameters else 'None'}")
            
            # Auto-configure the dialog
            if not dialog.configure_from_previous(processor_type, script_name, parameters):
                QMessageBox.critical(self, "Configuration Failed", "Failed to configure the processing dialog")
                return
            
            # Forward layer creation and plot signals
            dialog.layer_created.connect(self.layer_created.emit)
            dialog.plot_generated.connect(self.plot_generated.emit)
            
            # Show the dialog instead of starting silently to allow user to see what's happening
            if dialog.exec() == QDialog.Accepted:
                result = dialog.get_result()
                if result and result.success:
                    message = "Processing completed successfully!"
                    if result.output_file_path:
                        message += f"\nOutput saved to: {result.output_file_path}"
                    QMessageBox.information(self, "Success", message)
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to run previous processing: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")