"""
Data Viewer Widget for UXO Wizard - Display and analyze tabular data
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QTableView, QToolBar, 
    QComboBox, QToolButton, QLabel, QLineEdit,
    QHeaderView, QMenu, QMessageBox, QDialog, QDialogButtonBox,
    QTabWidget, QHBoxLayout, QPushButton, QFileDialog, QSizePolicy
)
from PySide6.QtGui import QAction, QIcon
from PySide6.QtCore import Qt, Signal, QAbstractTableModel, QModelIndex
from PySide6.QtGui import QKeySequence
import pandas as pd
import numpy as np
from loguru import logger
from typing import Optional, Tuple
import os

# Import map layer types for integration
try:
    from ..map.layer_types import UXOLayer, LayerType, GeometryType, LayerStyle, LayerSource
    MAP_INTEGRATION_AVAILABLE = True
except ImportError:
    logger.warning("Map integration not available - layer_types module not found")
    MAP_INTEGRATION_AVAILABLE = False


class PandasModel(QAbstractTableModel):
    """Model for displaying pandas DataFrame in QTableView"""
    
    def __init__(self, data=None):
        super().__init__()
        self._data = data if data is not None else pd.DataFrame()
        
    def rowCount(self, parent=QModelIndex()):
        return len(self._data)
        
    def columnCount(self, parent=QModelIndex()):
        return len(self._data.columns)
        
    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
            
        if role == Qt.DisplayRole:
            value = self._data.iloc[index.row(), index.column()]
            
            # Format display based on data type
            if pd.isna(value):
                return "NaN"
            elif isinstance(value, (np.float64, np.float32, float)):
                return f"{value:.6f}"
            else:
                return str(value)
                
        elif role == Qt.TextAlignmentRole:
            value = self._data.iloc[index.row(), index.column()]
            if isinstance(value, (int, float, np.number)):
                return Qt.AlignRight | Qt.AlignVCenter
            else:
                return Qt.AlignLeft | Qt.AlignVCenter
                
        return None
        
    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._data.columns[section])
            else:
                return str(self._data.index[section])
        return None
        
    def setData(self, data):
        """Update the data"""
        self.beginResetModel()
        self._data = data
        self.endResetModel()
        
    def get_dataframe(self):
        """Get the underlying DataFrame"""
        return self._data.copy()


class DataViewerTab(QWidget):
    """Individual tab for viewing and analyzing a single dataset"""
    
    # Signals
    data_selected = Signal(object)  # Emits selected data
    layer_created = Signal(object)  # Emits UXOLayer for map integration
    tab_title_changed = Signal(str)  # Emits when tab title should change
    
    def __init__(self, filepath=None):
        super().__init__()
        self.current_file = filepath
        self.setup_ui()
        
        # Load data if filepath provided
        if filepath:
            self.load_data(filepath)
        
    def setup_ui(self):
        """Initialize the UI for this tab"""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Toolbar
        toolbar = QToolBar()
        
        # Info label
        self.info_label = QLabel("No data loaded")
        toolbar.addWidget(self.info_label)
        
        toolbar.addSeparator()
        
        # Column filter
        self.column_combo = QComboBox()
        self.column_combo.setMinimumWidth(150)
        self.column_combo.currentTextChanged.connect(self.filter_columns)
        toolbar.addWidget(QLabel("Column:"))
        toolbar.addWidget(self.column_combo)
        
        # Search
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search...")
        self.search_edit.textChanged.connect(self.search_data)
        toolbar.addWidget(self.search_edit)
        
        toolbar.addSeparator()
        
        # Statistics button
        self.stats_btn = QToolButton()
        self.stats_btn.setText("Stats")
        self.stats_btn.clicked.connect(self.show_statistics)
        toolbar.addWidget(self.stats_btn)
        
        # Metadata button  
        self.metadata_btn = QToolButton()
        self.metadata_btn.setText("Metadata")
        self.metadata_btn.setToolTip("Show file metadata and header information")
        self.metadata_btn.clicked.connect(self.show_metadata)
        self.metadata_btn.setEnabled(False)  # Disabled until data with metadata is loaded
        toolbar.addWidget(self.metadata_btn)
        
        # Export button
        self.export_btn = QToolButton()
        self.export_btn.setText("Export")
        self.export_btn.clicked.connect(self.export_data)
        toolbar.addWidget(self.export_btn)
        
        toolbar.addSeparator()
        
        # Processing button
        self.process_btn = QToolButton()
        self.process_btn.setText("⚡ Process")
        self.process_btn.setToolTip("Open Processing Menu")
        self.process_btn.clicked.connect(self.show_processing_menu)
        toolbar.addWidget(self.process_btn)
        
        # Plot on Map button (only show if map integration available)
        if MAP_INTEGRATION_AVAILABLE:
            self.plot_map_btn = QToolButton()
            self.plot_map_btn.setText("📍 Plot on Map")
            self.plot_map_btn.setToolTip("Visualize this table as points on the map (requires coordinate columns)")
            self.plot_map_btn.clicked.connect(self.plot_on_map)
            toolbar.addWidget(self.plot_map_btn)
        
        # Table view
        self.table_view = QTableView()
        self.table_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.table_view.setSortingEnabled(True)
        self.table_view.setAlternatingRowColors(True)
        self.table_view.setSelectionBehavior(QTableView.SelectRows)
        
        # Model
        self.model = PandasModel()
        self.table_view.setModel(self.model)
        
        # Context menu
        self.table_view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table_view.customContextMenuRequested.connect(self.show_context_menu)
        
        # Layout
        layout.addWidget(toolbar)
        layout.addWidget(self.table_view)
        self.setLayout(layout)
        
    def _detect_csv_delimiter(self, filepath, encoding='utf-8', data_start_line=0):
        """Simple delimiter detection by counting occurrences"""
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                lines = f.readlines()
                
            # Get sample from data section
            if data_start_line > 0 and data_start_line < len(lines):
                sample_lines = lines[data_start_line:data_start_line + 5]  # Just 5 lines
            else:
                sample_lines = lines[:10]  # First 10 lines
                
            # Count delimiters - no csv.Sniffer needed
            delimiter_counts = {';': 0, ',': 0, '\t': 0, '|': 0}
            
            for line in sample_lines:
                for delimiter in delimiter_counts:
                    delimiter_counts[delimiter] += line.count(delimiter)
            
            # Return the most common delimiter
            best_delimiter = max(delimiter_counts, key=delimiter_counts.get)
            if delimiter_counts[best_delimiter] > 0:
                logger.debug(f"Detected delimiter '{best_delimiter}'")
                return best_delimiter
            
            # Default to semicolon (common for scientific data)
            return ';'
                
        except Exception as e:
            logger.warning(f"Error detecting delimiter: {e}")
            return ';'
    
    def _clean_trailing_delimiters(self, df):
        """Remove empty columns caused by trailing delimiters"""
        try:
            # Find columns that are completely empty or contain only NaN/empty strings
            empty_cols = []
            for col in df.columns:
                if (df[col].isna().all() or 
                    (df[col].astype(str).str.strip() == '').all() or
                    col.startswith('Unnamed:')):
                    # Check if this column has any actual data
                    non_empty_count = df[col].dropna().astype(str).str.strip().str.len().sum()
                    if non_empty_count == 0:
                        empty_cols.append(col)
            
            if empty_cols:
                logger.debug(f"Removing {len(empty_cols)} empty columns caused by trailing delimiters: {empty_cols}")
                df = df.drop(columns=empty_cols)
                
            return df
            
        except Exception as e:
            logger.warning(f"Error cleaning trailing delimiters: {e}")
            return df
    
    def _parse_file_header(self, filepath, encoding='utf-8'):
        """Parse multi-line header and find where tabular data starts"""
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                lines = f.readlines()
            
            metadata = {}
            data_start_line = 0
            header_line = None
            
            # Look for patterns that indicate data vs metadata
            for i, line in enumerate(lines):
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    continue
                    
                # Check if this line looks like a data header (contains multiple delimiters and
                # has "timestamp" as the first token – a much stronger indicator for the start
                # of tabular data than the earlier fuzzy approach that triggered on any "time"
                # occurrence and mis-classified explanatory lines such as the "GGA Quality" row).

                delimiter_count = max(line.count(';'), line.count(','), line.count('\t'), line.count('|'))

                if delimiter_count >= 3:
                    # Split by the most common delimiter in this line and inspect the first token.
                    # We choose the delimiter that appears most frequently in the current line to
                    # make the split (works even before global delimiter detection has run).
                    delim = max([';', ',', '\t', '|'], key=line.count)
                    tokens = [t.strip().lower() for t in line.split(delim)]

                    if tokens and 'timestamp' in tokens[0]:
                        header_line = i
                        data_start_line = i + 1
                        logger.debug(
                            f"Found data header at line {i + 1}: {line[:100]}..."
                        )
                        break
                
                # Parse metadata from header section
                if ':' in line and '=' not in line:
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip()
                        metadata[key] = value
                elif '=' in line and '[' in line and ']' in line:
                    # Handle lines like "Digital Probe Voltage: +3.64 [V]"
                    if ':' in line:
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            key = parts[0].strip()
                            value = parts[1].strip()
                            metadata[key] = value
            
            # If no clear header found, try to detect first line that looks like data
            if data_start_line == 0:
                for i, line in enumerate(lines):
                    line = line.strip()
                    if not line:
                        continue
                        
                    # Check if line has consistent delimiter pattern (likely data)
                    for delim in [';', ',', '\t', '|']:
                        if line.count(delim) >= 5:  # At least 5 delimiters suggests tabular data
                            # Check if values look numeric (basic heuristic)
                            parts = line.split(delim)
                            numeric_count = 0
                            for part in parts[:10]:  # Check first 10 parts
                                part = part.strip()
                                try:
                                    float(part)
                                    numeric_count += 1
                                except ValueError:
                                    pass
                            
                            if numeric_count >= len(parts) * 0.6:  # 60% numeric suggests data row
                                data_start_line = i
                                logger.debug(f"Detected data start at line {i+1} based on numeric content")
                                break
                    
                    if data_start_line > 0:
                        break
            
            return metadata, data_start_line, header_line
            
        except Exception as e:
            logger.warning(f"Error parsing file header: {e}")
            return {}, 0, None
    
    def _load_csv_enhanced(self, filepath):
        """Load CSV file with smart header parsing and delimiter detection"""
        # Try UTF-8 first (works for most modern files including ASCII)
        encoding = 'utf-8'
        
        # Parse header to find metadata and data start
        metadata, data_start_line, header_line = self._parse_file_header(filepath, encoding)
        
        if metadata:
            logger.info(f"Found {len(metadata)} metadata entries in file header")
        
        # Detect delimiter from the data section
        delimiter = self._detect_csv_delimiter(filepath, encoding, data_start_line)
        
        # ------------------------------------------------------------------
        # Determine how many lines to skip so that the *detected* header line
        # is preserved and used as the DataFrame header. The previous logic
        # skipped `data_start_line` which accidentally discarded the header
        # row and caused the first data row to be interpreted as the header
        # when metadata lines were present (e.g. in mag.csv).  We now skip
        # the lines **before** the detected header instead.
        # ------------------------------------------------------------------

        if header_line is not None and header_line >= 0:
            # Skip everything *before* the header line – pandas will then use
            # the first remaining line (the actual header) for column names.
            skip_rows = header_line
        else:
            # Fallback to previous behaviour when no explicit header line was
            # found (e.g. files that start immediately with column names).
            skip_rows = data_start_line if data_start_line > 0 else None
        
        # Using skiprows=0 is equivalent to not passing the argument at all,
        # but some pandas versions raise on integer 0, so normalise here.
        if skip_rows == 0:
            skip_rows = None
        
        try:
            df = pd.read_csv(
                filepath,
                delimiter=delimiter,
                encoding=encoding,
                skiprows=skip_rows,
                header=0  # first non-skipped line now safely holds the column names
            )
            
            # Clean up trailing delimiters
            df = self._clean_trailing_delimiters(df)

            # ------------------------------------------------------------------
            # Normalise column names: remove leading/trailing whitespace that can
            # occur when the delimiter is directly followed by a space (common
            # in semicolon-separated logger files). This avoids columns showing
            # up with awkward spaces like " B1x [nT]" in the UI.
            # ------------------------------------------------------------------
            df.columns = df.columns.str.strip()
            
            if hasattr(df, 'attrs'):
                df.attrs['file_metadata'] = metadata
                df.attrs['data_start_line'] = data_start_line
                df.attrs['source_file'] = filepath
            
            logger.info(f"Successfully loaded CSV: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
            
        except Exception as e:
            # Simple fallback: try semicolon delimiter and different encoding
            logger.warning(f"Primary load failed ({e}), trying fallback options")
            
            # Try different combinations
            fallback_options = [
                (';', 'latin-1'),
                (',', 'utf-8'), 
                (',', 'latin-1')
            ]
            
            for delim, enc in fallback_options:
                try:
                    df = pd.read_csv(
                        filepath,
                        delimiter=delim,
                        encoding=enc,
                        skiprows=skip_rows,
                        header=0
                    )
                    df = self._clean_trailing_delimiters(df)
                    
                    # Clean up trailing delimiters
                    df = self._clean_trailing_delimiters(df)

                    # ------------------------------------------------------------------
                    # Normalise column names: remove leading/trailing whitespace that can
                    # occur when the delimiter is directly followed by a space (common
                    # in semicolon-separated logger files). This avoids columns showing
                    # up with awkward spaces like " B1x [nT]" in the UI.
                    # ------------------------------------------------------------------
                    df.columns = df.columns.str.strip()
                    
                    if hasattr(df, 'attrs'):
                        df.attrs['file_metadata'] = metadata
                        df.attrs['data_start_line'] = data_start_line  
                        df.attrs['source_file'] = filepath
                    
                    logger.info(f"Loaded with fallback ({delim}, {enc}): {df.shape[0]} rows, {df.shape[1]} columns")
                    return df
                except Exception:
                    continue
                    
            raise Exception(f"Failed to load CSV file with all attempted methods")

    def load_data(self, filepath):
        """Load data from file"""
        try:
            # Determine file type and load accordingly
            if filepath.endswith('.csv'):
                df = self._load_csv_enhanced(filepath)
            elif filepath.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(filepath)
            elif filepath.endswith('.json'):
                df = pd.read_json(filepath)
            else:
                logger.error(f"Unsupported file type: {filepath}")
                return
                
            self.set_dataframe(df)
            self.current_file = filepath
            
            # Update tab title
            filename = os.path.basename(filepath)
            self.tab_title_changed.emit(filename)
            
            logger.info(f"Loaded data from: {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to load data:\n{str(e)}")
            
    def set_dataframe(self, df):
        """Set the DataFrame to display"""
        self.model.setData(df)
        
        # Check for metadata
        has_metadata = hasattr(df, 'attrs') and 'file_metadata' in df.attrs
        self.metadata_btn.setEnabled(has_metadata)
        
        # Update UI info label with basic info and key metadata
        info_text = f"Rows: {len(df)} | Columns: {len(df.columns)}"
        
        if has_metadata:
            metadata = df.attrs['file_metadata']
            # Add some key metadata to the info label
            key_info = []
            for key in ['Date', 'Time', 'Field-Nr.', 'MagWalk', 'Samples']:
                if key in metadata:
                    key_info.append(f"{key}: {metadata[key]}")
            
            if key_info:
                info_text += f" | {' | '.join(key_info[:2])}"  # Show first 2 metadata items
        
        self.info_label.setText(info_text)
        
        # Update column combo
        self.column_combo.clear()
        self.column_combo.addItem("All columns")
        self.column_combo.addItems(df.columns.tolist())
        
        # Improve header sizing for better fit in tabs
        header = self.table_view.horizontalHeader()
        
        # First, resize to contents
        self.table_view.resizeColumnsToContents()
        
        # Get available width and column count
        available_width = self.table_view.viewport().width()
        column_count = len(df.columns)
        
        if column_count > 0:
            # Calculate optimal column distribution
            total_content_width = sum(header.sectionSize(i) for i in range(column_count))
            
            if total_content_width > available_width:
                # Content is too wide - use stretch to fit all columns
                header.setSectionResizeMode(QHeaderView.Stretch)
            else:
                # Content fits - use interactive mode with some constraints
                header.setSectionResizeMode(QHeaderView.Interactive)
                
                # Set reasonable bounds: min 80px, max 250px
                for i in range(column_count):
                    current_size = header.sectionSize(i)
                    if current_size < 80:
                        header.resizeSection(i, 80)
                    elif current_size > 250:
                        header.resizeSection(i, 250)
                
                # Enable stretch on last section to fill remaining space
                header.setStretchLastSection(True)
                
    def filter_columns(self, column):
        """Filter displayed columns"""
        if column == "All columns" or not column:
            # Show all columns
            for i in range(self.model.columnCount()):
                self.table_view.showColumn(i)
        else:
            # Show only selected column
            df = self.model.get_dataframe()
            for i, col in enumerate(df.columns):
                if col == column:
                    self.table_view.showColumn(i)
                else:
                    self.table_view.hideColumn(i)
                    
    def search_data(self, text):
        """Search for text in data"""
        # TODO: Implement search functionality
        logger.debug(f"Searching for: {text}")
        
    def show_statistics(self):
        """Show statistics for the current data"""
        df = self.model.get_dataframe()
        if df.empty:
            return
            
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            QMessageBox.information(self, "Statistics", "No numeric columns found.")
            return
            
        # Calculate statistics
        stats_text = "Data Statistics:\n\n"
        
        for col in numeric_cols:
            stats_text += f"{col}:\n"
            stats_text += f"  Mean: {df[col].mean():.6f}\n"
            stats_text += f"  Std: {df[col].std():.6f}\n"
            stats_text += f"  Min: {df[col].min():.6f}\n"
            stats_text += f"  Max: {df[col].max():.6f}\n"
            stats_text += f"  Missing: {df[col].isna().sum()}\n\n"
            
        QMessageBox.information(self, "Statistics", stats_text)
        
    def show_metadata(self):
        """Show file metadata and header information"""
        df = self.model.get_dataframe()
        if df.empty or not hasattr(df, 'attrs') or 'file_metadata' not in df.attrs:
            QMessageBox.information(self, "Metadata", "No metadata available for this file.")
            return
            
        metadata = df.attrs['file_metadata']
        source_file = df.attrs.get('source_file', 'Unknown')
        data_start_line = df.attrs.get('data_start_line', 0)
        
        # Build metadata display text
        metadata_text = f"File: {source_file}\n"
        metadata_text += f"Data starts at line: {data_start_line + 1}\n"
        metadata_text += f"Total metadata entries: {len(metadata)}\n\n"
        
        # Group metadata by categories
        survey_info = {}
        probe_info = {}
        gps_info = {}
        other_info = {}
        
        for key, value in metadata.items():
            key_lower = key.lower()
            if any(term in key_lower for term in ['field', 'date', 'time', 'walk', 'sample']):
                survey_info[key] = value
            elif any(term in key_lower for term in ['probe', 'voltage', 'temperature', 'firmware']):
                probe_info[key] = value
            elif 'gps' in key_lower:
                gps_info[key] = value
            else:
                other_info[key] = value
        
        # Display organized metadata
        if survey_info:
            metadata_text += "=== Survey Information ===\n"
            for key, value in survey_info.items():
                metadata_text += f"{key}: {value}\n"
            metadata_text += "\n"
            
        if probe_info:
            metadata_text += "=== Probe Settings ===\n"
            for key, value in probe_info.items():
                metadata_text += f"{key}: {value}\n"
            metadata_text += "\n"
            
        if gps_info:
            metadata_text += "=== GPS Information ===\n"
            for key, value in gps_info.items():
                metadata_text += f"{key}: {value}\n"
            metadata_text += "\n"
            
        if other_info:
            metadata_text += "=== Other Information ===\n"
            for key, value in other_info.items():
                metadata_text += f"{key}: {value}\n"
        
        # Show in a scrollable dialog with content displayed directly
        dialog = QMessageBox(self)
        dialog.setWindowTitle("File Metadata")
        dialog.setText(metadata_text)
        dialog.setStandardButtons(QMessageBox.Ok)
        dialog.exec()
        
    def export_data(self):
        """Export current view to file"""
        # TODO: Implement export dialog
        logger.info("Export data requested")
        
    def show_context_menu(self, position):
        """Show context menu for table"""
        menu = QMenu()
        
        copy_action = QAction("Copy", self)
        copy_action.setShortcut(QKeySequence.Copy)
        copy_action.triggered.connect(self.copy_selection)
        menu.addAction(copy_action)
        
        menu.addSeparator()
        
        plot_action = QAction("Plot selected", self)
        plot_action.triggered.connect(self.plot_selection)
        menu.addAction(plot_action)
        
        # Add map plotting option if available
        if MAP_INTEGRATION_AVAILABLE:
            map_action = QAction("📍 Plot on Map", self)
            map_action.triggered.connect(self.plot_on_map)
            menu.addAction(map_action)
        
        menu.exec_(self.table_view.mapToGlobal(position))
        
    def copy_selection(self):
        """Copy selected cells to clipboard"""
        # TODO: Implement copy functionality
        logger.info("Copy selection requested")
        
    def plot_selection(self):
        """Plot selected data"""
        # TODO: Implement plotting functionality
        logger.info("Plot selection requested")
        
    def get_selected_data(self):
        """Get currently selected data"""
        selection = self.table_view.selectedIndexes()
        if not selection:
            return None
            
        # Get unique rows
        rows = sorted(set(index.row() for index in selection))
        df = self.model.get_dataframe()
        
        return df.iloc[rows]
    
    def show_processing_menu(self):
        """Show the processing menu dialog"""
        try:
            logger.debug("Processing button clicked!")
            df = self.model.get_dataframe()
            logger.debug(f"DataFrame shape: {df.shape}")
            
            if df.empty:
                logger.warning("No data available for processing")
                QMessageBox.warning(self, "No Data", "Please load data before processing.")
                return
                
            logger.debug("Importing ProcessingDialog...")
            # Import here to avoid circular imports
            from .processing.processing_dialog import ProcessingDialog
            
            logger.debug("Creating processing dialog...")
            dialog = ProcessingDialog(df, self, input_file_path=self.current_file)
            
            # Connect layer creation signal to forward to main application
            dialog.layer_created.connect(self.layer_created.emit)
            
            logger.debug("Showing processing dialog...")
            
            result = dialog.exec()
            logger.debug(f"Dialog result: {result}")
            
            if result == QDialog.Accepted:
                # Get processed data
                result = dialog.get_result()
                if result and result.success and result.data is not None:
                    logger.info("Processing completed successfully, updating data viewer")
                    # Update the data viewer with processed data
                    self.set_dataframe(result.data)
                    # NOTE: Automatic map signal emission removed for clean processor architecture
                    # self.data_selected.emit(result.data)
                    
                    # Show success message with file output info
                    message = f"Processing completed successfully!\n"
                    # Only show anomalies if they were actually found (not None or 0)
                    anomalies_found = result.metadata.get('anomalies_found')
                    if anomalies_found is not None and anomalies_found > 0:
                        message += f"Anomalies found: {anomalies_found}\n"
                    message += f"Processing time: {result.processing_time:.2f}s\n"
                    if result.output_file_path:
                        message += f"\nOutput file saved to:\n{result.output_file_path}"
                    
                    QMessageBox.information(
                        self, 
                        "Processing Complete",
                        message
                    )
                else:
                    logger.warning("Processing dialog accepted but no valid result returned")
        except Exception as e:
            logger.error(f"Error in show_processing_menu: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            QMessageBox.critical(self, "Error", f"Failed to open processing dialog:\n{str(e)}")
    
    def get_current_dataframe(self):
        """Get the current dataframe"""
        return self.model.get_dataframe()
    
    def detect_coordinate_columns(self, df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
        """Auto-detect latitude and longitude columns in DataFrame"""
        columns = df.columns.tolist()
        lat_col = None
        lon_col = None
        
        # Common latitude column names
        lat_keywords = ['lat', 'latitude', 'y', 'northing', 'north']
        lon_keywords = ['lon', 'lng', 'longitude', 'x', 'easting', 'east']
        
        for col in columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in lat_keywords):
                lat_col = col
            elif any(keyword in col_lower for keyword in lon_keywords):
                lon_col = col
        
        # Log what we found
        if lat_col and lon_col:
            logger.info(f"Detected coordinate columns: lat='{lat_col}', lon='{lon_col}'")
        else:
            logger.warning(f"Could not detect coordinate columns. Lat found: {lat_col}, Lon found: {lon_col}")
            
        return lat_col, lon_col
    
    def plot_on_map(self):
        """Create map layer from current data and emit signal"""
        if not MAP_INTEGRATION_AVAILABLE:
            QMessageBox.warning(self, "Map Integration", "Map integration is not available.")
            return
            
        layer = self.create_map_layer()
        if layer:
            self.layer_created.emit(layer)
            logger.info(f"Emitted map layer: {layer.name}")
            QMessageBox.information(self, "Map Layer Created", 
                                  f"Layer '{layer.name}' has been sent to the map.\n"
                                  f"Points: {len(layer.data)} | Bounds: {layer.bounds}")
        else:
            QMessageBox.warning(self, "No Coordinates", 
                              "Could not create map layer.\n"
                              "Make sure your data contains coordinate columns.")
    
    def create_map_layer(self) -> Optional['UXOLayer']:
        """Create UXOLayer from current DataFrame"""
        df = self.model.get_dataframe()
        
        if df.empty:
            logger.warning("Cannot create map layer from empty DataFrame")
            return None
            
        # Auto-detect geometry columns
        lat_col, lon_col = self.detect_coordinate_columns(df)
        
        if not lat_col or not lon_col:
            logger.warning("No valid coordinate columns found in data")
            return None
            
        # Calculate bounds
        try:
            lats = df[lat_col].dropna()
            lons = df[lon_col].dropna()
            
            if len(lats) == 0 or len(lons) == 0:
                logger.warning("No valid coordinates in data")
                return None
                
            bounds = [
                float(lons.min()), float(lats.min()),
                float(lons.max()), float(lats.max())
            ]
        except Exception as e:
            logger.error(f"Error calculating bounds: {e}")
            bounds = None
            
        # Create layer name
        if self.current_file:
            import os
            base_name = os.path.basename(self.current_file)
            layer_name = f"Survey Data - {base_name}"
        else:
            layer_name = f"Survey Data - {len(df)} points"
            
        # Extract metadata
        metadata = {
            "row_count": len(df),
            "columns": df.columns.tolist(),
            "lat_column": lat_col,
            "lon_column": lon_col,
            "source_file": self.current_file or "Unknown"
        }
        
        # Add any file metadata if available
        if hasattr(df, 'attrs') and 'file_metadata' in df.attrs:
            metadata['file_metadata'] = df.attrs['file_metadata']
            
        # Create style based on data characteristics
        style = self.create_default_style(df)
        
        # Create UXOLayer
        try:
            layer = UXOLayer(
                name=layer_name,
                layer_type=LayerType.POINTS,
                data=df,
                geometry_type=GeometryType.POINT,
                style=style,
                metadata=metadata,
                source=LayerSource.DATA_VIEWER,
                bounds=bounds
            )
            
            logger.info(f"Created UXO layer: {layer_name} with {len(df)} points")
            return layer
            
        except Exception as e:
            logger.error(f"Failed to create UXOLayer: {e}")
            return None
    
    def create_default_style(self, df: pd.DataFrame) -> 'LayerStyle':
        """Create default style based on data characteristics"""
        # Basic style
        style = LayerStyle()
        
        # Adjust based on data size
        num_points = len(df)
        if num_points > 10000:
            # For large datasets, use smaller points and enable clustering
            style.point_size = 4
            style.enable_clustering = True
            style.cluster_distance = 80
        elif num_points > 1000:
            style.point_size = 5
            style.enable_clustering = True
            style.cluster_distance = 50
        else:
            # For small datasets, use larger points without clustering
            style.point_size = 8
            style.enable_clustering = False
            
        # Check for anomaly or classification columns
        columns_lower = [col.lower() for col in df.columns]
        
        if any('anomaly' in col for col in columns_lower):
            # If anomaly data, use red color
            style.point_color = "#FF0000"
        elif any('class' in col or 'type' in col for col in columns_lower):
            # If classification data, prepare for graduated colors
            style.use_graduated_colors = True
            style.color_ramp = ["#0066CC", "#00CC66", "#FFCC00", "#FF6600", "#CC00CC"]
            
        return style


class DataViewer(QWidget):
    """Main data viewer widget with tabbed interface for multiple datasets"""
    
    # Signals
    data_selected = Signal(object)  # Emits selected data from active tab
    layer_created = Signal(object)  # Emits UXOLayer for map integration
    
    def __init__(self):
        super().__init__()
        # Set a reasonable minimum height and a flexible size policy
        self.setMinimumHeight(50)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.setup_ui()
        self.update_ui_state()
        
    def setup_ui(self):
        """Initialize the data viewer UI"""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Tab widget (no top toolbar needed - tabs handle everything)
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabsClosable(True)
        self.tab_widget.setMovable(True)
        self.tab_widget.tabCloseRequested.connect(self.close_tab)
        self.tab_widget.currentChanged.connect(self.tab_changed)
        
        # Add context menu for tab bar to create new tabs
        self.tab_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tab_widget.customContextMenuRequested.connect(self.show_tab_context_menu)
        
        # Placeholder for when no tabs are open
        self.placeholder = QWidget()
        placeholder_layout = QVBoxLayout()
        placeholder_layout.setAlignment(Qt.AlignCenter)

        label = QLabel("No data files open")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("color: gray; font-size: 14px;")

        help_label = QLabel("Open a new data file from the File Explorer")
        help_label.setAlignment(Qt.AlignCenter)
        help_label.setStyleSheet("color: gray; font-size: 12px;")

        placeholder_layout.addWidget(label)
        placeholder_layout.addWidget(help_label)
        self.placeholder.setLayout(placeholder_layout)
        
        layout.addWidget(self.tab_widget)
        layout.addWidget(self.placeholder)
        self.setLayout(layout)
        
    def add_empty_tab(self):
        """Add an empty data viewer tab"""
        tab = DataViewerTab()
        
        # Connect signals
        tab.data_selected.connect(self.data_selected)
        tab.layer_created.connect(self.layer_created)
        tab.tab_title_changed.connect(lambda title: self.update_tab_title(tab, title))
        
        # Add tab
        index = self.tab_widget.addTab(tab, "New Tab")
        self.tab_widget.setCurrentIndex(index)
        
        # Update UI state
        self.update_ui_state()
        
        return tab
        
    def add_file_tab(self, filepath):
        """Add a new tab with a specific file loaded"""
        tab = DataViewerTab(filepath)
        
        # Connect signals
        tab.data_selected.connect(self.data_selected)
        tab.layer_created.connect(self.layer_created)
        tab.tab_title_changed.connect(lambda title: self.update_tab_title(tab, title))
        
        # Add tab with filename as title
        filename = os.path.basename(filepath)
        index = self.tab_widget.addTab(tab, filename)
        self.tab_widget.setCurrentIndex(index)
        
        # Update UI state
        self.update_ui_state()
        
        return tab
        
    def show_tab_context_menu(self, position):
        """Show context menu for tab bar"""
        menu = QMenu()
        
        new_tab_action = QAction("📄 Open Data File", self)
        new_tab_action.setToolTip("Open a new data file in a new tab")
        new_tab_action.triggered.connect(self.open_new_tab)
        menu.addAction(new_tab_action)
        
        # Only show close option if we have actual data tabs
        has_data_tabs = self.tab_widget.count() > 0
        
        if has_data_tabs:
            menu.addSeparator()
            
            close_tab_action = QAction("✕ Close Current Tab", self)
            close_tab_action.triggered.connect(self.close_current_tab)
            menu.addAction(close_tab_action)
        
        # Show menu at the clicked position
        menu.exec_(self.tab_widget.mapToGlobal(position))
        
    def open_new_tab(self):
        """Open file dialog and create new tab with selected file"""
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Data files (*.csv *.xlsx *.xls *.json);;CSV files (*.csv);;Excel files (*.xlsx *.xls);;JSON files (*.json);;All files (*)")
        
        if file_dialog.exec() == QFileDialog.Accepted:
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                filepath = selected_files[0]
                try:
                    self.add_file_tab(filepath)
                except Exception as e:
                    logger.error(f"Error opening file {filepath}: {e}")
                    QMessageBox.critical(self, "Error", f"Failed to open file:\n{str(e)}")
                    
    def close_tab(self, index):
        """Close tab at given index"""
        # Always remove the tab
        self.tab_widget.removeTab(index)
        
        # Update UI state (will show welcome message if no tabs remain)
        self.update_ui_state()
        
    def close_current_tab(self):
        """Close the currently active tab"""
        current_index = self.tab_widget.currentIndex()
        if current_index >= 0:
            self.close_tab(current_index)
            
    def tab_changed(self, index):
        """Handle tab change"""
        self.update_ui_state()
        
        # Note: Removed automatic data emission to prevent unwanted map plotting
        # Data should only be emitted when explicitly requested (e.g., via "Plot on Map" button)
                
    def update_tab_title(self, tab, title):
        """Update the title of a specific tab"""
        index = self.tab_widget.indexOf(tab)
        if index >= 0:
            self.tab_widget.setTabText(index, title)
            
    def update_ui_state(self):
        """Update UI state based on current tabs"""
        has_tabs = self.tab_widget.count() > 0
        
        self.tab_widget.setVisible(has_tabs)
        self.placeholder.setVisible(not has_tabs)
            
    def get_current_tab(self) -> Optional[DataViewerTab]:
        """Get the currently active tab"""
        current_index = self.tab_widget.currentIndex()
        if current_index >= 0:
            return self.tab_widget.widget(current_index)
        return None
        
    def load_data(self, filepath):
        """Load data into current tab or create new tab"""
        current_tab = self.get_current_tab()
        
        if current_tab and isinstance(current_tab, DataViewerTab) and current_tab.get_current_dataframe().empty:
            # Use current empty data tab
            current_tab.load_data(filepath)
        else:
            # Create new tab
            self.add_file_tab(filepath)
            
    def get_current_dataframe(self):
        """Get the current DataFrame from active tab"""
        current_tab = self.get_current_tab()
        if current_tab and isinstance(current_tab, DataViewerTab):
            return current_tab.get_current_dataframe()
        return pd.DataFrame()
        
    def set_dataframe(self, df):
        """Set DataFrame in current tab"""
        current_tab = self.get_current_tab()
        if current_tab and isinstance(current_tab, DataViewerTab):
            current_tab.set_dataframe(df)
        else:
            # If no data tab exists, create one
            tab = self.add_empty_tab()
            tab.set_dataframe(df) 