
"""
Data Viewer Widget for UXO Wizard - Display and analyze tabular data and images
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QTableView, QToolBar, 
    QComboBox, QToolButton, QLabel, QLineEdit,
    QHeaderView, QMenu, QMessageBox, QDialog, QDialogButtonBox,
    QTabWidget, QHBoxLayout, QFileDialog, QSizePolicy,
    QStackedWidget, QScrollArea, QListWidget, QListWidgetItem,
    QPushButton, QGroupBox, QCheckBox, QFrame, QRadioButton,
    QSpinBox, QFormLayout, QProgressBar, QTextEdit, QApplication
)
from PySide6.QtGui import QAction, QIcon, QPixmap, QKeySequence, QFont
from PySide6.QtCore import Qt, Signal, QAbstractTableModel, QModelIndex, QEvent, QTimer
import pandas as pd
import numpy as np
from loguru import logger
from typing import Optional, Tuple
import os
import pickle
import re
from matplotlib.figure import Figure
from .plot_widget import PlotWidget

# Import map layer types for integration
try:
    from ..map.layer_types import UXOLayer, LayerType, GeometryType, LayerStyle, LayerSource
    MAP_INTEGRATION_AVAILABLE = True
except ImportError:
    logger.warning("Map integration not available - layer_types module not found")
    MAP_INTEGRATION_AVAILABLE = False


class ChunkedDataLoader:
    """Handles chunked loading of large CSV files for memory optimization"""
    
    def __init__(self, filepath, chunk_size=10000, encoding='utf-8'):
        self.filepath = filepath
        self.chunk_size = chunk_size
        self.encoding = encoding
        self.total_rows = 0
        self.column_names = []
        self.metadata = {}
        self.data_start_line = 0
        self.header_line = None
        self.delimiter = None
        self.chunks_cache = {}  # LRU cache for loaded chunks
        self.cache_size = 5  # Keep 5 chunks in memory
        self.initialized = False
        
    def initialize(self):
        """Initialize the loader by parsing file structure"""
        if self.initialized:
            return
            
        try:
            # Parse file header using existing logic
            self.metadata, self.data_start_line, self.header_line = self._parse_file_header()
            
            # Detect delimiter
            self.delimiter = self._detect_csv_delimiter()
            
            # Count total rows and get column names
            self._analyze_file_structure()
            
            self.initialized = True
            logger.info(f"Chunked loader initialized: {self.total_rows} rows, {len(self.column_names)} columns")
            
        except Exception as e:
            logger.error(f"Error initializing chunked loader: {e}")
            raise
    
    def _parse_file_header(self):
        """Parse multi-line header and find where tabular data starts"""
        try:
            with open(self.filepath, 'r', encoding=self.encoding) as f:
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
                    
                # Check if this line looks like a data header
                delimiter_count = max(line.count(';'), line.count(','), line.count('\t'), line.count('|'))

                if delimiter_count >= 3:
                    delim = max([';', ',', '\t', '|'], key=line.count)
                    tokens = [t.strip().lower() for t in line.split(delim)]

                    if tokens and 'timestamp' in tokens[0]:
                        header_line = i
                        data_start_line = i + 1
                        logger.debug(f"Found data header at line {i + 1}: {line[:100]}...")
                        break
                
                # Parse metadata from header section
                if ':' in line and '=' not in line:
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip()
                        metadata[key] = value
            
            return metadata, data_start_line, header_line
            
        except Exception as e:
            logger.warning(f"Error parsing file header: {e}")
            return {}, 0, None
    
    def _detect_csv_delimiter(self):
        """Detect the most likely delimiter in the CSV file"""
        try:
            with open(self.filepath, 'r', encoding=self.encoding) as f:
                # Skip to data section
                for _ in range(self.data_start_line):
                    f.readline()
                
                # Read a few lines to analyze
                sample_lines = []
                for _ in range(min(10, 100)):
                    line = f.readline()
                    if not line:
                        break
                    sample_lines.append(line.strip())
            
            # Count delimiters
            delimiters = [';', ',', '\t', '|']
            delimiter_counts = {delim: 0 for delim in delimiters}
            
            for line in sample_lines:
                for delim in delimiters:
                    delimiter_counts[delim] += line.count(delim)
            
            # Find most common delimiter
            best_delimiter = max(delimiter_counts, key=delimiter_counts.get)
            
            if delimiter_counts[best_delimiter] > 0:
                return best_delimiter
            else:
                return ';'  # Default fallback
                
        except Exception as e:
            logger.warning(f"Error detecting delimiter: {e}")
            return ';'  # Default fallback
    
    def _analyze_file_structure(self):
        """Analyze file to get total rows and column names"""
        try:
            # Get column names from header
            skip_rows = self.header_line if self.header_line is not None else self.data_start_line
            if skip_rows == 0:
                skip_rows = None
            
            # Read just the header to get column names
            sample_df = pd.read_csv(
                self.filepath,
                delimiter=self.delimiter,
                encoding=self.encoding,
                skiprows=skip_rows,
                header=0,
                nrows=1
            )
            
            self.column_names = sample_df.columns.str.strip().tolist()
            
            # Count total rows efficiently
            with open(self.filepath, 'r', encoding=self.encoding) as f:
                # Skip header lines
                for _ in range(self.data_start_line):
                    f.readline()
                
                # Count remaining lines
                self.total_rows = sum(1 for _ in f)
                
        except Exception as e:
            logger.error(f"Error analyzing file structure: {e}")
            self.column_names = []
            self.total_rows = 0
    
    def get_chunk(self, chunk_index):
        """Load a specific chunk of data"""
        if not self.initialized:
            self.initialize()
        
        # Check cache first
        if chunk_index in self.chunks_cache:
            return self.chunks_cache[chunk_index]
        
        try:
            # Calculate skip rows for this chunk
            skip_rows = self.data_start_line + (chunk_index * self.chunk_size)
            
            # Read chunk
            df = pd.read_csv(
                self.filepath,
                delimiter=self.delimiter,
                encoding=self.encoding,
                skiprows=skip_rows,
                header=None,
                names=self.column_names,
                nrows=self.chunk_size
            )
            
            # Cache the chunk
            self._cache_chunk(chunk_index, df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading chunk {chunk_index}: {e}")
            # Return empty DataFrame with correct columns
            return pd.DataFrame(columns=self.column_names)
    
    def _cache_chunk(self, chunk_index, df):
        """Cache a chunk with LRU eviction"""
        # Remove oldest cache entries if needed
        if len(self.chunks_cache) >= self.cache_size:
            # Remove the oldest entry (simple FIFO, could be improved to true LRU)
            oldest_key = next(iter(self.chunks_cache))
            del self.chunks_cache[oldest_key]
        
        self.chunks_cache[chunk_index] = df
    
    def get_chunk_count(self):
        """Get total number of chunks"""
        if not self.initialized:
            self.initialize()
        return (self.total_rows + self.chunk_size - 1) // self.chunk_size
    
    def get_full_dataframe(self):
        """Load complete DataFrame (fallback for small files)"""
        if not self.initialized:
            self.initialize()
        
        # If file is small, load it normally
        if self.total_rows <= self.chunk_size:
            return self.get_chunk(0)
        
        # For large files, load all chunks and combine
        logger.warning(f"Loading complete DataFrame with {self.total_rows} rows - this may use significant memory")
        
        chunks = []
        for i in range(self.get_chunk_count()):
            chunk = self.get_chunk(i)
            if not chunk.empty:
                chunks.append(chunk)
        
        if chunks:
            return pd.concat(chunks, ignore_index=True)
        else:
            return pd.DataFrame(columns=self.column_names)


class VirtualScrollTableModel(QAbstractTableModel):
    """Enhanced Qt model for displaying chunked data with optimized virtual scrolling"""
    
    def __init__(self, data_loader=None, parent=None):
        super().__init__(parent)
        self.data_loader = data_loader
        self.chunk_cache = {}  # Cache for multiple chunks
        self.cache_size = 3  # Keep 3 chunks in view cache
        self.current_visible_chunks = set()  # Track which chunks are currently visible
        
    def set_data_loader(self, data_loader):
        """Set the chunked data loader"""
        self.beginResetModel()
        self.data_loader = data_loader
        self.chunk_cache = {}
        self.current_visible_chunks = set()
        if data_loader:
            self._preload_initial_chunks()
        self.endResetModel()
    
    def _preload_initial_chunks(self):
        """Preload the first few chunks for smooth initial scrolling"""
        if not self.data_loader or not self.data_loader.initialized:
            return
        
        try:
            # Preload first 2 chunks
            chunks_to_preload = min(2, self.data_loader.get_chunk_count())
            for i in range(chunks_to_preload):
                chunk_data = self.data_loader.get_chunk(i)
                self.chunk_cache[i] = chunk_data
                self.current_visible_chunks.add(i)
                
        except Exception as e:
            logger.error(f"Error preloading initial chunks: {e}")
    
    def _get_chunk_data(self, chunk_index):
        """Get chunk data with intelligent caching"""
        if chunk_index in self.chunk_cache:
            return self.chunk_cache[chunk_index]
        
        # Load the chunk
        try:
            chunk_data = self.data_loader.get_chunk(chunk_index)
            
            # Manage cache size
            if len(self.chunk_cache) >= self.cache_size:
                # Remove chunks that are not in current visible set
                chunks_to_remove = []
                for cached_chunk in self.chunk_cache:
                    if cached_chunk not in self.current_visible_chunks:
                        chunks_to_remove.append(cached_chunk)
                        if len(chunks_to_remove) >= len(self.chunk_cache) - self.cache_size + 1:
                            break
                
                for chunk_to_remove in chunks_to_remove:
                    del self.chunk_cache[chunk_to_remove]
            
            # Cache the new chunk
            self.chunk_cache[chunk_index] = chunk_data
            return chunk_data
            
        except Exception as e:
            logger.error(f"Error loading chunk {chunk_index}: {e}")
            return pd.DataFrame(columns=self.data_loader.column_names)
    
    def _update_visible_chunks(self, visible_rows):
        """Update tracking of visible chunks based on visible rows"""
        if not self.data_loader:
            return
        
        # Calculate which chunks are visible
        chunk_size = self.data_loader.chunk_size
        visible_chunks = set()
        
        for row in visible_rows:
            chunk_index = row // chunk_size
            visible_chunks.add(chunk_index)
        
        self.current_visible_chunks = visible_chunks
        
        # Preload adjacent chunks for smooth scrolling
        adjacent_chunks = set()
        for chunk in visible_chunks:
            # Add previous and next chunks
            if chunk > 0:
                adjacent_chunks.add(chunk - 1)
            if chunk < self.data_loader.get_chunk_count() - 1:
                adjacent_chunks.add(chunk + 1)
        
        # Load adjacent chunks in background (don't block UI)
        for chunk in adjacent_chunks:
            if chunk not in self.chunk_cache:
                try:
                    chunk_data = self.data_loader.get_chunk(chunk)
                    if len(self.chunk_cache) < self.cache_size * 2:  # Allow some extra for smooth scrolling
                        self.chunk_cache[chunk] = chunk_data
                except Exception:
                    pass  # Ignore errors for background loading
    
    def rowCount(self, parent=QModelIndex()):
        if not self.data_loader:
            return 0
        return self.data_loader.total_rows
    
    def columnCount(self, parent=QModelIndex()):
        if not self.data_loader:
            return 0
        return len(self.data_loader.column_names)
    
    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or not self.data_loader:
            return None
        
        if role == Qt.DisplayRole:
            try:
                # Calculate which chunk contains this row
                chunk_index = index.row() // self.data_loader.chunk_size
                row_in_chunk = index.row() % self.data_loader.chunk_size
                
                # Get the chunk data
                chunk_data = self._get_chunk_data(chunk_index)
                
                # Get data from chunk
                if row_in_chunk < len(chunk_data):
                    value = chunk_data.iloc[row_in_chunk, index.column()]
                    
                    # Format the value appropriately
                    if pd.isna(value):
                        return "NaN"
                    elif isinstance(value, float):
                        return f"{value:.6g}"
                    else:
                        return str(value)
                else:
                    return "Loading..."
                    
            except Exception as e:
                logger.error(f"Error accessing data at {index.row()}, {index.column()}: {e}")
                return "Error"
        
        elif role == Qt.TextAlignmentRole:
            if self.data_loader and index.column() < len(self.data_loader.column_names):
                # Try to determine alignment from column name
                col_name = self.data_loader.column_names[index.column()]
                if any(keyword in col_name.lower() for keyword in ['time', 'id', 'index']):
                    return Qt.AlignLeft | Qt.AlignVCenter
                else:
                    return Qt.AlignRight | Qt.AlignVCenter
            return Qt.AlignLeft | Qt.AlignVCenter
        
        return None
    
    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                if self.data_loader and section < len(self.data_loader.column_names):
                    return self.data_loader.column_names[section]
                return f"Column {section}"
            else:
                return f"{section + 1}"
        return None
    
    def get_dataframe(self):
        """Get the complete DataFrame (warning: may use significant memory)"""
        if not self.data_loader:
            return pd.DataFrame()
        return self.data_loader.get_full_dataframe()
    
    def get_visible_data(self, start_row, end_row):
        """Get data for visible rows (optimized for exports and processing)"""
        if not self.data_loader:
            return pd.DataFrame()
        
        try:
            # Calculate which chunks we need
            chunk_size = self.data_loader.chunk_size
            start_chunk = start_row // chunk_size
            end_chunk = end_row // chunk_size
            
            # Collect chunks
            chunks = []
            for chunk_idx in range(start_chunk, end_chunk + 1):
                chunk_data = self._get_chunk_data(chunk_idx)
                
                # Calculate row range within this chunk
                chunk_start_row = chunk_idx * chunk_size
                chunk_end_row = (chunk_idx + 1) * chunk_size
                
                # Calculate the slice within the chunk
                start_offset = max(0, start_row - chunk_start_row)
                end_offset = min(len(chunk_data), end_row - chunk_start_row + 1)
                
                if start_offset < end_offset:
                    chunk_slice = chunk_data.iloc[start_offset:end_offset]
                    chunks.append(chunk_slice)
            
            # Combine chunks
            if chunks:
                return pd.concat(chunks, ignore_index=True)
            else:
                return pd.DataFrame(columns=self.data_loader.column_names)
                
        except Exception as e:
            logger.error(f"Error getting visible data: {e}")
            return pd.DataFrame(columns=self.data_loader.column_names if self.data_loader else [])


class ChunkedPandasModel(VirtualScrollTableModel):
    """Backward compatibility alias for VirtualScrollTableModel"""
    pass


class VirtualScrollTableView(QTableView):
    """Enhanced table view that works optimally with virtual scrolling models"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.last_visible_rows = set()
        self.scroll_timer = QTimer()
        self.scroll_timer.setSingleShot(True)
        self.scroll_timer.timeout.connect(self._on_scroll_finished)
        
    def scrollContentsBy(self, dx, dy):
        """Override to provide smooth scrolling feedback"""
        super().scrollContentsBy(dx, dy)
        
        # Restart timer on scroll
        self.scroll_timer.start(100)  # 100ms delay after scroll stops
        
    def _on_scroll_finished(self):
        """Called when scrolling has finished"""
        # Update visible chunks in the model
        if hasattr(self.model(), 'data_loader') and self.model().data_loader:
            visible_rows = self._get_visible_rows()
            if hasattr(self.model(), '_update_visible_chunks'):
                self.model()._update_visible_chunks(visible_rows)
    
    def _get_visible_rows(self):
        """Get the range of currently visible rows"""
        if not self.model():
            return []
        
        # Get the visible rect
        visible_rect = self.viewport().rect()
        
        # Convert to model coordinates
        top_index = self.indexAt(visible_rect.topLeft())
        bottom_index = self.indexAt(visible_rect.bottomLeft())
        
        if not top_index.isValid():
            top_row = 0
        else:
            top_row = top_index.row()
        
        if not bottom_index.isValid():
            bottom_row = self.model().rowCount() - 1
        else:
            bottom_row = bottom_index.row()
        
        return list(range(max(0, top_row), min(self.model().rowCount(), bottom_row + 1)))
    
    def setModel(self, model):
        """Override to setup connections with virtual scroll model"""
        super().setModel(model)
        
        # If it's a virtual scroll model, do initial update
        if isinstance(model, VirtualScrollTableModel):
            # Initial update of visible chunks
            self._on_scroll_finished()


class ExportDialog(QDialog):
    """Advanced export dialog with column/row selection and format options"""
    
    def __init__(self, model, selected_rows=None, parent=None):
        super().__init__(parent)
        self.model = model
        self.selected_rows = selected_rows or []
        self.setup_ui()
        self.update_preview()
        
    def setup_ui(self):
        """Setup the export dialog UI"""
        self.setWindowTitle("Export Data")
        self.setModal(True)
        self.resize(500, 600)
        
        layout = QVBoxLayout()
        
        # Column selection group
        self.column_group = QGroupBox("Columns to Export")
        column_layout = QVBoxLayout()
        
        # Column selection controls
        column_controls = QHBoxLayout()
        self.select_all_cols_btn = QPushButton("Select All")
        self.select_none_cols_btn = QPushButton("Select None")
        self.invert_cols_btn = QPushButton("Invert Selection")
        
        self.select_all_cols_btn.clicked.connect(self.select_all_columns)
        self.select_none_cols_btn.clicked.connect(self.select_none_columns)
        self.invert_cols_btn.clicked.connect(self.invert_column_selection)
        
        column_controls.addWidget(self.select_all_cols_btn)
        column_controls.addWidget(self.select_none_cols_btn)
        column_controls.addWidget(self.invert_cols_btn)
        column_controls.addStretch()
        
        # Column list
        self.column_list = QListWidget()
        self.column_list.setSelectionMode(QListWidget.MultiSelection)
        self.column_list.setMaximumHeight(150)
        
        # Populate columns
        if hasattr(self.model, 'data_loader') and self.model.data_loader:
            # Virtual scroll model
            columns = self.model.data_loader.column_names
        else:
            # Regular model
            df = self.model.get_dataframe()
            columns = df.columns.tolist()
        
        for col in columns:
            item = QListWidgetItem(col)
            self.column_list.addItem(item)
            item.setSelected(True)  # Default: select all
        
        self.column_list.itemSelectionChanged.connect(self.update_preview)
        
        column_layout.addLayout(column_controls)
        column_layout.addWidget(self.column_list)
        self.column_group.setLayout(column_layout)
        
        # Row selection group
        self.row_group = QGroupBox("Rows to Export")
        row_layout = QVBoxLayout()
        
        # Row selection options
        self.all_rows_radio = QRadioButton("All rows")
        self.selected_rows_radio = QRadioButton("Selected rows only")
        self.range_rows_radio = QRadioButton("Row range:")
        
        self.all_rows_radio.setChecked(True)
        
        # Row range controls
        range_layout = QHBoxLayout()
        self.start_row_spin = QSpinBox()
        self.end_row_spin = QSpinBox()
        
        self.start_row_spin.setMinimum(1)
        self.end_row_spin.setMinimum(1)
        
        if self.model:
            max_rows = self.model.rowCount()
            self.start_row_spin.setMaximum(max_rows)
            self.end_row_spin.setMaximum(max_rows)
            self.end_row_spin.setValue(min(1000, max_rows))  # Default to first 1000 rows
        
        range_layout.addWidget(QLabel("From:"))
        range_layout.addWidget(self.start_row_spin)
        range_layout.addWidget(QLabel("To:"))
        range_layout.addWidget(self.end_row_spin)
        range_layout.addStretch()
        
        # Enable/disable based on selection
        self.selected_rows_radio.setEnabled(len(self.selected_rows) > 0)
        
        # Connect radio buttons
        self.all_rows_radio.toggled.connect(self.update_preview)
        self.selected_rows_radio.toggled.connect(self.update_preview)
        self.range_rows_radio.toggled.connect(self.update_preview)
        self.start_row_spin.valueChanged.connect(self.update_preview)
        self.end_row_spin.valueChanged.connect(self.update_preview)
        
        row_layout.addWidget(self.all_rows_radio)
        row_layout.addWidget(self.selected_rows_radio)
        row_layout.addWidget(self.range_rows_radio)
        row_layout.addLayout(range_layout)
        self.row_group.setLayout(row_layout)
        
        # Format selection group
        self.format_group = QGroupBox("Export Format")
        format_layout = QFormLayout()
        
        self.format_combo = QComboBox()
        self.format_combo.addItems(["CSV", "Excel", "JSON", "TSV"])
        self.format_combo.currentTextChanged.connect(self.update_preview)
        
        format_layout.addRow("Format:", self.format_combo)
        self.format_group.setLayout(format_layout)
        
        # Preview group
        self.preview_group = QGroupBox("Export Preview")
        preview_layout = QVBoxLayout()
        
        self.preview_text = QTextEdit()
        self.preview_text.setMaximumHeight(100)
        self.preview_text.setReadOnly(True)
        
        preview_layout.addWidget(self.preview_text)
        self.preview_group.setLayout(preview_layout)
        
        # Progress bar (initially hidden)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        
        # Layout
        layout.addWidget(self.column_group)
        layout.addWidget(self.row_group)
        layout.addWidget(self.format_group)
        layout.addWidget(self.preview_group)
        layout.addWidget(self.progress_bar)
        layout.addWidget(button_box)
        
        self.setLayout(layout)
    
    def select_all_columns(self):
        """Select all columns"""
        for i in range(self.column_list.count()):
            self.column_list.item(i).setSelected(True)
    
    def select_none_columns(self):
        """Deselect all columns"""
        for i in range(self.column_list.count()):
            self.column_list.item(i).setSelected(False)
    
    def invert_column_selection(self):
        """Invert column selection"""
        for i in range(self.column_list.count()):
            item = self.column_list.item(i)
            item.setSelected(not item.isSelected())
    
    def update_preview(self):
        """Update the export preview"""
        try:
            config = self.get_export_config()
            
            # Calculate stats
            total_rows = self.model.rowCount() if self.model else 0
            selected_cols = len(config['columns'])
            
            if config['row_mode'] == 'all':
                selected_rows = total_rows
            elif config['row_mode'] == 'selected':
                selected_rows = len(self.selected_rows)
            else:  # range
                start = config.get('start_row', 1)
                end = config.get('end_row', total_rows)
                selected_rows = max(0, end - start + 1)
            
            # Estimate file size
            estimated_size = self.estimate_file_size(selected_rows, selected_cols, config['format'])
            
            # Update preview text
            preview_text = f"""Export Summary:
• Columns: {selected_cols} of {self.column_list.count()}
• Rows: {selected_rows:,} of {total_rows:,}
• Format: {config['format']}
• Estimated size: {estimated_size}

Selected columns: {', '.join(config['columns'][:5])}{'...' if len(config['columns']) > 5 else ''}
"""
            
            self.preview_text.setText(preview_text)
            
        except Exception as e:
            self.preview_text.setText(f"Error updating preview: {e}")
    
    def estimate_file_size(self, rows, cols, format_type):
        """Estimate the file size of the export"""
        # Rough estimates based on typical data
        bytes_per_cell = 20  # Average characters per cell
        overhead_factor = {
            'CSV': 1.1,
            'TSV': 1.1,
            'Excel': 1.5,
            'JSON': 2.0
        }
        
        total_bytes = rows * cols * bytes_per_cell * overhead_factor.get(format_type, 1.2)
        
        if total_bytes < 1024:
            return f"{total_bytes:.0f} bytes"
        elif total_bytes < 1024 * 1024:
            return f"{total_bytes/1024:.1f} KB"
        else:
            return f"{total_bytes/(1024*1024):.1f} MB"
    
    def get_export_config(self):
        """Get the current export configuration"""
        selected_columns = [
            item.text() for item in self.column_list.selectedItems()
        ]
        
        if self.all_rows_radio.isChecked():
            row_mode = 'all'
        elif self.selected_rows_radio.isChecked():
            row_mode = 'selected'
        else:
            row_mode = 'range'
        
        config = {
            'columns': selected_columns,
            'row_mode': row_mode,
            'format': self.format_combo.currentText(),
            'start_row': self.start_row_spin.value(),
            'end_row': self.end_row_spin.value()
        }
        
        return config
    
    def show_progress(self, show=True):
        """Show/hide progress bar"""
        self.progress_bar.setVisible(show)
        if show:
            self.progress_bar.setValue(0)
    
    def update_progress(self, value):
        """Update progress bar value"""
        self.progress_bar.setValue(value)


class SearchResultsHighlighter:
    """Manages search result highlighting in the table view"""
    
    def __init__(self, table_view):
        self.table_view = table_view
        self.search_results = []  # List of (row, col) tuples
        self.current_result_index = -1
        self.search_text = ""
        self.case_sensitive = False
        self.regex_mode = False
        self.whole_word = False
        
    def clear_results(self):
        """Clear all search results"""
        self.search_results = []
        self.current_result_index = -1
        self.search_text = ""
        self.update_selection()
    
    def set_search_params(self, text, case_sensitive=False, regex_mode=False, whole_word=False):
        """Set search parameters"""
        self.search_text = text
        self.case_sensitive = case_sensitive
        self.regex_mode = regex_mode
        self.whole_word = whole_word
    
    def search(self, model, column_filter=None):
        """Perform search and store results"""
        self.search_results = []
        
        if not self.search_text.strip():
            return
        
        # Prepare search pattern
        if self.regex_mode:
            try:
                flags = 0 if self.case_sensitive else re.IGNORECASE
                pattern = re.compile(self.search_text, flags)
            except re.error:
                logger.warning(f"Invalid regex pattern: {self.search_text}")
                return
        else:
            # Escape special regex characters for literal search
            escaped_text = re.escape(self.search_text)
            if self.whole_word:
                escaped_text = r'\b' + escaped_text + r'\b'
            
            flags = 0 if self.case_sensitive else re.IGNORECASE
            pattern = re.compile(escaped_text, flags)
        
        # Search through data
        if hasattr(model, 'data_loader') and model.data_loader:
            # Handle chunked model
            self._search_chunked_model(model, pattern, column_filter)
        else:
            # Handle regular model
            self._search_regular_model(model, pattern, column_filter)
        
        # Reset to first result
        self.current_result_index = 0 if self.search_results else -1
        self.update_selection()
    
    def _search_chunked_model(self, model, pattern, column_filter):
        """Search in chunked model"""
        total_rows = model.rowCount()
        chunk_size = model.data_loader.chunk_size
        
        # Search through chunks
        for chunk_start in range(0, total_rows, chunk_size):
            chunk_data = model.data_loader.get_chunk(chunk_start // chunk_size)
            
            if chunk_data.empty:
                continue
                
            # Apply column filter if specified
            columns_to_search = column_filter if column_filter else range(len(chunk_data.columns))
            
            for col_idx in columns_to_search:
                if col_idx >= len(chunk_data.columns):
                    continue
                    
                col_data = chunk_data.iloc[:, col_idx].astype(str)
                
                for row_idx, cell_value in enumerate(col_data):
                    if pattern.search(cell_value):
                        actual_row = chunk_start + row_idx
                        self.search_results.append((actual_row, col_idx))
    
    def _search_regular_model(self, model, pattern, column_filter):
        """Search in regular model"""
        df = model.get_dataframe()
        
        if df.empty:
            return
        
        # Apply column filter if specified
        columns_to_search = column_filter if column_filter else range(len(df.columns))
        
        for col_idx in columns_to_search:
            if col_idx >= len(df.columns):
                continue
                
            col_data = df.iloc[:, col_idx].astype(str)
            
            for row_idx, cell_value in enumerate(col_data):
                if pattern.search(cell_value):
                    self.search_results.append((row_idx, col_idx))
    
    def next_result(self):
        """Navigate to next search result"""
        if not self.search_results:
            return False
        
        self.current_result_index = (self.current_result_index + 1) % len(self.search_results)
        self.update_selection()
        return True
    
    def previous_result(self):
        """Navigate to previous search result"""
        if not self.search_results:
            return False
        
        self.current_result_index = (self.current_result_index - 1) % len(self.search_results)
        self.update_selection()
        return True
    
    def update_selection(self):
        """Update table selection to show current result"""
        if not self.search_results or self.current_result_index < 0:
            return
        
        row, col = self.search_results[self.current_result_index]
        
        # Select the cell
        index = self.table_view.model().index(row, col)
        self.table_view.selectionModel().select(index, self.table_view.selectionModel().ClearAndSelect)
        
        # Scroll to the cell
        self.table_view.scrollTo(index, self.table_view.PositionAtCenter)
    
    def get_results_info(self):
        """Get information about search results"""
        if not self.search_results:
            return "No results"
        
        current = self.current_result_index + 1 if self.current_result_index >= 0 else 0
        total = len(self.search_results)
        return f"{current}/{total} results"


class SearchToolBar(QFrame):
    """Advanced search toolbar with options"""
    
    search_requested = Signal(str, bool, bool, bool)  # text, case_sensitive, regex, whole_word
    next_result = Signal()
    previous_result = Signal()
    close_search = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        """Setup search toolbar UI"""
        layout = QHBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Search input
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search...")
        self.search_input.textChanged.connect(self.on_search_text_changed)
        self.search_input.returnPressed.connect(self.on_search_requested)
        
        # Search options
        self.case_sensitive_cb = QCheckBox("Case sensitive")
        self.regex_cb = QCheckBox("Regex")
        self.whole_word_cb = QCheckBox("Whole word")
        
        # Connect options to trigger search
        self.case_sensitive_cb.toggled.connect(self.on_search_requested)
        self.regex_cb.toggled.connect(self.on_search_requested)
        self.whole_word_cb.toggled.connect(self.on_search_requested)
        
        # Navigation buttons
        self.prev_btn = QPushButton("Previous")
        self.next_btn = QPushButton("Next")
        self.close_btn = QPushButton("Close")
        
        self.prev_btn.clicked.connect(self.previous_result.emit)
        self.next_btn.clicked.connect(self.next_result.emit)
        self.close_btn.clicked.connect(self.close_search.emit)
        
        # Results label
        self.results_label = QLabel("No results")
        
        # Layout
        layout.addWidget(QLabel("Search:"))
        layout.addWidget(self.search_input)
        layout.addWidget(self.case_sensitive_cb)
        layout.addWidget(self.regex_cb)
        layout.addWidget(self.whole_word_cb)
        layout.addWidget(self.prev_btn)
        layout.addWidget(self.next_btn)
        layout.addWidget(self.results_label)
        layout.addStretch()
        layout.addWidget(self.close_btn)
        
        self.setLayout(layout)
        
        # Initial state
        self.set_navigation_enabled(False)
        
    def on_search_text_changed(self, text):
        """Handle search text changes"""
        self.set_navigation_enabled(bool(text.strip()))
    
    def on_search_requested(self):
        """Handle search request"""
        text = self.search_input.text().strip()
        if text:
            self.search_requested.emit(
                text,
                self.case_sensitive_cb.isChecked(),
                self.regex_cb.isChecked(),
                self.whole_word_cb.isChecked()
            )
    
    def set_navigation_enabled(self, enabled):
        """Enable/disable navigation buttons"""
        self.prev_btn.setEnabled(enabled)
        self.next_btn.setEnabled(enabled)
    
    def update_results_info(self, info):
        """Update results information"""
        self.results_label.setText(info)
    
    def focus_search(self):
        """Focus the search input"""
        self.search_input.setFocus()


class ColumnSelectorWidget(QFrame):
    """Multi-column selection widget with enhanced functionality"""
    
    selection_changed = Signal(list)  # Emits list of selected column names
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.StyledPanel)
        self.setMaximumHeight(200)
        self.setMinimumWidth(250)
        self.columns = []
        self.setup_ui()
        
        # Set focus policy to allow keyboard navigation
        self.setFocusPolicy(Qt.StrongFocus)
        
    def setup_ui(self):
        """Initialize the UI components"""
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Title bar with close button
        title_layout = QHBoxLayout()
        title_label = QLabel("Column Selection")
        title_label.setStyleSheet("font-weight: bold; padding: 2px;")
        title_layout.addWidget(title_label)
        
        title_layout.addStretch()
        
        # Close button
        self.close_btn = QPushButton("✕")
        self.close_btn.setMaximumWidth(25)
        self.close_btn.setMaximumHeight(25)
        self.close_btn.setStyleSheet("QPushButton { border: none; background: transparent; font-weight: bold; } QPushButton:hover { background: #ffcccc; }")
        self.close_btn.clicked.connect(self.hide)
        title_layout.addWidget(self.close_btn)
        
        layout.addLayout(title_layout)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.select_all_btn = QPushButton("All")
        self.select_all_btn.setMaximumWidth(40)
        self.select_all_btn.setToolTip("Select all columns")
        self.select_all_btn.clicked.connect(self.select_all_columns)
        button_layout.addWidget(self.select_all_btn)
        
        self.select_none_btn = QPushButton("None")
        self.select_none_btn.setMaximumWidth(40)
        self.select_none_btn.setToolTip("Deselect all columns")
        self.select_none_btn.clicked.connect(self.select_no_columns)
        button_layout.addWidget(self.select_none_btn)
        
        self.invert_btn = QPushButton("Invert")
        self.invert_btn.setMaximumWidth(50)
        self.invert_btn.setToolTip("Invert current selection")
        self.invert_btn.clicked.connect(self.invert_selection)
        button_layout.addWidget(self.invert_btn)
        
        button_layout.addStretch()
        
        layout.addLayout(button_layout)
        
        # Column list
        self.column_list = QListWidget()
        self.column_list.setSelectionMode(QListWidget.MultiSelection)
        self.column_list.itemSelectionChanged.connect(self.on_selection_changed)
        self.column_list.setAlternatingRowColors(True)
        self.column_list.setToolTip("Click to select/deselect columns. Use Ctrl+Click for multi-selection.")
        layout.addWidget(self.column_list)
        
        # Status label
        self.status_label = QLabel("0 columns selected")
        self.status_label.setStyleSheet("color: #666; font-size: 10px;")
        layout.addWidget(self.status_label)
        
        self.setLayout(layout)
        
    def set_columns(self, columns):
        """Set available columns"""
        self.columns = columns
        self.column_list.clear()
        
        for column in columns:
            item = QListWidgetItem(column)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            self.column_list.addItem(item)
        
        # Select all by default
        self.select_all_columns()
        
    def select_all_columns(self):
        """Select all columns"""
        for i in range(self.column_list.count()):
            item = self.column_list.item(i)
            item.setSelected(True)
        self.on_selection_changed()
        
    def select_no_columns(self):
        """Deselect all columns"""
        self.column_list.clearSelection()
        self.on_selection_changed()
        
    def invert_selection(self):
        """Invert current selection"""
        for i in range(self.column_list.count()):
            item = self.column_list.item(i)
            item.setSelected(not item.isSelected())
        self.on_selection_changed()
        
    def on_selection_changed(self):
        """Handle selection changes"""
        selected_items = self.column_list.selectedItems()
        selected_columns = [item.text() for item in selected_items]
        
        # Update status
        count = len(selected_columns)
        total = len(self.columns)
        self.status_label.setText(f"{count} of {total} columns selected")
        
        # Emit signal
        self.selection_changed.emit(selected_columns)
        
    def get_selected_columns(self):
        """Get list of selected column names"""
        selected_items = self.column_list.selectedItems()
        return [item.text() for item in selected_items]
        
    def set_selected_columns(self, column_names):
        """Set selected columns by name"""
        self.column_list.clearSelection()
        
        for i in range(self.column_list.count()):
            item = self.column_list.item(i)
            if item.text() in column_names:
                item.setSelected(True)
        
        self.on_selection_changed()
        
    def mousePressEvent(self, event):
        """Handle mouse press events"""
        super().mousePressEvent(event)
        # Keep the popup open when clicked inside
        event.accept()
        
    def keyPressEvent(self, event):
        """Handle key press events"""
        if event.key() == Qt.Key_Escape:
            self.hide()
        elif event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            self.hide()
        else:
            super().keyPressEvent(event)
            
    def focusOutEvent(self, event):
        """Handle focus lost events"""
        # Simple focus out handling - hide popup when focus is lost
        if event.reason() == Qt.ActiveWindowFocusReason:
            self.hide()
        super().focusOutEvent(event)


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
    
    def __init__(self, filepath=None, project_manager=None):
        super().__init__()
        self.current_file = filepath
        self.current_content_type = None  # 'data' or 'image'
        self.original_pixmap = None
        self.zoom_factor = 1.0
        self.project_manager = project_manager
        self.data_loader = None  # For chunked data loading
        self.search_highlighter = None  # Will be initialized after table_view
        self.setup_ui()
        
        # Load data if filepath provided, otherwise set UI for empty data tab
        if filepath:
            self.load_data(filepath)
        else:
            self.set_ui_for_content_type('data')
        
    def setup_ui(self):
        """Initialize the UI for this tab"""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Toolbar with horizontal scrolling
        toolbar_scroll = QScrollArea()
        toolbar_scroll.setWidgetResizable(True)
        toolbar_scroll.setMaximumHeight(50)
        toolbar_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        toolbar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # Hide scrollbar but allow scrolling
        toolbar_scroll.setFrameStyle(QScrollArea.NoFrame)
        
        # Create a container widget for the toolbar content
        toolbar_container = QWidget()
        toolbar_layout = QHBoxLayout(toolbar_container)
        toolbar_layout.setContentsMargins(5, 5, 5, 5)
        toolbar_layout.setSpacing(8)
        
        toolbar_scroll.setWidget(toolbar_container)
        
        # Install event filter for mouse wheel horizontal scrolling
        toolbar_scroll.installEventFilter(self)
        self.toolbar_scroll = toolbar_scroll  # Store reference for event handling
        
        # Info label
        self.info_label = QLabel("No data loaded")
        toolbar_layout.addWidget(self.info_label)
        
        # Add separator
        separator1 = QFrame()
        separator1.setFrameShape(QFrame.VLine)
        separator1.setFrameShadow(QFrame.Sunken)
        toolbar_layout.addWidget(separator1)
        
        # Create a container widget for data-specific controls
        self.data_toolbar_widget = QWidget()
        data_toolbar_layout = QHBoxLayout()
        data_toolbar_layout.setContentsMargins(0, 5, 0, 5)
        data_toolbar_layout.setSpacing(5)  # Consistent spacing between controls
        self.data_toolbar_widget.setLayout(data_toolbar_layout)

        # Column filter
        self.column_label = QLabel("Columns:")
        data_toolbar_layout.addWidget(self.column_label)
        
        # Column selection button that opens dropdown
        self.column_selector_btn = QPushButton("All columns")
        self.column_selector_btn.setMinimumWidth(150)
        self.column_selector_btn.setFixedHeight(30)
        self.column_selector_btn.clicked.connect(self.toggle_column_selector)
        data_toolbar_layout.addWidget(self.column_selector_btn)
        
        # Column selector widget (initially hidden)
        self.column_selector = ColumnSelectorWidget(self)
        self.column_selector.setVisible(False)
        self.column_selector.setWindowFlags(Qt.Popup | Qt.FramelessWindowHint)  # Make it act like a popup
        self.column_selector.selection_changed.connect(self.filter_columns_advanced)
        
        # Search button (to open advanced search) - make it smaller
        self.search_btn = QPushButton("🔍")
        self.search_btn.setFixedSize(30, 30)
        self.search_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.search_btn.setStyleSheet("QPushButton { padding: 0px; margin: 0px; }")
        self.search_btn.setToolTip("Search data (Ctrl+F)")
        self.search_btn.clicked.connect(self.toggle_search_toolbar)
        data_toolbar_layout.addWidget(self.search_btn)
        
        # Process button - moved here from later in the toolbar
        self.process_btn = QToolButton()
        self.process_btn.setText("⚡ Process")
        self.process_btn.setFixedHeight(30)
        self.process_btn.setToolTip("Open Processing Menu")
        self.process_btn.clicked.connect(self.show_processing_menu)
        data_toolbar_layout.addWidget(self.process_btn)

        toolbar_layout.addWidget(self.data_toolbar_widget)
        
        # Add separator
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.VLine)
        separator2.setFrameShadow(QFrame.Sunken)
        toolbar_layout.addWidget(separator2)
        
        # Statistics button
        self.stats_btn = QToolButton()
        self.stats_btn.setText("Stats")
        self.stats_btn.setFixedHeight(30)
        self.stats_btn.clicked.connect(self.show_statistics)
        toolbar_layout.addWidget(self.stats_btn)
        
        # Metadata button  
        self.metadata_btn = QToolButton()
        self.metadata_btn.setText("Metadata")
        self.metadata_btn.setFixedHeight(30)
        self.metadata_btn.setToolTip("Show file metadata and header information")
        self.metadata_btn.clicked.connect(self.show_metadata)
        self.metadata_btn.setEnabled(False)  # Disabled until data with metadata is loaded
        toolbar_layout.addWidget(self.metadata_btn)
        
        # Export button
        self.export_btn = QToolButton()
        self.export_btn.setText("Export")
        self.export_btn.setFixedHeight(30)
        self.export_btn.clicked.connect(self.export_data)
        toolbar_layout.addWidget(self.export_btn)
        
        # Add separator
        separator3 = QFrame()
        separator3.setFrameShape(QFrame.VLine)
        separator3.setFrameShadow(QFrame.Sunken)
        toolbar_layout.addWidget(separator3)
        
        # Create a container widget for image controls
        self.image_toolbar_widget = QWidget()
        image_toolbar_layout = QHBoxLayout()
        image_toolbar_layout.setContentsMargins(0, 5, 0, 5) # Add some vertical margin
        self.image_toolbar_widget.setLayout(image_toolbar_layout)
        
        # Image zoom controls
        self.zoom_in_btn = QToolButton()
        self.zoom_in_btn.setText("🔍+")
        self.zoom_in_btn.setToolTip("Zoom In (Ctrl+Mouse Wheel)")
        self.zoom_in_btn.clicked.connect(self.zoom_in)
        image_toolbar_layout.addWidget(self.zoom_in_btn)
        
        self.zoom_out_btn = QToolButton()
        self.zoom_out_btn.setText("🔍-")
        self.zoom_out_btn.setToolTip("Zoom Out (Ctrl+Mouse Wheel)")
        self.zoom_out_btn.clicked.connect(self.zoom_out)
        image_toolbar_layout.addWidget(self.zoom_out_btn)
        
        self.zoom_fit_btn = QToolButton()
        self.zoom_fit_btn.setText("⌂")
        self.zoom_fit_btn.setToolTip("Fit to Window")
        self.zoom_fit_btn.clicked.connect(self.zoom_to_fit)
        image_toolbar_layout.addWidget(self.zoom_fit_btn)
        
        self.zoom_label = QLabel("100%")
        self.zoom_label.setMinimumWidth(50)
        self.zoom_label.setToolTip("Current zoom level")
        image_toolbar_layout.addWidget(self.zoom_label)
        
        # Add the container widget to the toolbar and get the associated QAction.
        # Controlling the action's visibility is the robust way to do this.
        toolbar_layout.addWidget(self.image_toolbar_widget)
        
        # Plot on Map button (only show if map integration available)
        if MAP_INTEGRATION_AVAILABLE:
            self.plot_map_btn = QToolButton()
            self.plot_map_btn.setText("📍 Plot on Map")
            self.plot_map_btn.setFixedHeight(30)
            self.plot_map_btn.setToolTip("Visualize this table as points on the map (requires coordinate columns)")
            self.plot_map_btn.clicked.connect(self.plot_on_map)
            toolbar_layout.addWidget(self.plot_map_btn)
        
        # Add stretch to push everything to the left
        toolbar_layout.addStretch()

        # Central content area
        self.content_stack = QStackedWidget()
        
        # Table view for data
        self.table_view = VirtualScrollTableView()
        self.table_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.table_view.setSortingEnabled(True)
        self.table_view.setAlternatingRowColors(True)
        self.table_view.setSelectionBehavior(QTableView.SelectRows)
        
        # Enable horizontal scrolling for better data visibility
        self.table_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.table_view.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Image viewer for images
        self.image_viewer = QScrollArea()
        self.image_viewer.setWidgetResizable(True)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_viewer.setWidget(self.image_label)
        # Add event filter for wheel zoom
        self.image_viewer.viewport().installEventFilter(self)

        self.plot_widget = PlotWidget()
        
        # Text viewer for plain text files
        self.text_viewer = QTextEdit()
        self.text_viewer.setReadOnly(True)
        self.text_viewer.setFont(QFont("Courier", 12))  # Monospace font for better text display

        self.content_stack.addWidget(self.table_view)
        self.content_stack.addWidget(self.image_viewer)
        self.content_stack.addWidget(self.plot_widget)
        self.content_stack.addWidget(self.text_viewer)

        # Model
        self.model = PandasModel()
        self.table_view.setModel(self.model)
        
        # Context menu
        self.table_view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table_view.customContextMenuRequested.connect(self.show_context_menu)
        
        # Search toolbar (initially hidden)
        self.search_toolbar = SearchToolBar()
        self.search_toolbar.setVisible(False)
        
        # Connect search toolbar signals
        self.search_toolbar.search_requested.connect(self.perform_search)
        self.search_toolbar.next_result.connect(self.next_search_result)
        self.search_toolbar.previous_result.connect(self.previous_search_result)
        self.search_toolbar.close_search.connect(self.close_search)
        
        # Initialize search highlighter
        self.search_highlighter = SearchResultsHighlighter(self.table_view)
        
        # Add keyboard shortcuts for column sizing
        self.setup_keyboard_shortcuts()
        
        # Layout
        layout.addWidget(toolbar_scroll)
        layout.addWidget(self.search_toolbar)
        layout.addWidget(self.content_stack)
        self.setLayout(layout)
        
    def setup_keyboard_shortcuts(self):
        """Setup keyboard shortcuts for common actions"""
        # Column sizing shortcuts
        resize_shortcut = QKeySequence("Ctrl+R")
        resize_action = QAction(self)
        resize_action.setShortcut(resize_shortcut)
        resize_action.triggered.connect(self.resize_columns_to_contents)
        self.addAction(resize_action)
        
        # Reset sizing shortcut
        reset_shortcut = QKeySequence("Ctrl+Shift+R")
        reset_action = QAction(self)
        reset_action.setShortcut(reset_shortcut)
        reset_action.triggered.connect(self.reset_column_sizing)
        self.addAction(reset_action)
        
        # Search shortcut
        search_shortcut = QKeySequence("Ctrl+F")
        search_action = QAction(self)
        search_action.setShortcut(search_shortcut)
        search_action.triggered.connect(self.toggle_search_toolbar)
        self.addAction(search_action)
        
        logger.debug("Keyboard shortcuts configured")
        
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
                    
            raise Exception("Failed to load CSV file with all attempted methods")

    def set_ui_for_content_type(self, content_type: str):
        """Configures the UI for 'data', 'image', 'plot', or 'text' content type"""
        self.current_content_type = content_type
        is_data = (content_type == 'data')
        is_image = (content_type == 'image')
        is_plot = (content_type == 'plot')
        is_text = (content_type == 'text')

        # Show/hide data-specific controls
        self.data_toolbar_widget.setVisible(is_data)
        self.stats_btn.setVisible(is_data)
        self.metadata_btn.setVisible(is_data)
        if MAP_INTEGRATION_AVAILABLE and hasattr(self, 'plot_map_btn'):
            self.plot_map_btn.setVisible(is_data)

        # Show/hide image-specific controls
        self.image_toolbar_widget.setVisible(is_image)
        
        # The export button is always visible, but its action depends on content type
        self.export_btn.setEnabled(is_data or is_image or is_plot or is_text)

        # For data, metadata button state is set in set_dataframe
        # For images, let's disable it for now.
        if not is_data:
            self.metadata_btn.setEnabled(False)


    def load_data(self, filepath):
        """Load data or image from file"""
        try:
            filename = os.path.basename(filepath)
            file_lower = filepath.lower()

            # Determine file type and load accordingly
            if file_lower.endswith(('.csv', '.xlsx', '.xls', '.json')):
                if file_lower.endswith('.csv'):
                    # Check file size to determine loading strategy
                    file_size = os.path.getsize(filepath)
                    size_mb = file_size / (1024 * 1024)
                    
                    if size_mb > 50:  # Use chunked loading for files > 50MB
                        logger.info(f"Large file detected ({size_mb:.1f} MB), using chunked loading")
                        self._load_csv_chunked(filepath)
                    else:
                        df = self._load_csv_enhanced(filepath)
                        self.set_dataframe(df)
                elif file_lower.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(filepath)
                    self.set_dataframe(df)
                elif file_lower.endswith('.json'):
                    df = pd.read_json(filepath)
                    self.set_dataframe(df)
            elif file_lower.endswith(('.txt', '.dat')):
                # Display text files as plain text
                self.load_text_file(filepath)
            elif file_lower.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.tif')):
                self.load_image(filepath)
            elif file_lower.endswith('.mplplot'):
                self.load_plot(filepath)
            else:
                logger.error(f"Unsupported file type for Data Viewer: {filepath}")
                QMessageBox.warning(self, "Unsupported File", f"The file type of '{filename}' is not supported in the Data Viewer.")
                return

            self.current_file = filepath
            self.tab_title_changed.emit(filename)
            logger.info(f"Loaded file in Data Viewer: {filepath}")

        except Exception as e:
            logger.error(f"Error loading file in Data Viewer: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to load file:\n{str(e)}")

    def _load_csv_chunked(self, filepath):
        """Load CSV file using chunked loading for better memory efficiency"""
        try:
            # Create chunked data loader
            data_loader = ChunkedDataLoader(filepath, chunk_size=10000)
            data_loader.initialize()
            
            # Create virtual scroll model
            virtual_model = VirtualScrollTableModel(data_loader)
            
            # Set the chunked model
            self.set_chunked_model(virtual_model, data_loader)
            
            logger.info(f"Loaded {data_loader.total_rows} rows in virtual scroll mode")
            
        except Exception as e:
            logger.error(f"Error in chunked CSV loading: {e}")
            # Fallback to regular loading
            logger.info("Falling back to regular CSV loading")
            df = self._load_csv_enhanced(filepath)
            self.set_dataframe(df)

    def load_image(self, filepath):
        """Load and display an image file"""
        try:
            # Load the image
            self.original_pixmap = QPixmap(filepath)
            
            if self.original_pixmap.isNull():
                raise Exception("Failed to load image - invalid format or corrupted file")
            
            # Reset zoom and display image
            self.zoom_factor = 1.0
            self.content_stack.setCurrentWidget(self.image_viewer)
            self.set_ui_for_content_type('image')
            
            # Initial display at actual size
            self.update_image_display()
            
            # Update info label with image details
            width = self.original_pixmap.width()
            height = self.original_pixmap.height()
            file_size = os.path.getsize(filepath)
            file_size_mb = file_size / (1024 * 1024)
            
            self.info_label.setText(f"Image: {width}×{height} pixels | Size: {file_size_mb:.1f} MB")
            
            logger.info(f"Successfully loaded image: {width}×{height} pixels")
            
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            raise
    
    def load_text_file(self, filepath):
        """Load and display a text file as plain text"""
        try:
            # Try to read with different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252']
            content = None
            
            for encoding in encodings:
                try:
                    with open(filepath, 'r', encoding=encoding) as file:
                        content = file.read()
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                raise Exception("Could not decode file with any supported encoding")
            
            # Clear any previous data model
            self.model.setData(pd.DataFrame())  # Clear with empty DataFrame
            self.data_loader = None
            
            # Set the text content
            self.text_viewer.setPlainText(content)
            
            # Switch to text viewer
            self.content_stack.setCurrentWidget(self.text_viewer)
            self.set_ui_for_content_type('text')
            
            # Update info label with file details
            file_size = os.path.getsize(filepath)
            file_size_kb = file_size / 1024
            line_count = content.count('\n') + 1
            char_count = len(content)
            
            self.info_label.setText(f"Text: {line_count} lines | {char_count} characters | Size: {file_size_kb:.1f} KB")
            
            logger.info(f"Successfully loaded text file: {line_count} lines, {char_count} characters")
            
        except Exception as e:
            logger.error(f"Error loading text file: {e}")
            raise
    
    def load_plot(self, filepath):
        """Load and display a pickled matplotlib figure."""
        try:
            with open(filepath, 'rb') as f:
                figure = pickle.load(f)
            self.set_figure(figure, os.path.basename(filepath))
            logger.info(f"Successfully loaded plot: {filepath}")
        except Exception as e:
            logger.error(f"Error loading plot: {e}")
            raise

    def set_dataframe(self, df):
        """Set the DataFrame to display"""
        # Check if the current model supports setData (PandasModel)
        if hasattr(self.model, 'setData') and hasattr(self.model, '_data'):
            self.model.setData(df)
        else:
            # For chunked/virtual models, replace with regular PandasModel
            self.model = PandasModel(df)
            self.table_view.setModel(self.model)
        
        self.content_stack.setCurrentWidget(self.table_view)
        self.set_ui_for_content_type('data')

        # Check for metadata
        has_metadata = hasattr(df, 'attrs') and 'file_metadata' in df.attrs
        self.metadata_btn.setEnabled(has_metadata)
        
        # Update column selector with new columns
        if hasattr(self, 'column_selector') and df is not None:
            self.column_selector.set_columns(df.columns.tolist())
        
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
        
        # Update column selector
        self.column_selector.set_columns(df.columns.tolist())
        
        # Optimize column sizing for better data visibility
        self.optimize_column_sizing(df)

    def set_chunked_model(self, chunked_model, data_loader):
        """Set the chunked model for large file display"""
        self.model = chunked_model
        self.table_view.setModel(chunked_model)
        self.content_stack.setCurrentWidget(self.table_view)
        self.set_ui_for_content_type('data')

        # Check for metadata
        has_metadata = bool(data_loader.metadata)
        self.metadata_btn.setEnabled(has_metadata)
        
        # Update column selector with new columns
        if hasattr(self, 'column_selector') and data_loader.column_names:
            self.column_selector.set_columns(data_loader.column_names)
        
        # Update UI info label with data info
        info_text = f"Rows: {data_loader.total_rows:,} | Columns: {len(data_loader.column_names)}"
        
        if has_metadata:
            # Add some key metadata to the info label
            key_info = []
            for key in ['Date', 'Time', 'Field-Nr.', 'MagWalk', 'Samples']:
                if key in data_loader.metadata:
                    key_info.append(f"{key}: {data_loader.metadata[key]}")
            
            if key_info:
                info_text += f" | {' | '.join(key_info[:2])}"  # Show first 2 metadata items
        
        self.info_label.setText(info_text)
        
        # Update column selector
        self.column_selector.set_columns(data_loader.column_names)
        
        # Store data loader reference for access by other methods
        self.data_loader = data_loader
        
        # Optimize column sizing for chunked data
        self.optimize_column_sizing_chunked(data_loader)

    def optimize_column_sizing(self, df):
        """Optimize column sizing for better data visibility with horizontal scrolling"""
        if df.empty:
            return
            
        header = self.table_view.horizontalHeader()
        
        # Always use interactive mode for full control
        header.setSectionResizeMode(QHeaderView.Interactive)
        
        # Don't stretch last section - let it scroll horizontally
        header.setStretchLastSection(False)
        
        # Enable horizontal scrolling
        self.table_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Calculate optimal column widths based on content
        column_count = len(df.columns)
        for i in range(column_count):
            # Get column name width
            column_name = df.columns[i]
            header_width = header.fontMetrics().boundingRect(column_name).width() + 20  # Add padding
            
            # Sample data to estimate content width
            sample_data = df.iloc[:min(100, len(df)), i].astype(str)  # Sample first 100 rows
            if not sample_data.empty:
                # Get maximum content width from sample
                max_content_width = 0
                font_metrics = self.table_view.fontMetrics()
                
                for value in sample_data:
                    content_width = font_metrics.boundingRect(str(value)).width() + 20  # Add padding
                    max_content_width = max(max_content_width, content_width)
                
                # Use the larger of header width or content width, with reasonable bounds
                optimal_width = max(header_width, max_content_width)
                optimal_width = max(80, min(optimal_width, 300))  # Min 80px, max 300px
            else:
                optimal_width = max(80, header_width)
            
            header.resizeSection(i, optimal_width)
        
        logger.debug(f"Optimized column sizing for {column_count} columns")

    def optimize_column_sizing_chunked(self, data_loader):
        """Optimize column sizing for chunked data models"""
        if not data_loader or not data_loader.column_names:
            return
            
        header = self.table_view.horizontalHeader()
        
        # Always use interactive mode for full control
        header.setSectionResizeMode(QHeaderView.Interactive)
        
        # Don't stretch last section - let it scroll horizontally
        header.setStretchLastSection(False)
        
        # Enable horizontal scrolling
        self.table_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # For chunked data, we'll use a more conservative approach
        # since we can't easily sample all data
        column_count = len(data_loader.column_names)
        
        # Get first chunk to sample data
        if data_loader.get_chunk_count() > 0:
            sample_chunk = data_loader.get_chunk(0)
            
            for i, column_name in enumerate(data_loader.column_names):
                # Get column name width
                header_width = header.fontMetrics().boundingRect(column_name).width() + 20
                
                # Sample content width from first chunk
                if not sample_chunk.empty and i < len(sample_chunk.columns):
                    sample_data = sample_chunk.iloc[:, i].astype(str)
                    max_content_width = 0
                    font_metrics = self.table_view.fontMetrics()
                    
                    for value in sample_data:
                        content_width = font_metrics.boundingRect(str(value)).width() + 20
                        max_content_width = max(max_content_width, content_width)
                    
                    # Use the larger of header width or content width
                    optimal_width = max(header_width, max_content_width)
                    optimal_width = max(80, min(optimal_width, 300))  # Min 80px, max 300px
                else:
                    optimal_width = max(80, header_width)
                
                header.resizeSection(i, optimal_width)
        else:
            # Fallback: use header width + some padding
            for i, column_name in enumerate(data_loader.column_names):
                header_width = header.fontMetrics().boundingRect(column_name).width() + 20
                optimal_width = max(80, min(header_width, 200))
                header.resizeSection(i, optimal_width)
        
        logger.debug(f"Optimized chunked column sizing for {column_count} columns")

    def reset_column_sizing(self):
        """Reset column sizing to content-based optimal widths"""
        if isinstance(self.model, (ChunkedPandasModel, VirtualScrollTableModel)):
            if self.data_loader:
                self.optimize_column_sizing_chunked(self.data_loader)
        else:
            df = self.model.get_dataframe()
            if not df.empty:
                self.optimize_column_sizing(df)

    def resize_columns_to_contents(self):
        """Resize all columns to fit their contents"""
        self.table_view.resizeColumnsToContents()
        header = self.table_view.horizontalHeader()
        
        # Apply reasonable maximum width to prevent extremely wide columns
        for i in range(header.count()):
            current_width = header.sectionSize(i)
            if current_width > 400:  # Max 400px per column
                header.resizeSection(i, 400)

    def set_figure(self, figure: Figure, title: str):
        """Set a figure directly, for plots generated by scripts."""
        self.plot_widget.set_figure(figure)
        self.content_stack.setCurrentWidget(self.plot_widget)
        self.set_ui_for_content_type('plot')
        self.info_label.setText(f"Interactive Plot: {title}")
        self.current_file = None
        self.tab_title_changed.emit(title)

    def filter_columns(self, column):
        """Filter displayed columns - legacy method for backward compatibility"""
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
    
    def toggle_column_selector(self):
        """Toggle column selector widget visibility"""
        if self.column_selector.isVisible():
            self.column_selector.setVisible(False)
        else:
            # Position the selector below the button
            button_rect = self.column_selector_btn.rect()
            button_pos = self.column_selector_btn.mapToGlobal(button_rect.bottomLeft())
            
            # Adjust position to ensure it's visible
            self.column_selector.adjustSize()
            self.column_selector.move(button_pos)
            self.column_selector.setVisible(True)
            self.column_selector.raise_()
            self.column_selector.setFocus()
    
    def filter_columns_advanced(self, selected_columns):
        """Enhanced column filtering with multi-selection support"""
        if not selected_columns:
            # No selection - hide all columns
            for i in range(self.model.columnCount()):
                self.table_view.hideColumn(i)
            self.column_selector_btn.setText("No columns")
            self.column_selector_btn.setStyleSheet("color: #cc0000; font-weight: bold;")
        else:
            # Show only selected columns - handle both regular and chunked models
            selected_set = set(selected_columns)
            
            # Get column names depending on model type
            if isinstance(self.model, (ChunkedPandasModel, VirtualScrollTableModel)):
                column_names = self.model.data_loader.column_names
            else:
                df = self.model.get_dataframe()
                column_names = df.columns
            
            for i, col in enumerate(column_names):
                if col in selected_set:
                    self.table_view.showColumn(i)
                else:
                    self.table_view.hideColumn(i)
            
            # Update button text and styling
            count = len(selected_columns)
            total = len(column_names)
            if count == total:
                self.column_selector_btn.setText("All columns")
                self.column_selector_btn.setStyleSheet("")
            elif count == 1:
                # Show the actual column name if only one is selected
                self.column_selector_btn.setText(selected_columns[0])
                self.column_selector_btn.setStyleSheet("color: #006600; font-weight: bold;")
            else:
                self.column_selector_btn.setText(f"{count} columns")
                self.column_selector_btn.setStyleSheet("color: #006600; font-weight: bold;")
        
        # Don't auto-hide selector - let user decide when to close it
        # self.column_selector.setVisible(False)
                    
    def toggle_search_toolbar(self):
        """Toggle the search toolbar visibility"""
        if self.search_toolbar.isVisible():
            self.close_search()
        else:
            self.search_toolbar.setVisible(True)
            self.search_toolbar.focus_search()
    
    def close_search(self):
        """Close the search toolbar and clear results"""
        self.search_toolbar.setVisible(False)
        self.search_highlighter.clear_results()
    
    def perform_search(self, text, case_sensitive, regex_mode, whole_word):
        """Perform search with given parameters"""
        if not text.strip():
            self.search_highlighter.clear_results()
            return
        
        # Set search parameters
        self.search_highlighter.set_search_params(text, case_sensitive, regex_mode, whole_word)
        
        # Get column filter if active
        column_filter = None
        if hasattr(self, 'column_selector'):
            selected_columns = self.column_selector.get_selected_columns()
            if selected_columns:
                # Convert column names to indices
                if isinstance(self.model, (ChunkedPandasModel, VirtualScrollTableModel)):
                    all_columns = self.model.data_loader.column_names
                else:
                    all_columns = self.model.get_dataframe().columns.tolist()
                
                column_filter = []
                for col_name in selected_columns:
                    if col_name in all_columns:
                        column_filter.append(all_columns.index(col_name))
        
        # Perform search
        self.search_highlighter.search(self.model, column_filter)
        
        # Update results info
        results_info = self.search_highlighter.get_results_info()
        self.search_toolbar.update_results_info(results_info)
        
        logger.info(f"Search completed: {results_info}")
    
    def next_search_result(self):
        """Navigate to next search result"""
        if self.search_highlighter.next_result():
            results_info = self.search_highlighter.get_results_info()
            self.search_toolbar.update_results_info(results_info)
    
    def previous_search_result(self):
        """Navigate to previous search result"""
        if self.search_highlighter.previous_result():
            results_info = self.search_highlighter.get_results_info()
            self.search_toolbar.update_results_info(results_info)
    
    def search_data(self, text):
        """Legacy search method for backward compatibility"""
        if text.strip():
            self.perform_search(text, False, False, False)
        else:
            self.search_highlighter.clear_results()
        
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
        try:
            if self.current_content_type == 'image':
                self.export_image()
            elif self.current_content_type == 'data':
                self.export_tabular_data()
            elif self.current_content_type == 'plot':
                self.export_plot()
            elif self.current_content_type == 'text':
                self.export_text()
            else:
                QMessageBox.information(self, "Export", "No content to export.")
        except Exception as e:
            logger.error(f"Error during export: {e}")
            QMessageBox.critical(self, "Export Error", f"Failed to export: {str(e)}")

    def export_image(self):
        """Export the currently displayed image"""
        if not self.original_pixmap:
            QMessageBox.warning(self, "Export", "No image loaded to export.")
            return

        # Get export file path
        file_dialog = QFileDialog()
        file_dialog.setAcceptMode(QFileDialog.AcceptSave)
        file_dialog.setNameFilter("PNG files (*.png);;JPEG files (*.jpg);;BMP files (*.bmp);;TIFF files (*.tiff);;All files (*)")
        file_dialog.setDefaultSuffix("png")
        
        if self.current_file:
            # Suggest a filename based on the current file
            base_name = os.path.splitext(os.path.basename(self.current_file))[0]
            file_dialog.selectFile(f"{base_name}_exported.png")

        if file_dialog.exec() == QFileDialog.Accepted:
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                export_path = selected_files[0]
                
                # Save the original image (not the scaled display version)
                success = self.original_pixmap.save(export_path)
                
                if success:
                    QMessageBox.information(self, "Export Complete", f"Image exported successfully to: {export_path}")
                    logger.info(f"Image exported to: {export_path}")
                else:
                    QMessageBox.critical(self, "Export Failed", "Failed to save the image.")

    def export_tabular_data(self):
        """Export the currently displayed tabular data with advanced options"""
        if not self.model:
            QMessageBox.warning(self, "Export", "No data to export.")
            return

        # Get current row selection
        selected_rows = self.get_selected_rows()
        
        # Show export dialog
        dialog = ExportDialog(self.model, selected_rows, self)
        if dialog.exec() == QDialog.Accepted:
            config = dialog.get_export_config()
            config['selected_rows'] = selected_rows  # Add selected rows to config
            self.perform_export(config)
    
    def get_selected_rows(self):
        """Get list of currently selected row indices"""
        selection = self.table_view.selectionModel()
        if not selection:
            return []
        
        selected_rows = []
        for index in selection.selectedRows():
            selected_rows.append(index.row())
        
        return sorted(selected_rows)
    
    def perform_export(self, config):
        """Perform the actual export based on configuration"""
        try:
            # Get the data to export
            export_data = self.prepare_export_data(config)
            
            if export_data.empty:
                QMessageBox.warning(self, "Export", "No data to export with current selection.")
                return
            
            # Get export file path
            file_dialog = QFileDialog()
            file_dialog.setAcceptMode(QFileDialog.AcceptSave)
            
            # Set file filter based on format
            format_filters = {
                'CSV': "CSV files (*.csv)",
                'Excel': "Excel files (*.xlsx)",
                'JSON': "JSON files (*.json)",
                'TSV': "TSV files (*.tsv)"
            }
            
            selected_format = config['format']
            filter_str = format_filters.get(selected_format, "CSV files (*.csv)")
            file_dialog.setNameFilter(f"{filter_str};;All files (*)")
            
            # Set default suffix
            suffix_map = {'CSV': 'csv', 'Excel': 'xlsx', 'JSON': 'json', 'TSV': 'tsv'}
            file_dialog.setDefaultSuffix(suffix_map.get(selected_format, 'csv'))
            
            if self.current_file:
                # Suggest a filename based on the current file
                base_name = os.path.splitext(os.path.basename(self.current_file))[0]
                suggested_name = f"{base_name}_exported.{suffix_map.get(selected_format, 'csv')}"
                file_dialog.selectFile(suggested_name)

            if file_dialog.exec() == QFileDialog.Accepted:
                selected_files = file_dialog.selectedFiles()
                if selected_files:
                    export_path = selected_files[0]
                    self.save_export_data(export_data, export_path, selected_format)
                    
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Failed to export data:\n{str(e)}")
            logger.error(f"Export failed: {e}")
    
    def prepare_export_data(self, config):
        """Prepare the data for export based on configuration"""
        # Get the appropriate data based on model type
        if isinstance(self.model, (ChunkedPandasModel, VirtualScrollTableModel)):
            # For chunked models, we need to handle this differently
            if config['row_mode'] == 'all':
                # Get full dataframe (warning: may use significant memory)
                df = self.model.get_dataframe()
            elif config['row_mode'] == 'selected':
                # Get specific rows
                if not config.get('selected_rows'):
                    return pd.DataFrame()
                
                # Use get_visible_data for efficient row extraction
                min_row = min(config['selected_rows'])
                max_row = max(config['selected_rows'])
                df = self.model.get_visible_data(min_row, max_row)
                
                # Filter to exact selected rows
                relative_indices = [row - min_row for row in config['selected_rows']]
                df = df.iloc[relative_indices]
            else:  # range
                start_row = config['start_row'] - 1  # Convert to 0-based index
                end_row = config['end_row'] - 1
                df = self.model.get_visible_data(start_row, end_row)
        else:
            # Regular model
            df = self.model.get_dataframe()
            
            if config['row_mode'] == 'selected':
                if not config.get('selected_rows'):
                    return pd.DataFrame()
                df = df.iloc[config['selected_rows']]
            elif config['row_mode'] == 'range':
                start_row = config['start_row'] - 1  # Convert to 0-based index
                end_row = config['end_row'] - 1
                df = df.iloc[start_row:end_row + 1]
        
        # Filter columns
        if config['columns']:
            # Make sure all requested columns exist
            available_columns = df.columns.tolist()
            valid_columns = [col for col in config['columns'] if col in available_columns]
            if valid_columns:
                df = df[valid_columns]
        
        return df
    
    def save_export_data(self, df, export_path, format_type):
        """Save the export data to file"""
        try:
            if format_type == 'CSV':
                df.to_csv(export_path, index=False)
            elif format_type == 'Excel':
                df.to_excel(export_path, index=False)
            elif format_type == 'JSON':
                df.to_json(export_path, orient='records', indent=2)
            elif format_type == 'TSV':
                df.to_csv(export_path, sep='\t', index=False)
            else:
                # Default to CSV
                df.to_csv(export_path, index=False)
            
            # Show success message with stats
            rows, cols = df.shape
            file_size = os.path.getsize(export_path)
            if file_size < 1024:
                size_str = f"{file_size} bytes"
            elif file_size < 1024 * 1024:
                size_str = f"{file_size/1024:.1f} KB"
            else:
                size_str = f"{file_size/(1024*1024):.1f} MB"
            
            QMessageBox.information(
                self, 
                "Export Complete", 
                f"Data exported successfully!\n\n"
                f"File: {os.path.basename(export_path)}\n"
                f"Rows: {rows:,}\n"
                f"Columns: {cols}\n"
                f"Size: {size_str}"
            )
            logger.info(f"Data exported to: {export_path} ({rows} rows, {cols} columns)")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Failed to save export data:\n{str(e)}")
            logger.error(f"Export save failed: {e}")
    
    def export_plot(self):
        """Export the currently displayed plot."""
        figure = self.plot_widget.get_figure()
        if not figure:
            QMessageBox.warning(self, "Export", "No plot to export.")
            return

        file_dialog = QFileDialog()
        file_dialog.setAcceptMode(QFileDialog.AcceptSave)
        file_dialog.setNameFilter("Interactive Plot (*.mplplot);;PNG Image (*.png);;SVG Vector Image (*.svg)")
        file_dialog.setDefaultSuffix("mplplot")

        if file_dialog.exec() == QFileDialog.Accepted:
            export_path = file_dialog.selectedFiles()[0]
            try:
                if export_path.lower().endswith('.mplplot'):
                    with open(export_path, 'wb') as f:
                        pickle.dump(figure, f)
                else:
                    figure.savefig(export_path)
                QMessageBox.information(self, "Export Complete", f"Plot exported successfully to: {export_path}")
                logger.info(f"Plot exported to: {export_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Failed", f"Failed to export plot: {str(e)}")
                logger.error(f"Export failed: {e}")
    
    def export_text(self):
        """Export the currently displayed text content"""
        content = self.text_viewer.toPlainText()
        if not content:
            QMessageBox.warning(self, "Export", "No text content to export.")
            return

        file_dialog = QFileDialog()
        file_dialog.setAcceptMode(QFileDialog.AcceptSave)
        file_dialog.setNameFilter("Text Files (*.txt);;All Files (*)")
        file_dialog.setDefaultSuffix("txt")

        if file_dialog.exec() == QFileDialog.Accepted:
            export_path = file_dialog.selectedFiles()[0]
            try:
                with open(export_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                QMessageBox.information(self, "Export Complete", f"Text exported successfully to: {export_path}")
                logger.info(f"Text content exported to: {export_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Failed", f"Failed to export text: {str(e)}")
                logger.error(f"Text export failed: {e}")

    def show_context_menu(self, position):
        """Show context menu for table (only for data content)"""
        if self.current_content_type != 'data':
            return
            
        menu = QMenu()
        
        copy_action = QAction("Copy", self)
        copy_action.setShortcut(QKeySequence.Copy)
        copy_action.triggered.connect(self.copy_selection)
        menu.addAction(copy_action)
        
        menu.addSeparator()
        
        # Column sizing options
        column_menu = menu.addMenu("Column Sizing")
        
        fit_contents_action = QAction("Resize to Contents (Ctrl+R)", self)
        fit_contents_action.setToolTip("Resize all columns to fit their content")
        fit_contents_action.triggered.connect(self.resize_columns_to_contents)
        column_menu.addAction(fit_contents_action)
        
        reset_sizing_action = QAction("Reset Optimal Sizing (Ctrl+Shift+R)", self)
        reset_sizing_action.setToolTip("Reset columns to optimal width with horizontal scrolling")
        reset_sizing_action.triggered.connect(self.reset_column_sizing)
        column_menu.addAction(reset_sizing_action)
        
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
    
    def eventFilter(self, source, event):
        """Handle wheel events for zooming on the image viewer and toolbar scrolling."""
        if source == self.image_viewer.viewport() and event.type() == QEvent.Wheel:
            if event.modifiers() == Qt.ControlModifier:
                if event.angleDelta().y() > 0:
                    self.zoom_in()
                else:
                    self.zoom_out()
                return True  # Event handled
        elif source == self.toolbar_scroll and event.type() == QEvent.Wheel:
            # Handle horizontal scrolling for toolbar
            scroll_bar = self.toolbar_scroll.horizontalScrollBar()
            if scroll_bar:
                delta = event.angleDelta().y()
                # Scroll horizontally instead of vertically
                scroll_bar.setValue(scroll_bar.value() - delta // 8)
                return True
        return super().eventFilter(source, event)

    def zoom_in(self):
        """Zoom in on the image."""
        self.zoom_factor *= 1.25
        self.update_image_display()

    def zoom_out(self):
        """Zoom out of the image."""
        self.zoom_factor *= 0.8
        self.update_image_display()

    def zoom_to_fit(self):
        """Fit the image to the viewport."""
        if not self.original_pixmap:
            return
    
        view_size = self.image_viewer.viewport().size()
        pixmap_size = self.original_pixmap.size()

        if view_size.width() <= 0 or view_size.height() <= 0 or pixmap_size.width() <= 0 or pixmap_size.height() <= 0:
            return

        width_ratio = view_size.width() / pixmap_size.width()
        height_ratio = view_size.height() / pixmap_size.height()
        self.zoom_factor = min(width_ratio, height_ratio)
    
        self.update_image_display()

    def update_image_display(self):
        """Update the image label with the scaled pixmap."""
        if not self.original_pixmap:
            return

        new_width = int(self.original_pixmap.width() * self.zoom_factor)
        new_height = int(self.original_pixmap.height() * self.zoom_factor)

        # Use a smooth transformation for better quality
        scaled_pixmap = self.original_pixmap.scaled(
            new_width, new_height,
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
    
        self.image_label.setPixmap(scaled_pixmap)
        self.zoom_label.setText(f"{self.zoom_factor:.0%}")

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
            dialog = ProcessingDialog(df, self, input_file_path=self.current_file, project_manager=self.project_manager)
            
            # Connect layer creation signal to forward to main application
            dialog.layer_created.connect(self.layer_created.emit)
            
            # Connect plot generation signal to create new plot tab - find parent DataViewer
            parent_viewer = self.parent()
            while parent_viewer and not isinstance(parent_viewer, DataViewer):
                parent_viewer = parent_viewer.parent()
            if parent_viewer:
                dialog.plot_generated.connect(parent_viewer.add_plot_tab)
            
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
                    message = "Processing completed successfully!\n"
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
        """Get the current dataframe if content is data, otherwise empty."""
        if self.current_content_type == 'data' and hasattr(self, 'model'):
            return self.model.get_dataframe()
        return pd.DataFrame()
    
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
    
    def __init__(self, project_manager=None):
        super().__init__()
        # Set a reasonable minimum height and a flexible size policy
        self.setMinimumHeight(50)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.project_manager = project_manager
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
        tab = DataViewerTab(project_manager=self.project_manager)
        
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
        tab = DataViewerTab(filepath, project_manager=self.project_manager)
        
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
        file_dialog.setNameFilter("All supported (*.csv *.xlsx *.xls *.json *.png *.jpg *.jpeg *.bmp *.gif *.tiff *.tif);;Data files (*.csv *.xlsx *.xls *.json);;Image files (*.png *.jpg *.jpeg *.bmp *.gif *.tiff *.tif);;CSV files (*.csv);;Excel files (*.xlsx *.xls);;JSON files (*.json);;All files (*)")
        
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
        _ = index  # Suppress unused parameter warning
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
        
        if current_tab and current_tab.current_file is None:
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
            self.update_tab_title(tab, "Processed Data")
    
    def add_plot_tab(self, figure: Figure, title: str):
        """Add a new tab containing a matplotlib figure"""
        tab = DataViewerTab(project_manager=self.project_manager)
        
        # Connect signals
        tab.data_selected.connect(self.data_selected)
        tab.layer_created.connect(self.layer_created)
        tab.tab_title_changed.connect(lambda title: self.update_tab_title(tab, title))
        
        # Set the figure directly
        tab.set_figure(figure, title)
        
        # Add tab with title
        index = self.tab_widget.addTab(tab, title)
        self.tab_widget.setCurrentIndex(index)
        
        # Update UI state
        self.update_ui_state()
        
        logger.info(f"Added new plot tab: {title}")
        return tab 