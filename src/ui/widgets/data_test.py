"""
Data Viewer Widget for UXO Wizard - Display and analyze tabular data, images, and plots
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QTableView, QToolBar,
    QComboBox, QToolButton, QLabel, QLineEdit,
    QHeaderView, QMenu, QMessageBox, QDialog, QDialogButtonBox,
    QTabWidget, QHBoxLayout, QFileDialog, QSizePolicy,
    QStackedWidget, QScrollArea
)
from PySide6.QtGui import QAction, QIcon, QPixmap, QKeySequence
from PySide6.QtCore import Qt, Signal, QAbstractTableModel, QModelIndex, QEvent
import pandas as pd
import numpy as np
from loguru import logger
from typing import Optional, Tuple
import pickle
import os
from matplotlib.figure import Figure

# Import map layer types for integration
try:
    from ..map.layer_types import UXOLayer, LayerType, GeometryType, LayerStyle, LayerSource
    MAP_INTEGRATION_AVAILABLE = True
except ImportError:
    logger.warning("Map integration not available - layer_types module not found")
    MAP_INTEGRATION_AVAILABLE = False

from .plot_widget import PlotWidget


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
        self.current_content_type = None  # 'data', 'image', or 'plot'
        self.original_pixmap = None
        self.zoom_factor = 1.0
        self.project_manager = project_manager
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

        # Toolbar
        toolbar = QToolBar()

        # Info label
        self.info_label = QLabel("No data loaded")
        toolbar.addWidget(self.info_label)

        toolbar.addSeparator()

        # Create a container widget for data-specific controls
        self.data_toolbar_widget = QWidget()
        data_toolbar_layout = QHBoxLayout()
        data_toolbar_layout.setContentsMargins(0, 5, 0, 5)
        self.data_toolbar_widget.setLayout(data_toolbar_layout)

        # Column filter
        self.column_label = QLabel("Column:")
        data_toolbar_layout.addWidget(self.column_label)
        self.column_combo = QComboBox()
        self.column_combo.setMinimumWidth(150)
        self.column_combo.currentTextChanged.connect(self.filter_columns)
        data_toolbar_layout.addWidget(self.column_combo)

        # Search
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search...")
        self.search_edit.textChanged.connect(self.search_data)
        data_toolbar_layout.addWidget(self.search_edit)

        self.data_toolbar_action = toolbar.addWidget(self.data_toolbar_widget)

        toolbar.addSeparator()

        # Statistics button
        self.stats_btn = QToolButton()
        self.stats_btn.setText("Stats")
        self.stats_btn.clicked.connect(self.show_statistics)
        self.stats_action = toolbar.addWidget(self.stats_btn)

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

        self.image_toolbar_action = toolbar.addWidget(self.image_toolbar_widget)

        toolbar.addSeparator()

        # Processing button
        self.process_btn = QToolButton()
        self.process_btn.setText("⚡ Process")
        self.process_btn.setToolTip("Open Processing Menu")
        self.process_btn.clicked.connect(self.show_processing_menu)
        self.process_action = toolbar.addWidget(self.process_btn)

        # Plot on Map button (only show if map integration available)
        if MAP_INTEGRATION_AVAILABLE:
            self.plot_map_btn = QToolButton()
            self.plot_map_btn.setText("📍 Plot on Map")
            self.plot_map_btn.setToolTip("Visualize this table as points on the map (requires coordinate columns)")
            self.plot_map_btn.clicked.connect(self.plot_on_map)
            self.plot_map_action = toolbar.addWidget(self.plot_map_btn)

        toolbar.addSeparator()

        # Central content area
        self.content_stack = QStackedWidget()

        # Table view for data
        self.table_view = QTableView()
        self.table_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.table_view.setSortingEnabled(True)
        self.table_view.setAlternatingRowColors(True)
        self.table_view.setSelectionBehavior(QTableView.SelectRows)

        # Image viewer for images
        self.image_viewer = QScrollArea()
        self.image_viewer.setWidgetResizable(True)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_viewer.setWidget(self.image_label)
        self.image_viewer.viewport().installEventFilter(self)

        # Plot viewer for matplotlib plots
        self.plot_widget = PlotWidget()

        self.content_stack.addWidget(self.table_view)
        self.content_stack.addWidget(self.image_viewer)
        self.content_stack.addWidget(self.plot_widget)

        # Model
        self.model = PandasModel()
        self.table_view.setModel(self.model)

        # Context menu
        self.table_view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table_view.customContextMenuRequested.connect(self.show_context_menu)

        # Layout
        layout.addWidget(toolbar)
        layout.addWidget(self.content_stack)
        self.setLayout(layout)

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

    def _clean_trailing_delimiters(self, df):
        """Remove empty columns caused by trailing delimiters"""
        try:
            empty_cols = []
            for col in df.columns:
                if (df[col].isna().all() or
                    (df[col].astype(str).str.strip() == '').all() or
                    col.startswith('Unnamed:')):
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
                        logger.debug(
                            f"Found data header at line {i + 1}: {line[:100]}..."
                        )
                        break

                if ':' in line and '=' not in line:
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip()
                        metadata[key] = value
                elif '=' in line and '[' in line and ']' in line:
                    if ':' in line:
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            key = parts[0].strip()
                            value = parts[1].strip()
                            metadata[key] = value

            if data_start_line == 0:
                for i, line in enumerate(lines):
                    line = line.strip()
                    if not line:
                        continue

                    for delim in [';', ',', '\t', '|']:
                        if line.count(delim) >= 5:
                            parts = line.split(delim)
                            numeric_count = 0
                            for part in parts[:10]:
                                part = part.strip()
                                try:
                                    float(part)
                                    numeric_count += 1
                                except ValueError:
                                    pass

                            if numeric_count >= len(parts) * 0.6:
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
        encoding = 'utf-8'
        metadata, data_start_line, header_line = self._parse_file_header(filepath, encoding)

        if metadata:
            logger.info(f"Found {len(metadata)} metadata entries in file header")

        delimiter = self._detect_csv_delimiter(filepath, encoding, data_start_line)

        if header_line is not None and header_line >= 0:
            skip_rows = header_line
        else:
            skip_rows = data_start_line if data_start_line > 0 else None

        if skip_rows == 0:
            skip_rows = None

        try:
            df = pd.read_csv(
                filepath,
                delimiter=delimiter,
                encoding=encoding,
                skiprows=skip_rows,
                header=0
            )

            df = self._clean_trailing_delimiters(df)
            df.columns = df.columns.str.strip()

            if hasattr(df, 'attrs'):
                df.attrs['file_metadata'] = metadata
                df.attrs['data_start_line'] = data_start_line
                df.attrs['source_file'] = filepath

            logger.info(f"Successfully loaded CSV: {df.shape[0]} rows, {df.shape[1]} columns")
            return df

        except Exception as e:
            logger.warning(f"Primary load failed ({e}), trying fallback options")

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
        """Configures the UI for 'data', 'image', or 'plot' content type"""
        self.current_content_type = content_type
        is_data = (content_type == 'data')
        is_image = (content_type == 'image')
        is_plot = (content_type == 'plot')

        if hasattr(self, 'data_toolbar_action'):
            self.data_toolbar_action.setVisible(is_data)
        if hasattr(self, 'stats_action'):
            self.stats_action.setVisible(is_data)
        if hasattr(self, 'process_action'):
            self.process_action.setVisible(is_data)
        if MAP_INTEGRATION_AVAILABLE and hasattr(self, 'plot_map_action'):
            self.plot_map_action.setVisible(is_data)
        if hasattr(self, 'image_toolbar_action'):
            self.image_toolbar_action.setVisible(is_image)

        self.export_btn.setEnabled(is_data or is_image or is_plot)
        self.metadata_btn.setEnabled(is_data)

    def load_data(self, filepath):
        """Load data, image, or plot from file"""
        try:
            filename = os.path.basename(filepath)
            file_lower = filepath.lower()

            if file_lower.endswith(('.csv', '.xlsx', '.xls', '.json')):
                if file_lower.endswith('.csv'):
                    df = self._load_csv_enhanced(filepath)
                elif file_lower.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(filepath)
                elif file_lower.endswith('.json'):
                    df = pd.read_json(filepath)
                self.set_dataframe(df)
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
            QMessageBox.critical(self, "Error", "Failed to load file:\n" + str(e))

    def load_image(self, filepath):
        """Load and display an image file"""
        try:
            self.original_pixmap = QPixmap(filepath)
            if self.original_pixmap.isNull():
                raise Exception("Failed to load image - invalid format or corrupted file")

            self.zoom_factor = 1.0
            self.content_stack.setCurrentWidget(self.image_viewer)
            self.set_ui_for_content_type('image')
            self.update_image_display()

            width = self.original_pixmap.width()
            height = self.original_pixmap.height()
            file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
            self.info_label.setText(f"Image: {width}×{height} pixels | Size: {file_size_mb:.1f} MB")
            logger.info(f"Successfully loaded image: {width}×{height} pixels")

        except Exception as e:
            logger.error(f"Error loading image: {e}")
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
        self.model.setData(df)
        self.content_stack.setCurrentWidget(self.table_view)
        self.set_ui_for_content_type('data')

        has_metadata = hasattr(df, 'attrs') and 'file_metadata' in df.attrs
        self.metadata_btn.setEnabled(has_metadata)

        info_text = f"Rows: {len(df)} | Columns: {len(df.columns)}"
        if has_metadata:
            metadata = df.attrs['file_metadata']
            key_info = [f"{key}: {metadata[key]}" for key in ['Date', 'Time', 'Field-Nr.', 'MagWalk', 'Samples'] if key in metadata]
            if key_info:
                info_text += f" | {' | '.join(key_info[:2])}"

        self.info_label.setText(info_text)

        self.column_combo.clear()
        self.column_combo.addItem("All columns")
        self.column_combo.addItems(df.columns.tolist())

        header = self.table_view.horizontalHeader()
        self.table_view.resizeColumnsToContents()
        available_width = self.table_view.viewport().width()
        column_count = len(df.columns)

        if column_count > 0:
            total_content_width = sum(header.sectionSize(i) for i in range(column_count))
            if total_content_width > available_width:
                header.setSectionResizeMode(QHeaderView.Stretch)
            else:
                header.setSectionResizeMode(QHeaderView.Interactive)
                for i in range(column_count):
                    current_size = header.sectionSize(i)
                    if current_size < 80:
                        header.resizeSection(i, 80)
                    elif current_size > 250:
                        header.resizeSection(i, 250)
                header.setStretchLastSection(True)

    def set_figure(self, figure: Figure, title: str):
        """Set a figure directly, for plots generated by scripts."""
        self.plot_widget.set_figure(figure)
        self.content_stack.setCurrentWidget(self.plot_widget)
        self.set_ui_for_content_type('plot')
        self.info_label.setText(f"Interactive Plot: {title}")
        self.current_file = None
        self.tab_title_changed.emit(title)

    def filter_columns(self, column):
        """Filter displayed columns"""
        if column == "All columns" or not column:
            for i in range(self.model.columnCount()):
                self.table_view.showColumn(i)
        else:
            df = self.model.get_dataframe()
            for i, col in enumerate(df.columns):
                self.table_view.setColumnHidden(i, col != column)

    def search_data(self, text):
        """Search for text in data"""
        logger.debug(f"Searching for: {text}")

    def show_statistics(self):
        """Show statistics for the current data"""
        df = self.model.get_dataframe()
        if df.empty:
            return

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            QMessageBox.information(self, "Statistics", "No numeric columns found.")
            return

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
        """Export the currently displayed tabular data"""
        df = self.model.get_dataframe()
        if df.empty:
            QMessageBox.warning(self, "Export", "No data to export.")
            return

        # Get export file path
        file_dialog = QFileDialog()
        file_dialog.setAcceptMode(QFileDialog.AcceptSave)
        file_dialog.setNameFilter("CSV files (*.csv);;Excel files (*.xlsx);;JSON files (*.json);;All files (*)")
        file_dialog.setDefaultSuffix("csv")

        if self.current_file:
            # Suggest a filename based on the current file
            base_name = os.path.splitext(os.path.basename(self.current_file))[0]
            file_dialog.selectFile(f"{base_name}_exported.csv")

        if file_dialog.exec() == QFileDialog.Accepted:
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                export_path = selected_files[0]

                try:
                    if export_path.lower().endswith('.csv'):
                        df.to_csv(export_path, index=False)
                    elif export_path.lower().endswith('.xlsx'):
                        df.to_excel(export_path, index=False)
                    elif export_path.lower().endswith('.json'):
                        df.to_json(export_path, orient='records', indent=2)
                    else:
                        # Default to CSV
                        df.to_csv(export_path, index=False)

                    QMessageBox.information(self, "Export Complete", f"Data exported successfully to: {export_path}")
                    logger.info(f"Data exported to: {export_path}")

                except Exception as e:
                    QMessageBox.critical(self, "Export Failed", f"Failed to export data: {str(e)}")
                    logger.error(f"Export failed: {e}")

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
        """Handle wheel events for zooming on the image viewer."""
        if source == self.image_viewer.viewport() and event.type() == QEvent.Wheel:
            if event.modifiers() == Qt.ControlModifier:
                if event.angleDelta().y() > 0:
                    self.zoom_in()
                else:
                    self.zoom_out()
                return True  # Event handled
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
