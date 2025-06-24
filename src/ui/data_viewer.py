"""
Data Viewer Widget for UXO Wizard - Display and analyze tabular data
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QTableView, QToolBar, 
    QComboBox, QToolButton, QLabel, QLineEdit,
    QHeaderView, QMenu, QMessageBox
)
from PySide6.QtGui import QAction
from PySide6.QtCore import Qt, Signal, QAbstractTableModel, QModelIndex
from PySide6.QtGui import QKeySequence
import pandas as pd
import numpy as np
from loguru import logger


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


class DataViewer(QWidget):
    """Widget for viewing and analyzing tabular data"""
    
    # Signals
    data_selected = Signal(object)  # Emits selected data
    
    def __init__(self):
        super().__init__()
        self.current_file = None
        self.setup_ui()
        
    def setup_ui(self):
        """Initialize the UI"""
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
        
        # Export button
        self.export_btn = QToolButton()
        self.export_btn.setText("Export")
        self.export_btn.clicked.connect(self.export_data)
        toolbar.addWidget(self.export_btn)
        
        # Table view
        self.table_view = QTableView()
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
        
    def load_data(self, filepath):
        """Load data from file"""
        try:
            # Determine file type and load accordingly
            if filepath.endswith('.csv'):
                df = pd.read_csv(filepath)
            elif filepath.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(filepath)
            elif filepath.endswith('.json'):
                df = pd.read_json(filepath)
            else:
                logger.error(f"Unsupported file type: {filepath}")
                return
                
            self.set_dataframe(df)
            self.current_file = filepath
            logger.info(f"Loaded data from: {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to load data:\n{str(e)}")
            
    def set_dataframe(self, df):
        """Set the DataFrame to display"""
        self.model.setData(df)
        
        # Update UI
        self.info_label.setText(f"Rows: {len(df)} | Columns: {len(df.columns)}")
        
        # Update column combo
        self.column_combo.clear()
        self.column_combo.addItem("All columns")
        self.column_combo.addItems(df.columns.tolist())
        
        # Auto-resize columns
        self.table_view.resizeColumnsToContents()
        
        # Limit column width
        header = self.table_view.horizontalHeader()
        for i in range(len(df.columns)):
            if header.sectionSize(i) > 200:
                header.resizeSection(i, 200)
                
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