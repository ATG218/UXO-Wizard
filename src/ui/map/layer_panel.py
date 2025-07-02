"""
Professional Layer Control Panel - QGIS-style layer management UI
"""

import os
os.environ["QT_API"] = "pyside6"

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QTreeWidget, QTreeWidgetItem,
    QMenu, QSlider, QLabel, QHBoxLayout, QPushButton,
    QCheckBox, QStyle, QStyledItemDelegate, QStyleOptionViewItem,
    QInputDialog, QFrame, QSpacerItem, QSizePolicy, QHeaderView
)
from qtpy.QtCore import Qt, Signal, QMimeData, QByteArray, QRect, QSize
from qtpy.QtGui import (
    QAction, QPainter, QColor, QBrush, QPen, QFont, QLinearGradient, 
    QIcon, QPixmap, QPalette, QFontMetrics
)
from typing import Dict, Optional
from loguru import logger

from .layer_manager import LayerManager
from .layer_types import UXOLayer, LayerStyle, LayerType


class QGISStyleDelegate(QStyledItemDelegate):
    """QGIS-style delegate for clean, professional layer items"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index):
        """Paint layer items with QGIS-style clean design"""
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Only custom paint column 0 (layer name). Let default paint handle checkbox column.
        if index.column() != 0:
            super().paint(painter, option, index)
            return

        # Get item data from column0
        item_data = index.data(Qt.UserRole)
        if not isinstance(item_data, dict):
            super().paint(painter, option, index)
            return

        rect = option.rect
        is_group = item_data.get('is_group', False)
        
        if is_group:
            self._paint_group_item(painter, option, index, item_data)
        else:
            self._paint_layer_item(painter, option, index, item_data)
            
    def _paint_group_item(self, painter: QPainter, option: QStyleOptionViewItem, index, item_data):
        """Paint group header in dark theme style"""
        rect = option.rect
        
        # Dark background for groups
        if option.state & QStyle.State_Selected:
            painter.fillRect(rect, QColor(13, 115, 119))  # #0d7377
        else:
            painter.fillRect(rect, QColor(60, 60, 60))  # #3c3c3c
            
        # Group text
        painter.setPen(QColor(255, 255, 255))  # White text
        font = QFont()
        font.setBold(True)
        font.setPointSize(9)
        painter.setFont(font)
        
        text_rect = rect.adjusted(20, 0, -5, 0)
        painter.drawText(text_rect, Qt.AlignVCenter | Qt.AlignLeft, index.data(Qt.DisplayRole))
        
    def _paint_layer_item(self, painter: QPainter, option: QStyleOptionViewItem, index, item_data):
        """Paint layer item in dark theme style"""
        rect = option.rect
        layer_type = item_data.get('layer_type', 'points')
        is_visible = item_data.get('is_visible', True)
        opacity = item_data.get('opacity', 1.0)
        is_selected = option.state & QStyle.State_Selected
        
        # Dark background
        if is_selected:
            painter.fillRect(rect, QColor(13, 115, 119))  # #0d7377
        elif not is_visible:
            painter.fillRect(rect, QColor(50, 50, 50))  # Darker gray for hidden
        else:
            painter.fillRect(rect, QColor(43, 43, 43))  # #2b2b2b
            
        # Layer type indicator (small colored square)
        indicator_size = 12
        indicator_x = rect.left() + 25
        indicator_y = rect.center().y() - indicator_size // 2
        indicator_rect = QRect(indicator_x, indicator_y, indicator_size, indicator_size)
        
        type_color = self._get_layer_type_color(layer_type)
        painter.fillRect(indicator_rect, type_color)
        painter.setPen(QPen(type_color.lighter(120), 1))
        painter.drawRect(indicator_rect)
        
        # Layer name
        text_color = QColor(255, 255, 255) if is_visible else QColor(150, 150, 150)
        painter.setPen(text_color)
        
        font = QFont()
        font.setPointSize(9)
        if is_selected:
            font.setBold(True)
        painter.setFont(font)
        
        text_rect = QRect(indicator_x + indicator_size + 8, rect.top(), 
                         rect.width() - (indicator_x + indicator_size + 30), rect.height())
        painter.drawText(text_rect, Qt.AlignVCenter | Qt.AlignLeft, index.data(Qt.DisplayRole))
        
        # Opacity indicator (if not 100%)
        if opacity < 1.0:
            self._draw_opacity_badge(painter, rect, opacity)
            
    def _get_layer_type_color(self, layer_type: str) -> QColor:
        """Get subtle color for layer type (QGIS-style)"""
        colors = {
            'points': QColor(228, 26, 28),       # Red
            'raster': QColor(55, 126, 184),      # Blue  
            'vector': QColor(77, 175, 74),       # Green
            'processed': QColor(152, 78, 163),   # Purple
            'annotation': QColor(255, 127, 0),   # Orange
        }
        return colors.get(layer_type, QColor(166, 166, 166))  # Gray default
        
    def _draw_opacity_badge(self, painter: QPainter, rect: QRect, opacity: float):
        """Draw small opacity percentage badge"""
        # Small badge in top right
        badge_text = f"{int(opacity * 100)}%"
        font = QFont()
        font.setPointSize(7)
        painter.setFont(font)
        
        metrics = QFontMetrics(font)
        text_width = metrics.horizontalAdvance(badge_text)
        
        badge_x = rect.right() - text_width - 8
        badge_y = rect.top() + 3
        badge_rect = QRect(badge_x - 2, badge_y, text_width + 4, 12)
        
        # Badge background
        painter.fillRect(badge_rect, QColor(100, 100, 100, 180))
        
        # Badge text
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(badge_rect, Qt.AlignCenter, badge_text)


class LayerControlPanel(QWidget):
    """Professional QGIS-style layer management panel"""
    
    # Signals
    layer_selected = Signal(str)  # layer_name
    visibility_changed = Signal(str, bool)  # layer_name, is_visible
    opacity_changed = Signal(str, float)  # layer_name, opacity
    style_edit_requested = Signal(str)  # layer_name
    zoom_to_layer = Signal(str)  # layer_name
    
    def __init__(self, layer_manager: LayerManager):
        super().__init__()
        self.layer_manager = layer_manager
        self.layer_items: Dict[str, QTreeWidgetItem] = {}
        self.group_items: Dict[str, QTreeWidgetItem] = {}
        
        self.setup_ui()
        self.setup_styling()
        self.connect_signals()
        
    def setup_ui(self):
        """Initialize the professional UI"""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Header
        self._create_header(layout)
        
        # Toolbar
        self._create_toolbar(layout)
        
        # Tree widget
        self._create_tree_widget(layout)
        
        # Opacity control
        self._create_opacity_control(layout)
        
        self.setLayout(layout)
        self._initialize_groups()
        
    def _create_header(self, layout):
        """Create clean header section"""
        header = QFrame()
        header.setFixedHeight(30)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(10, 5, 10, 5)
        
        # Title
        title_label = QLabel("Layers")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(10)
        title_label.setFont(title_font)
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        
        layout.addWidget(header)
        
    def _create_toolbar(self, layout):
        """Create clean toolbar"""
        toolbar = QFrame()
        toolbar.setFixedHeight(35)
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(5, 3, 5, 3)
        toolbar_layout.setSpacing(2)
        
        # Add Group button
        add_group_btn = self._create_toolbar_button("Add Group")
        add_group_btn.clicked.connect(self.add_group)
        toolbar_layout.addWidget(add_group_btn)
        
        # Remove button
        remove_btn = self._create_toolbar_button("Remove")
        remove_btn.clicked.connect(self.remove_selected)
        toolbar_layout.addWidget(remove_btn)
        
        toolbar_layout.addStretch()
        
        layout.addWidget(toolbar)
        
    def _create_toolbar_button(self, text: str) -> QPushButton:
        """Create a clean toolbar button"""
        btn = QPushButton(text)
        btn.setFixedHeight(24)
        btn.setFixedWidth(60)
        return btn
        
    def _create_tree_widget(self, layout):
        """Create QGIS-style tree widget"""
        self.tree_widget = QTreeWidget()
        
        # Headers
        self.tree_widget.setHeaderLabels(["Layer", "Vis"])
        header = self.tree_widget.header()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.Fixed)
        header.resizeSection(0, 160)
        header.resizeSection(1, 20)
        header.setDefaultSectionSize(20)
        
        # Tree styling
        self.tree_widget.setRootIsDecorated(True)
        self.tree_widget.setItemDelegate(QGISStyleDelegate())
        self.tree_widget.setAlternatingRowColors(False)
        self.tree_widget.setIndentation(15)
        self.tree_widget.setItemsExpandable(True)
        
        # Drag and drop
        self.tree_widget.setDragDropMode(QTreeWidget.InternalMove)
        self.tree_widget.setDefaultDropAction(Qt.MoveAction)
        
        # Context menu
        self.tree_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree_widget.customContextMenuRequested.connect(self.show_context_menu)
        
        # Interactions
        self.tree_widget.itemSelectionChanged.connect(self._on_selection_changed)
        self.tree_widget.itemChanged.connect(self._on_item_changed)
        self.tree_widget.itemDoubleClicked.connect(self._on_item_double_clicked)
        
        layout.addWidget(self.tree_widget)
        
    def _create_opacity_control(self, layout):
        """Create clean opacity control"""
        opacity_frame = QFrame()
        opacity_frame.setFixedHeight(50)
        opacity_layout = QVBoxLayout(opacity_frame)
        opacity_layout.setContentsMargins(10, 5, 10, 5)
        
        # Label row
        label_row = QHBoxLayout()
        opacity_label = QLabel("Layer opacity")
        opacity_label.setFont(QFont("", 8))
        label_row.addWidget(opacity_label)
        
        label_row.addStretch()
        
        self.opacity_value_label = QLabel("100%")
        self.opacity_value_label.setFont(QFont("", 8))
        label_row.addWidget(self.opacity_value_label)
        
        opacity_layout.addLayout(label_row)
        
        # Slider
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(100)
        self.opacity_slider.setEnabled(False)
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        opacity_layout.addWidget(self.opacity_slider)
        
        layout.addWidget(opacity_frame)
        
    def setup_styling(self):
        """Apply dark theme styling to match the rest of the application"""
        self.setStyleSheet("""
            LayerControlPanel {
                background-color: #2b2b2b;
                border: 1px solid #3c3c3c;
            }
            
            QFrame {
                background-color: #2b2b2b;
                border: none;
                border-bottom: 1px solid #3c3c3c;
            }
            
            QLabel {
                color: #ffffff;
                background: transparent;
            }
            
            QPushButton {
                background-color: #3c3c3c;
                color: #ffffff;
                border: 1px solid #4a4a4a;
                border-radius: 3px;
                padding: 2px 6px;
                font-size: 11px;
            }
            
            QPushButton:hover {
                background-color: #4a4a4a;
                border: 1px solid #0d7377;
            }
            
            QPushButton:pressed {
                background-color: #0d7377;
            }
            
            QTreeWidget {
                background-color: #2b2b2b;
                color: #ffffff;
                border: 1px solid #3c3c3c;
                outline: none;
                show-decoration-selected: 1;
                selection-background-color: #0d7377;
            }
            
            QTreeWidget::item {
                height: 22px;
                border: none;
                padding: 1px;
                color: #ffffff;
            }
            
            QTreeWidget::item:selected {
                background-color: #0d7377;
                border: none;
            }
            
            QTreeWidget::item:hover {
                background-color: #3c3c3c;
            }
            
            QTreeWidget::branch:has-siblings:!adjoins-item {
                border-image: none;
                border: none;
            }
            
            QTreeWidget::branch:has-siblings:adjoins-item {
                border-image: none;
                border: none;
            }
            
            QTreeWidget::branch:!has-children:!has-siblings:adjoins-item {
                border-image: none;
                border: none;
            }
            
            QSlider::groove:horizontal {
                height: 6px;
                background-color: #3c3c3c;
                border-radius: 3px;
            }
            
            QSlider::handle:horizontal {
                background-color: #0d7377;
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
                border: none;
            }
            
            QSlider::handle:horizontal:hover {
                background-color: #14ffec;
            }
            
            QSlider::sub-page:horizontal {
                background-color: #0d7377;
                border-radius: 3px;
            }
        """)
        
    def _initialize_groups(self):
        """Create initial layer groups"""
        for group_name in self.layer_manager.layer_groups.keys():
            self._create_group_item(group_name)
            
    def _create_group_item(self, group_name: str) -> QTreeWidgetItem:
        """Create a group item in the tree"""
        group_item = QTreeWidgetItem(self.tree_widget)
        group_item.setText(0, group_name)
        group_item.setFlags(group_item.flags() | Qt.ItemIsDropEnabled)
        group_item.setExpanded(True)
        
        # Store group metadata
        group_item.setData(0, Qt.UserRole, {'is_group': True, 'group_name': group_name})
        
        self.group_items[group_name] = group_item
        return group_item
        
    def connect_signals(self):
        """Connect layer manager signals"""
        self.layer_manager.layer_added.connect(self._on_layer_added)
        self.layer_manager.layer_removed.connect(self._on_layer_removed)
        self.layer_manager.layer_visibility_changed.connect(self._on_visibility_changed)
        self.layer_manager.layer_style_changed.connect(self._on_style_changed)
        
    def _on_layer_added(self, layer: UXOLayer):
        """Handle layer addition"""
        # Find appropriate group
        group_name = None
        for group, layers in self.layer_manager.layer_groups.items():
            if layer.name in layers:
                group_name = group
                break
                
        if not group_name:
            group_name = "Data Layers"
            
        # Get or create group item
        if group_name not in self.group_items:
            self._create_group_item(group_name)
            
        group_item = self.group_items[group_name]
        
        # Create layer item
        layer_item = QTreeWidgetItem(group_item)
        layer_item.setText(0, layer.name)
        layer_item.setText(1, "")
        layer_item.setFlags(
            layer_item.flags() | 
            Qt.ItemIsUserCheckable | 
            Qt.ItemIsDragEnabled |
            Qt.ItemIsSelectable
        )
        layer_item.setCheckState(1, Qt.Checked if layer.is_visible else Qt.Unchecked)
        
        # Store layer metadata
        layer_item.setData(0, Qt.UserRole, {
            'layer_name': layer.name,
            'opacity': layer.opacity,
            'layer_type': layer.layer_type.value,
            'is_visible': layer.is_visible,
            'source': layer.source.value,
            'row_count': getattr(layer.metadata, 'row_count', 0)
        })
        
        self.layer_items[layer.name] = layer_item
        
        # Expand group
        group_item.setExpanded(True)
        
        logger.debug(f"Added layer '{layer.name}' to layer panel")
        
    def _on_layer_removed(self, layer_name: str):
        """Handle layer removal"""
        if layer_name in self.layer_items:
            item = self.layer_items[layer_name]
            parent = item.parent()
            if parent:
                parent.removeChild(item)
            del self.layer_items[layer_name]
            logger.debug(f"Removed layer '{layer_name}' from layer panel")
            
    def _on_visibility_changed(self, layer_name: str, is_visible: bool):
        """Handle external visibility change"""
        if layer_name in self.layer_items:
            item = self.layer_items[layer_name]
            item.setCheckState(1, Qt.Checked if is_visible else Qt.Unchecked)
            
            # Update metadata
            data = item.data(0, Qt.UserRole)
            if data:
                data['is_visible'] = is_visible
                item.setData(0, Qt.UserRole, data)
            
    def _on_style_changed(self, layer_name: str, style: LayerStyle):
        """Handle style change"""
        if layer_name in self.layer_items:
            item = self.layer_items[layer_name]
            self.tree_widget.update(self.tree_widget.indexFromItem(item))
        
    def _on_selection_changed(self):
        """Handle selection change in tree"""
        items = self.tree_widget.selectedItems()
        if items:
            item = items[0]
            data = item.data(0, Qt.UserRole)
            if data and 'layer_name' in data:
                layer_name = data['layer_name']
                self.layer_selected.emit(layer_name)
                
                # Update opacity slider
                layer = self.layer_manager.get_layer(layer_name)
                if layer:
                    self.opacity_slider.setEnabled(True)
                    new_value = int(layer.opacity * 100)
                    self.opacity_slider.setValue(new_value)
                    self.opacity_value_label.setText(f"{new_value}%")
            else:
                self.opacity_slider.setEnabled(False)
                self.opacity_value_label.setText("--")
                
    def _on_item_changed(self, item: QTreeWidgetItem, column: int):
        """Handle item changes (checkbox state)"""
        if column == 1:
            data = item.data(0, Qt.UserRole)
            if data and 'layer_name' in data:
                layer_name = data['layer_name']
                is_visible = item.checkState(1) == Qt.Checked
                
                # Update metadata
                data['is_visible'] = is_visible
                item.setData(0, Qt.UserRole, data)
                
                # Update layer manager
                self.layer_manager.set_layer_visibility(layer_name, is_visible)
                self.visibility_changed.emit(layer_name, is_visible)
                
    def _on_item_double_clicked(self, item: QTreeWidgetItem, column: int):
        """Handle double-click to zoom to layer"""
        data = item.data(0, Qt.UserRole)
        if data and 'layer_name' in data:
            layer_name = data['layer_name']
            self.zoom_to_layer.emit(layer_name)
                
    def _on_opacity_changed(self, value: int):
        """Handle opacity slider change"""
        opacity = value / 100.0
        self.opacity_value_label.setText(f"{value}%")
        
        # Update selected layer
        items = self.tree_widget.selectedItems()
        if items:
            item = items[0]
            data = item.data(0, Qt.UserRole)
            if data and 'layer_name' in data:
                layer_name = data['layer_name']
                layer = self.layer_manager.get_layer(layer_name)
                if layer:
                    layer.opacity = opacity
                    data['opacity'] = opacity
                    item.setData(0, Qt.UserRole, data)
                    self.opacity_changed.emit(layer_name, opacity)
                    
                    # Force repaint
                    self.tree_widget.update(self.tree_widget.indexFromItem(item))
                    
    def show_context_menu(self, position):
        """Show context menu for layer items"""
        item = self.tree_widget.itemAt(position)
        if not item:
            return
            
        data = item.data(0, Qt.UserRole)
        if not data or 'layer_name' not in data:
            return  # Not a layer item
            
        layer_name = data['layer_name']
        
        menu = QMenu(self)
        
        # Zoom to layer
        zoom_action = QAction("Zoom to Layer", self)
        zoom_action.triggered.connect(lambda: self.zoom_to_layer.emit(layer_name))
        menu.addAction(zoom_action)
        
        # Properties
        props_action = QAction("Properties...", self)
        props_action.triggered.connect(lambda: self.style_edit_requested.emit(layer_name))
        menu.addAction(props_action)
        
        menu.addSeparator()
        
        # Remove layer
        remove_action = QAction("Remove Layer", self)
        remove_action.triggered.connect(lambda: self.layer_manager.remove_layer(layer_name))
        menu.addAction(remove_action)
        
        # Show menu
        menu.exec_(self.tree_widget.viewport().mapToGlobal(position))
        
    def add_group(self):
        """Add a new layer group"""
        name, ok = QInputDialog.getText(
            self, 
            "New Group", 
            "Enter group name:",
            text="New Group"
        )
        if ok and name:
            if name not in self.group_items:
                self._create_group_item(name)
                self.layer_manager.layer_groups[name] = []
                logger.info(f"Added new group: {name}")
                
    def remove_selected(self):
        """Remove selected layer"""
        items = self.tree_widget.selectedItems()
        if items:
            item = items[0]
            data = item.data(0, Qt.UserRole)
            if data and 'layer_name' in data:
                layer_name = data['layer_name']
                self.layer_manager.remove_layer(layer_name)


# Keep alias for backwards compatibility
ModernLayerControlPanel = LayerControlPanel 