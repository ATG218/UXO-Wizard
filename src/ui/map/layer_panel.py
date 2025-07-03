"""
Modern Layer Control Panel - Clean, simple, and functional design
"""

import os
os.environ["QT_API"] = "pyside6"

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QLabel, QPushButton, QSlider, QCheckBox, QFrame, QSizePolicy,
    QInputDialog, QMenu, QSpacerItem, QToolButton, QGroupBox,
    QScrollArea, QStackedWidget
)
from qtpy.QtCore import Qt, Signal, QSize, QTimer
from qtpy.QtGui import QFont, QIcon, QPixmap, QPainter, QColor, QBrush
from typing import Dict, Optional, List
from loguru import logger

from .layer_manager import LayerManager
from .layer_types import UXOLayer, LayerStyle, LayerType


class LayerItemWidget(QWidget):
    """Modern layer item widget with clean design"""
    
    visibility_changed = Signal(str, bool)  # layer_name, visible
    layer_selected = Signal(str)  # layer_name
    layer_double_clicked = Signal(str)  # layer_name
    
    def __init__(self, layer: UXOLayer):
        super().__init__()
        self.layer = layer
        self.setup_ui()
        
    def setup_ui(self):
        """Create modern layer item UI"""
        layout = QHBoxLayout()
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)
        
        # Visibility toggle icon (replaces both checkbox and type indicator)
        self.visibility_icon = QPushButton()
        self.visibility_icon.setFixedSize(8, 8)
        self.visibility_icon.setFlat(True)
        self.visibility_icon.clicked.connect(self._toggle_visibility)
        self._update_visibility_icon()
        layout.addWidget(self.visibility_icon)
        
        # Layer name and info
        info_layout = QVBoxLayout()
        info_layout.setSpacing(2)
        
        # Layer name
        self.name_label = QLabel(self.layer.name)
        name_font = QFont()
        name_font.setPointSize(9)
        name_font.setBold(True)
        self.name_label.setFont(name_font)
        info_layout.addWidget(self.name_label)
        
        # Layer details
        details = self._get_layer_details()
        self.details_label = QLabel(details)
        details_font = QFont()
        details_font.setPointSize(8)
        self.details_label.setFont(details_font)
        self.details_label.setStyleSheet("color: #999999;")
        info_layout.addWidget(self.details_label)
        
        layout.addLayout(info_layout)
        layout.addStretch()
        
        # Opacity indicator
        if self.layer.opacity < 1.0:
            opacity_label = QLabel(f"{int(self.layer.opacity * 100)}%")
            opacity_label.setStyleSheet("color: #999999; font-size: 8px;")
            layout.addWidget(opacity_label)
        
        self.setLayout(layout)
        self.setMinimumHeight(40)
        
        # Click handling
        self.mousePressEvent = self._on_click
        self.mouseDoubleClickEvent = self._on_double_click
        
    def _update_visibility_icon(self):
        """Update the visibility icon based on layer state"""
        if self.layer.is_visible:
            # Green indicator for visible layer
            self.visibility_icon.setText("")
            self.visibility_icon.setToolTip("Layer is visible - click to hide")
            self.visibility_icon.setStyleSheet("""
                QPushButton {
                    background-color: #28a745;
                    border: 1px solid #20a83a;
                    border-radius: 4px;
                    min-width: 6px;
                    min-height: 6px;
                }
                QPushButton:hover {
                    background-color: #34ce57;
                    border-color: #28a745;
                }
                QPushButton:pressed {
                    background-color: #1e7e34;
                }
            """)
        else:
            # Red indicator for hidden layer
            self.visibility_icon.setText("")
            self.visibility_icon.setToolTip("Layer is hidden - click to show")
            self.visibility_icon.setStyleSheet("""
                QPushButton {
                    background-color: #dc3545;
                    border: 1px solid #c82333;
                    border-radius: 4px;
                    min-width: 6px;
                    min-height: 6px;
                }
                QPushButton:hover {
                    background-color: #e85a67;
                    border-color: #dc3545;
                }
                QPushButton:pressed {
                    background-color: #bd2130;
                }
            """)
            
    def _toggle_visibility(self):
        """Toggle layer visibility"""
        new_visibility = not self.layer.is_visible
        # Emit signal to let the layer manager handle the change
        # This maintains the proper signal chain: widget -> layer panel -> layer manager -> map widget
        self.visibility_changed.emit(self.layer.name, new_visibility)
            
    def _get_layer_color(self) -> QColor:
        """Get color for layer type"""
        colors = {
            LayerType.POINTS: QColor(52, 152, 219),      # Blue
            LayerType.RASTER: QColor(46, 204, 113),      # Green
            LayerType.VECTOR: QColor(155, 89, 182),      # Purple
            LayerType.PROCESSED: QColor(230, 126, 34),   # Orange
            LayerType.ANNOTATION: QColor(231, 76, 60),   # Red
        }
        return colors.get(self.layer.layer_type, QColor(149, 165, 166))
        
    def _get_layer_details(self) -> str:
        """Get layer details string"""
        details = []
        
        if hasattr(self.layer.metadata, 'row_count'):
            details.append(f"{self.layer.metadata['row_count']} items")
        elif isinstance(self.layer.metadata, dict) and 'row_count' in self.layer.metadata:
            details.append(f"{self.layer.metadata['row_count']} items")
            
        details.append(self.layer.layer_type.value.title())
        
        return " • ".join(details)
        
    def _on_click(self, event):
        """Handle single click"""
        self.layer_selected.emit(self.layer.name)
        
    def _on_double_click(self, event):
        """Handle double click"""
        self.layer_double_clicked.emit(self.layer.name)
        
    def update_opacity(self, opacity: float):
        """Update opacity display"""
        self.layer.opacity = opacity
        # Trigger UI refresh
        self.setup_ui()


class ModernLayerControlPanel(QWidget):
    """Modern, clean layer management panel"""
    
    # Signals
    layer_selected = Signal(str)  # layer_name
    visibility_changed = Signal(str, bool)  # layer_name, is_visible
    opacity_changed = Signal(str, float)  # layer_name, opacity
    style_edit_requested = Signal(str)  # layer_name
    zoom_to_layer = Signal(str)  # layer_name
    
    def __init__(self, layer_manager: LayerManager):
        super().__init__()
        self.layer_manager = layer_manager
        self.layer_widgets: Dict[str, LayerItemWidget] = {}
        self.current_layer = None
        
        self.setup_ui()
        self.connect_signals()
        
    def setup_ui(self):
        """Initialize the modern UI"""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Header with title and controls
        self._create_header(layout)
        
        # Layer list
        self._create_layer_list(layout)
        
        # Controls section
        self._create_controls(layout)
        
        self.setLayout(layout)
        
    def _create_header(self, layout):
        """Create modern header"""
        header = QFrame()
        header.setFixedHeight(40)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(12, 8, 12, 8)
        
        # Title
        title = QLabel("Layers")
        title_font = QFont()
        title_font.setPointSize(11)
        title_font.setBold(True)
        title.setFont(title_font)
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        # Add button
        self.add_btn = QToolButton()
        self.add_btn.setText("+")
        self.add_btn.setFixedSize(24, 24)
        self.add_btn.setToolTip("Add Group")
        self.add_btn.clicked.connect(self.add_group)
        header_layout.addWidget(self.add_btn)
        
        # Remove button
        self.remove_btn = QToolButton()
        self.remove_btn.setText("−")
        self.remove_btn.setFixedSize(24, 24)
        self.remove_btn.setToolTip("Remove Selected Layer")
        self.remove_btn.setEnabled(False)
        self.remove_btn.clicked.connect(self.remove_selected)
        header_layout.addWidget(self.remove_btn)
        
        layout.addWidget(header)
        
    def _create_layer_list(self, layout):
        """Create scrollable layer list"""
        # Scroll area for layers
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Container for layer widgets
        self.layers_container = QWidget()
        self.layers_layout = QVBoxLayout(self.layers_container)
        self.layers_layout.setContentsMargins(0, 0, 0, 0)
        self.layers_layout.setSpacing(1)
        self.layers_layout.addStretch()  # Push items to top
        
        scroll.setWidget(self.layers_container)
        layout.addWidget(scroll)
        
    def _create_controls(self, layout):
        """Create opacity and other controls"""
        controls = QFrame()
        controls.setFixedHeight(80)
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(12, 8, 12, 8)
        controls_layout.setSpacing(8)
        
        # Opacity control
        opacity_group = QGroupBox("Layer Opacity")
        opacity_layout = QVBoxLayout(opacity_group)
        opacity_layout.setContentsMargins(8, 8, 8, 8)
        
        # Opacity slider with label
        slider_layout = QHBoxLayout()
        
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(100)
        self.opacity_slider.setEnabled(False)
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        slider_layout.addWidget(self.opacity_slider)
        
        self.opacity_label = QLabel("100%")
        self.opacity_label.setFixedWidth(30)
        self.opacity_label.setAlignment(Qt.AlignRight)
        slider_layout.addWidget(self.opacity_label)
        
        opacity_layout.addLayout(slider_layout)
        controls_layout.addWidget(opacity_group)
        
        layout.addWidget(controls)
        
    def connect_signals(self):
        """Connect layer manager signals"""
        self.layer_manager.layer_added.connect(self._on_layer_added)
        self.layer_manager.layer_removed.connect(self._on_layer_removed)
        self.layer_manager.layer_visibility_changed.connect(self._on_visibility_changed)
        self.layer_manager.layer_style_changed.connect(self._on_style_changed)
        
    def _on_layer_added(self, layer: UXOLayer):
        """Handle layer addition"""
        if layer.name in self.layer_widgets:
            return  # Already exists
            
        # Create layer widget
        layer_widget = LayerItemWidget(layer)
        layer_widget.visibility_changed.connect(self._on_layer_visibility_toggled)
        layer_widget.layer_selected.connect(self._on_layer_selected)
        layer_widget.layer_double_clicked.connect(self.zoom_to_layer.emit)
        
        # Add to layout (insert before stretch)
        self.layers_layout.insertWidget(self.layers_layout.count() - 1, layer_widget)
        self.layer_widgets[layer.name] = layer_widget
        
        # Context menu
        layer_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        layer_widget.customContextMenuRequested.connect(
            lambda pos, name=layer.name: self._show_context_menu(pos, name, layer_widget)
        )
        
        logger.debug(f"Added layer widget for '{layer.name}'")
        
    def _on_layer_removed(self, layer_name: str):
        """Handle layer removal"""
        if layer_name in self.layer_widgets:
            widget = self.layer_widgets[layer_name]
            self.layers_layout.removeWidget(widget)
            widget.deleteLater()
            del self.layer_widgets[layer_name]
            
            # Clear selection if this was selected
            if self.current_layer == layer_name:
                self.current_layer = None
                self.opacity_slider.setEnabled(False)
                self.opacity_label.setText("--")
                self.remove_btn.setEnabled(False)
                
            logger.debug(f"Removed layer widget for '{layer_name}'")
            
    def _on_visibility_changed(self, layer_name: str, is_visible: bool):
        """Handle external visibility change"""
        if layer_name in self.layer_widgets:
            widget = self.layer_widgets[layer_name]
            widget.layer.is_visible = is_visible
            widget._update_visibility_icon()
            
            # Update visual style for the whole widget
            if is_visible:
                widget.setStyleSheet("")
            else:
                widget.setStyleSheet("QWidget { color: #666666; }")
            
    def _on_style_changed(self, layer_name: str, style: LayerStyle):
        """Handle style change"""
        if layer_name in self.layer_widgets:
            widget = self.layer_widgets[layer_name]
            widget.setup_ui()  # Refresh the widget
        
    def _on_layer_visibility_toggled(self, layer_name: str, is_visible: bool):
        """Handle layer visibility toggle from widget"""
        self.layer_manager.set_layer_visibility(layer_name, is_visible)
        self.visibility_changed.emit(layer_name, is_visible)
        
    def _on_layer_selected(self, layer_name: str):
        """Handle layer selection"""
        self.current_layer = layer_name
        self.layer_selected.emit(layer_name)
        self.remove_btn.setEnabled(True)
        
        # Update opacity slider
        layer = self.layer_manager.get_layer(layer_name)
        if layer:
            self.opacity_slider.setEnabled(True)
            opacity_value = int(layer.opacity * 100)
            self.opacity_slider.setValue(opacity_value)
            self.opacity_label.setText(f"{opacity_value}%")
            
        # Visual feedback
        for name, widget in self.layer_widgets.items():
            if name == layer_name:
                widget.setStyleSheet("""
                    LayerItemWidget {
                        background-color: #1a4f52;
                        border: 2px solid #0d7377;
                    }
                """)
            else:
                widget.setStyleSheet("")
                
    def _on_opacity_changed(self, value: int):
        """Handle opacity slider change"""
        opacity = value / 100.0
        self.opacity_label.setText(f"{value}%")
        
        if self.current_layer:
            # Update layer manager (this will emit the signal)
            self.layer_manager.set_layer_opacity(self.current_layer, opacity)
            
            # Update widget display
            if self.current_layer in self.layer_widgets:
                self.layer_widgets[self.current_layer].update_opacity(opacity)
                    
            # Emit signal for map update
            self.opacity_changed.emit(self.current_layer, opacity)
                    
    def _show_context_menu(self, position, layer_name: str, widget: LayerItemWidget):
        """Show context menu for layer"""
        menu = QMenu(self)
        
        # Zoom to layer
        zoom_action = menu.addAction("Zoom to Layer")
        zoom_action.triggered.connect(lambda: self.zoom_to_layer.emit(layer_name))
        
        # Properties
        props_action = menu.addAction("Properties...")
        props_action.triggered.connect(lambda: self.style_edit_requested.emit(layer_name))
        
        menu.addSeparator()
        
        # Remove layer
        remove_action = menu.addAction("Remove Layer")
        remove_action.triggered.connect(lambda: self.layer_manager.remove_layer(layer_name))
        
        # Show menu
        menu.exec_(widget.mapToGlobal(position))
        
    def add_group(self):
        """Add a new layer group (placeholder)"""
        name, ok = QInputDialog.getText(
            self, 
            "New Group", 
            "Enter group name:",
            text="New Group"
        )
        if ok and name:
            # For now, just add to layer manager
            self.layer_manager.layer_groups[name] = []
            logger.info(f"Added new group: {name}")
            
    def remove_selected(self):
        """Remove selected layer"""
        if self.current_layer:
            self.layer_manager.remove_layer(self.current_layer)


# Keep backwards compatibility
LayerControlPanel = ModernLayerControlPanel 