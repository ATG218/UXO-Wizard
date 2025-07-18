"""
Layer Style Editor - Interactive dialog for editing layer visual properties
"""

import os
os.environ["QT_API"] = "pyside6"

from qtpy.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QSpinBox, 
    QDoubleSpinBox, QCheckBox, QComboBox, QPushButton, QColorDialog, 
    QFrame, QGroupBox, QSlider, QWidget, QDialogButtonBox
)
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor, QPalette, QFont
from typing import Optional
from loguru import logger

from .layer_types import UXOLayer, LayerStyle, LayerType
from .layer_manager import LayerManager


class ColorButton(QPushButton):
    """Custom button for color selection"""
    
    color_changed = Signal(QColor)
    
    def __init__(self, color: str = "#0066CC", parent=None):
        super().__init__(parent)
        self.current_color = QColor(color)
        self.setFixedSize(30, 30)
        self.clicked.connect(self.choose_color)
        self.update_color()
        
    def choose_color(self):
        """Open color picker dialog"""
        color = QColorDialog.getColor(self.current_color, self, "Choose Color")
        if color.isValid():
            self.current_color = color
            self.update_color()
            self.color_changed.emit(color)
            
    def update_color(self):
        """Update button appearance with current color"""
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.current_color.name()};
                border: 2px solid #333;
                border-radius: 4px;
            }}
            QPushButton:hover {{
                border-color: #555;
            }}
        """)
        
    def set_color(self, color: QColor):
        """Set color without triggering signal"""
        self.current_color = color
        self.update_color()


class LayerStyleEditor(QDialog):
    """Dialog for editing layer visual properties"""
    
    def __init__(self, layer: UXOLayer, layer_manager: LayerManager, parent=None):
        super().__init__(parent)
        self.layer = layer
        self.layer_manager = layer_manager
        self.style = layer.style
        
        self.setWindowTitle(f"Layer Properties - {layer.display_name or layer.name}")
        self.setModal(True)
        self.setMinimumSize(400, 500)
        
        self.setup_ui()
        self.load_current_style()
        
    def setup_ui(self):
        """Create the dialog UI"""
        layout = QVBoxLayout()
        layout.setSpacing(10)
        
        # Header
        header_layout = QHBoxLayout()
        title_label = QLabel(f"Style Properties")
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title_label.setFont(title_font)
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        
        layer_info = QLabel(f"Layer: {self.layer.display_name or self.layer.name}")
        layer_info.setStyleSheet("color: #666; font-size: 10px;")
        header_layout.addWidget(layer_info)
        
        layout.addLayout(header_layout)
        
        # Create relevant style sections based on layer type
        self.create_relevant_style_sections(layout)
        
        # Dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.Apply
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        button_box.button(QDialogButtonBox.Apply).clicked.connect(self.apply_changes)
        layout.addWidget(button_box)
        
        self.setLayout(layout)
        
    def create_relevant_style_sections(self, layout):
        """Create style sections relevant to the layer type"""
        # Always show opacity control
        self.create_opacity_section(layout)
        
        # Show layer-type specific controls
        if self.layer.layer_type == LayerType.POINTS:
            self.create_point_style_section(layout)
        elif self.layer.layer_type == LayerType.VECTOR:
            self.create_line_style_section(layout)
        elif self.layer.layer_type == LayerType.RASTER:
            # For raster layers, only show opacity (already added above)
            pass
        else:
            # For other types, show all sections
            self.create_point_style_section(layout)
            self.create_line_style_section(layout)
            self.create_fill_style_section(layout)
    
    def create_opacity_section(self, layout):
        """Create opacity control section"""
        group = QGroupBox("Opacity")
        grid = QGridLayout()
        
        # Layer opacity
        grid.addWidget(QLabel("Opacity:"), 0, 0)
        opacity_layout = QHBoxLayout()
        self.layer_opacity_slider = QSlider(Qt.Horizontal)
        self.layer_opacity_slider.setRange(0, 100)
        self.layer_opacity_slider.setValue(int(self.layer.opacity * 100))
        self.layer_opacity_label = QLabel(f"{int(self.layer.opacity * 100)}%")
        self.layer_opacity_label.setMinimumWidth(35)
        self.layer_opacity_slider.valueChanged.connect(
            lambda v: self.layer_opacity_label.setText(f"{v}%")
        )
        opacity_layout.addWidget(self.layer_opacity_slider)
        opacity_layout.addWidget(self.layer_opacity_label)
        grid.addLayout(opacity_layout, 0, 1)
        
        group.setLayout(grid)
        layout.addWidget(group)
        
    def create_point_style_section(self, layout):
        """Create point style controls"""
        group = QGroupBox("Point Style")
        grid = QGridLayout()
        
        # Point color
        grid.addWidget(QLabel("Color:"), 0, 0)
        self.point_color_btn = ColorButton(self.style.point_color)
        grid.addWidget(self.point_color_btn, 0, 1)
        
        # Point size
        grid.addWidget(QLabel("Size:"), 1, 0)
        self.point_size_spin = QSpinBox()
        self.point_size_spin.setRange(1, 50)
        self.point_size_spin.setValue(self.style.point_size)
        grid.addWidget(self.point_size_spin, 1, 1)
        
        # Point opacity
        grid.addWidget(QLabel("Opacity:"), 2, 0)
        opacity_layout = QHBoxLayout()
        self.point_opacity_slider = QSlider(Qt.Horizontal)
        self.point_opacity_slider.setRange(0, 100)
        self.point_opacity_slider.setValue(int(self.style.point_opacity * 100))
        self.point_opacity_label = QLabel(f"{int(self.style.point_opacity * 100)}%")
        self.point_opacity_label.setMinimumWidth(35)
        self.point_opacity_slider.valueChanged.connect(
            lambda v: self.point_opacity_label.setText(f"{v}%")
        )
        opacity_layout.addWidget(self.point_opacity_slider)
        opacity_layout.addWidget(self.point_opacity_label)
        grid.addLayout(opacity_layout, 2, 1)
        
        group.setLayout(grid)
        layout.addWidget(group)
        
    def create_line_style_section(self, layout):
        """Create line style controls"""
        group = QGroupBox("Line Style")
        grid = QGridLayout()
        
        # Line color
        grid.addWidget(QLabel("Color:"), 0, 0)
        self.line_color_btn = ColorButton(self.style.line_color)
        grid.addWidget(self.line_color_btn, 0, 1)
        
        # Line width
        grid.addWidget(QLabel("Width:"), 1, 0)
        self.line_width_spin = QSpinBox()
        self.line_width_spin.setRange(1, 20)
        self.line_width_spin.setValue(self.style.line_width)
        grid.addWidget(self.line_width_spin, 1, 1)
        
        group.setLayout(grid)
        layout.addWidget(group)
        
    def create_fill_style_section(self, layout):
        """Create fill style controls"""
        group = QGroupBox("Fill Style")
        grid = QGridLayout()
        
        # Fill color
        grid.addWidget(QLabel("Color:"), 0, 0)
        self.fill_color_btn = ColorButton(self.style.fill_color)
        grid.addWidget(self.fill_color_btn, 0, 1)
        
        # Fill opacity
        grid.addWidget(QLabel("Opacity:"), 1, 0)
        opacity_layout = QHBoxLayout()
        self.fill_opacity_slider = QSlider(Qt.Horizontal)
        self.fill_opacity_slider.setRange(0, 100)
        self.fill_opacity_slider.setValue(int(self.style.fill_opacity * 100))
        self.fill_opacity_label = QLabel(f"{int(self.style.fill_opacity * 100)}%")
        self.fill_opacity_label.setMinimumWidth(35)
        self.fill_opacity_slider.valueChanged.connect(
            lambda v: self.fill_opacity_label.setText(f"{v}%")
        )
        opacity_layout.addWidget(self.fill_opacity_slider)
        opacity_layout.addWidget(self.fill_opacity_label)
        grid.addLayout(opacity_layout, 1, 1)
        
        group.setLayout(grid)
        layout.addWidget(group)
        
    def load_current_style(self):
        """Load current style values into controls"""
        # Layer opacity (always present)
        if hasattr(self, 'layer_opacity_slider'):
            self.layer_opacity_slider.setValue(int(self.layer.opacity * 100))
        
        # Point style (if present)
        if hasattr(self, 'point_color_btn'):
            self.point_color_btn.set_color(QColor(self.style.point_color))
            self.point_size_spin.setValue(self.style.point_size)
            self.point_opacity_slider.setValue(int(self.style.point_opacity * 100))
        
        # Line style (if present)
        if hasattr(self, 'line_color_btn'):
            self.line_color_btn.set_color(QColor(self.style.line_color))
            self.line_width_spin.setValue(self.style.line_width)
        
        # Fill style (if present)
        if hasattr(self, 'fill_color_btn'):
            self.fill_color_btn.set_color(QColor(self.style.fill_color))
            self.fill_opacity_slider.setValue(int(self.style.fill_opacity * 100))
            
    def apply_changes(self):
        """Apply style changes to the layer"""
        # Update layer opacity
        if hasattr(self, 'layer_opacity_slider'):
            new_opacity = self.layer_opacity_slider.value() / 100.0
            self.layer_manager.set_layer_opacity(self.layer.name, new_opacity)
        
        # Create new style with updated values
        new_style = LayerStyle(
            # Point style
            point_color=self.point_color_btn.current_color.name() if hasattr(self, 'point_color_btn') else self.style.point_color,
            point_size=self.point_size_spin.value() if hasattr(self, 'point_size_spin') else self.style.point_size,
            point_opacity=self.point_opacity_slider.value() / 100.0 if hasattr(self, 'point_opacity_slider') else self.style.point_opacity,
            point_symbol=self.style.point_symbol,
            
            # Line style
            line_color=self.line_color_btn.current_color.name() if hasattr(self, 'line_color_btn') else self.style.line_color,
            line_width=self.line_width_spin.value() if hasattr(self, 'line_width_spin') else self.style.line_width,
            line_opacity=self.style.line_opacity,
            line_style=self.style.line_style,
            
            # Fill style
            fill_color=self.fill_color_btn.current_color.name() if hasattr(self, 'fill_color_btn') else self.style.fill_color,
            fill_opacity=self.fill_opacity_slider.value() / 100.0 if hasattr(self, 'fill_opacity_slider') else self.style.fill_opacity,
            
            # Keep existing values for other properties
            show_labels=self.style.show_labels,
            label_field=self.style.label_field,
            label_size=self.style.label_size,
            enable_clustering=self.style.enable_clustering,
            cluster_distance=self.style.cluster_distance,
            use_graduated_colors=self.style.use_graduated_colors,
            color_field=self.style.color_field,
            color_ramp=self.style.color_ramp
        )
        
        # Apply the style through the layer manager
        self.layer_manager.set_layer_style(self.layer.name, new_style)
        logger.info(f"Applied style changes to layer '{self.layer.name}'")
        
    def accept(self):
        """Accept dialog and apply changes"""
        self.apply_changes()
        super().accept()