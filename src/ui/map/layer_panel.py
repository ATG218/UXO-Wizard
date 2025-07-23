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
from qtpy.QtGui import QFont, QIcon, QPixmap, QPainter, QColor, QBrush, QAction
from typing import Dict, Optional, List
from loguru import logger

from .layer_manager import LayerManager
from .layer_types import UXOLayer, LayerStyle, LayerType


class LayerNameParser:
    """Utility to parse human-readable names from layer metadata."""
    @staticmethod
    def parse(layer: UXOLayer) -> str:
        """Generates a clean, descriptive name for a layer."""
        # First try to use metadata for more intelligent naming
        if hasattr(layer, 'metadata') and layer.metadata:
            data_type = layer.metadata.get('data_type', '').lower()
            if data_type:
                # Use data type for naming
                if 'anomaly' in data_type or 'anomalies' in data_type:
                    return "Anomalies"
                elif 'grid' in data_type or 'interpolated' in data_type:
                    return "Interpolated Grid"
                elif 'contour' in data_type:
                    return "Contours"
                elif 'flight' in data_type or 'path' in data_type:
                    return "Flight Path"
                elif 'segment' in data_type:
                    return "Segments"
                elif 'processed' in data_type:
                    return "Processed Data"
        
        # Try to infer from source script
        if layer.source_script:
            script_name = os.path.basename(layer.source_script).lower()
            if "magbase" in script_name:
                return "Magbase Data"
            elif "grid_interpolator" in script_name:
                return "Interpolated Grid"
            elif "anomaly" in script_name:
                return "Anomalies"
            elif "flight_path_segmenter" in script_name:
                return "Flight Segments"
            elif "basic_processing" in script_name:
                return "Basic Processing"
        
        # Try to infer from layer name patterns
        name_lower = layer.name.lower()
        if any(word in name_lower for word in ['anomaly', 'anomalies']):
            return "Anomalies"
        elif any(word in name_lower for word in ['grid', 'interpolated', 'interp']):
            return "Interpolated Grid"
        elif any(word in name_lower for word in ['contour', 'contours']):
            return "Contours"
        elif any(word in name_lower for word in ['flight', 'path', 'segment']):
            return "Flight Data"
        elif any(word in name_lower for word in ['processed', 'proc']):
            return "Processed Data"
        elif any(word in name_lower for word in ['magbase', 'magnetic']):
            return "Magnetic Data"
        
        # Fallback to a cleaned-up version of the original name
        name = layer.name.replace("_", " ").replace(".csv", "").title()
        
        # Remove common timestamp patterns
        import re
        # Remove patterns like "20240115 143000" or "20240115_143000"
        name = re.sub(r'\d{8}[\s_]\d{6}', '', name)
        # Remove patterns like "20240115143000"
        name = re.sub(r'\d{14}', '', name)
        # Remove "Output" suffix if present
        name = re.sub(r'\bOutput\b', '', name, flags=re.IGNORECASE)
        
        # Clean up extra spaces
        name = ' '.join(name.split())
        
        return name if name else "Layer Data"

class ScriptGroupWidget(QFrame):
    """A widget to represent a collapsible group of layers from a specific script."""
    def __init__(self, script_name: str, parent=None):
        super().__init__(parent)
        self.script_name = script_name
        self.is_expanded = True
        self.run_subgroups: List['SubgroupItemWidget'] = []
        self.layer_widgets: List['LayerItemWidget'] = []

        self.setFrameShape(QFrame.NoFrame)
        self.setObjectName("scriptGroupItem")
        self.setStyleSheet("""
            #scriptGroupItem {
                background-color: #2c3e50;
                border-top: 1px solid #34495e;
                margin-left: 5px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.header_widget = QWidget()
        self.header_widget.setObjectName("scriptGroupHeader")
        self.header_widget.setStyleSheet("""
            #scriptGroupHeader { background-color: #3498db; }
            #scriptGroupHeader:hover { background-color: #2980b9; }
        """)
        self.header_widget.mousePressEvent = self._toggle_expansion
        
        # Add context menu support
        self.header_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.header_widget.customContextMenuRequested.connect(self._show_script_group_context_menu)

        header_layout = QHBoxLayout(self.header_widget)
        header_layout.setContentsMargins(8, 5, 8, 5)
        header_layout.setSpacing(6)

        self.arrow_label = QLabel()
        self.arrow_label.setFixedWidth(10)

        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(9)  # Smaller font size
        self.title_label = QLabel(self._format_script_name(script_name))
        self.title_label.setFont(title_font)
        self.title_label.setStyleSheet("color: white;")
        
        self.item_count_label = QLabel("")
        self.item_count_label.setStyleSheet("color: #ecf0f1;")

        header_layout.addWidget(self.arrow_label)
        header_layout.addWidget(self.title_label)
        header_layout.addStretch()
        header_layout.addWidget(self.item_count_label)

        self.container = QWidget()
        self.container_layout = QVBoxLayout(self.container)
        self.container_layout.setContentsMargins(0, 0, 0, 0)
        self.container_layout.setSpacing(1)

        layout.addWidget(self.header_widget)
        layout.addWidget(self.container)

        self._update_arrow()
        self._update_item_count()

    def _format_script_name(self, script_name: str) -> str:
        """Format script name for display"""
        if not script_name or script_name == "Unknown Script":
            return "Unknown Script"
        
        # Extract just the filename without extension
        base_name = os.path.basename(script_name)
        if base_name.endswith('.py'):
            base_name = base_name[:-3]
        
        # Convert snake_case to Title Case
        formatted = base_name.replace('_', ' ').title()
        
        # Special cases for common script names
        replacements = {
            'Magbase Processing': 'Magbase Processing',
            'Anomaly Detector': 'Anomaly Detection',
            'Flight Path Segmenter': 'Flight Path Segmentation',
            'Grid Interpolator': 'Grid Interpolation',
            'Basic Processing': 'Basic Processing'
        }
        
        return replacements.get(formatted, formatted)
    
    def _show_script_group_context_menu(self, position):
        """Show context menu for script group"""
        menu = QMenu()
        
        # Remove script group action
        remove_action = menu.addAction("Remove Script Group")
        remove_action.triggered.connect(self._remove_script_group)
        
        # Show the menu
        menu.exec(self.header_widget.mapToGlobal(position))
    
    def _remove_script_group(self):
        """Remove this script group and all its layers"""
        # Emit a signal to the parent to handle removal
        parent = self.parent()
        while parent and not hasattr(parent, 'remove_script_group'):
            parent = parent.parent()
        
        if parent and hasattr(parent, 'remove_script_group'):
            parent.remove_script_group(self)

    def _toggle_expansion(self, event=None):
        # Only toggle on left-click, not right-click
        if event and hasattr(event, 'button') and event.button() != Qt.LeftButton:
            return
        self.is_expanded = not self.is_expanded
        self.container.setVisible(self.is_expanded)
        self._update_arrow()

    def _update_arrow(self):
        self.arrow_label.setText("▼" if self.is_expanded else "►")
        self.arrow_label.setStyleSheet("color: white;")
        
    def _update_item_count(self):
        run_count = len(self.run_subgroups)
        layer_count = len(self.layer_widgets)
        total_layers = layer_count + sum(len(run.layer_widgets) for run in self.run_subgroups)
        
        if run_count > 0:
            self.item_count_label.setText(f"({run_count} runs, {total_layers} layers)")
        else:
            self.item_count_label.setText(f"({total_layers} layers)")

    def add_run_subgroup(self, widget: 'SubgroupItemWidget'):
        self.run_subgroups.append(widget)
        self.container_layout.addWidget(widget)
        self._update_item_count()
        self._update_parent_counts()

    def add_layer_item(self, widget: 'LayerItemWidget'):
        """Add layer directly to script group (for layers without run_id)"""
        self.layer_widgets.append(widget)
        self.container_layout.addWidget(widget)
        self._update_item_count()
        self._update_parent_counts()

    def remove_run_subgroup(self, widget: 'SubgroupItemWidget'):
        if widget in self.run_subgroups:
            self.run_subgroups.remove(widget)
            self.container_layout.removeWidget(widget)
            widget.setParent(None)
            self._update_item_count()
            self._update_parent_counts()

    def remove_layer_item(self, widget: 'LayerItemWidget'):
        if widget in self.layer_widgets:
            self.layer_widgets.remove(widget)
            self.container_layout.removeWidget(widget)
            widget.setParent(None)
            self._update_item_count()
            self._update_parent_counts()
    
    def _update_parent_counts(self):
        """Update parent group counts"""
        parent = self.parent()
        while parent:
            if hasattr(parent, '_update_item_count'):
                parent._update_item_count()
                break
            parent = parent.parent()


class SubgroupItemWidget(QFrame):
    """A widget to represent a collapsible subgroup of layers from a single processing run."""
    def __init__(self, run_id: str, script_name: str, parent=None):
        super().__init__(parent)
        self.run_id = run_id
        self.script_name = script_name
        self.is_expanded = True
        self.layer_widgets: List['LayerItemWidget'] = []

        self.setFrameShape(QFrame.NoFrame)
        self.setObjectName("subgroupItem")
        self.setStyleSheet("""
            #subgroupItem {
                background-color: #34495e;
                border-top: 1px solid #2c3e50;
                margin-left: 15px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.header_widget = QWidget()
        self.header_widget.setObjectName("subgroupHeader")
        self.header_widget.setStyleSheet("""
            #subgroupHeader { background-color: #2c3e50; }
            #subgroupHeader:hover { background-color: #34495e; }
        """)
        self.header_widget.mousePressEvent = self._toggle_expansion
        self.header_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.header_widget.customContextMenuRequested.connect(self._show_context_menu)

        header_layout = QHBoxLayout(self.header_widget)
        header_layout.setContentsMargins(8, 4, 8, 4)
        header_layout.setSpacing(6)

        self.arrow_label = QLabel()
        self.arrow_label.setFixedWidth(10)

        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(8)  # Smaller font size
        # Format run_id for display - show timestamp more clearly
        display_run_id = self._format_run_id(run_id)
        self.title_label = QLabel(f"Run: {display_run_id}")
        self.title_label.setFont(title_font)
        
        self.item_count_label = QLabel("")
        self.item_count_label.setStyleSheet("color: #bdc3c7;")

        header_layout.addWidget(self.arrow_label)
        header_layout.addWidget(self.title_label)
        header_layout.addStretch()
        header_layout.addWidget(self.item_count_label)

        self.layer_container = QWidget()
        self.layer_container_layout = QVBoxLayout(self.layer_container)
        self.layer_container_layout.setContentsMargins(5, 0, 0, 0)
        self.layer_container_layout.setSpacing(1)

        layout.addWidget(self.header_widget)
        layout.addWidget(self.layer_container)

        self._update_arrow()
        self._update_item_count()

    def _format_run_id(self, run_id: str) -> str:
        """Format run_id for better display"""
        if not run_id:
            return "Unknown"
        
        # If it looks like a timestamp (YYYYMMDD_HHMMSS), format it nicely
        if len(run_id) == 15 and '_' in run_id:
            date_part, time_part = run_id.split('_')
            if len(date_part) == 8 and len(time_part) == 6:
                try:
                    formatted_date = f"{date_part[6:8]}/{date_part[4:6]}/{date_part[0:4]}"
                    formatted_time = f"{time_part[0:2]}:{time_part[2:4]}:{time_part[4:6]}"
                    return f"{formatted_date} {formatted_time}"
                except:
                    pass
        
        return run_id

    def _toggle_expansion(self, event=None):
        # Only toggle on left-click, not right-click
        if event and hasattr(event, 'button') and event.button() != Qt.LeftButton:
            return
        self.is_expanded = not self.is_expanded
        self.layer_container.setVisible(self.is_expanded)
        self._update_arrow()

    def _update_arrow(self):
        self.arrow_label.setText("▼" if self.is_expanded else "►")
        
    def _update_item_count(self):
        count = len(self.layer_widgets)
        self.item_count_label.setText(f"({count} layers)")

    def add_layer_item(self, widget: 'LayerItemWidget'):
        self.layer_widgets.append(widget)
        self.layer_container_layout.addWidget(widget)
        self._update_item_count()
        self._update_parent_counts()

    def remove_layer_item(self, widget: 'LayerItemWidget'):
        if widget in self.layer_widgets:
            self.layer_widgets.remove(widget)
            self.layer_container_layout.removeWidget(widget)
            widget.setParent(None)
            self._update_item_count()
            self._update_parent_counts()
    
    def _show_context_menu(self, position):
        """Show context menu for the run group"""
        menu = QMenu(self)
        
        # Delete this run group
        delete_action = QAction("Delete Run Group", self)
        delete_action.triggered.connect(self._delete_run_group)
        menu.addAction(delete_action)
        
        # Show menu at the requested position
        menu.exec(self.header_widget.mapToGlobal(position))
    
    def _delete_run_group(self):
        """Delete this entire run group and all its layers"""
        from qtpy.QtWidgets import QMessageBox
        
        # Confirm deletion
        reply = QMessageBox.question(
            self,
            "Delete Run Group",
            f"Are you sure you want to delete the run group '{self._format_run_id(self.run_id)}' and all its layers?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Find the layer panel to access the layer manager
            layer_panel = self
            while layer_panel and not hasattr(layer_panel, 'layer_manager'):
                layer_panel = layer_panel.parent()
            
            if layer_panel and hasattr(layer_panel, 'layer_manager'):
                # Remove all layers in this run group
                layer_names_to_remove = []
                for layer_widget in self.layer_widgets:
                    layer_names_to_remove.append(layer_widget.layer.name)
                
                # Remove each layer
                for layer_name in layer_names_to_remove:
                    layer_panel.layer_manager.remove_layer(layer_name)
                
                # The widget will be automatically removed when the layers are removed
                # and the UI is refreshed
    
    def _update_parent_counts(self):
        """Update parent group counts"""
        parent = self.parent()
        while parent:
            if hasattr(parent, '_update_item_count'):
                parent._update_item_count()
                if hasattr(parent, '_update_parent_counts'):
                    parent._update_parent_counts()
                break
            parent = parent.parent()



class GroupItemWidget(QFrame):
    """A widget to represent a collapsible group of layers."""
    def __init__(self, group_name: str, parent=None):
        super().__init__(parent)
        self.group_name = group_name
        self.is_expanded = True
        self.script_groups: List[ScriptGroupWidget] = []
        self.layer_widgets: List[LayerItemWidget] = []

        self.setFrameShape(QFrame.NoFrame)
        self.setObjectName("groupItem")
        self.setStyleSheet("""
            #groupItem {
                background-color: transparent;
                border-bottom: 1px solid #333;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.header_widget = QWidget()
        self.header_widget.setObjectName("groupHeader")
        self.header_widget.setStyleSheet("""
            #groupHeader {
                background-color: #2a2a2a;
                border-bottom: 1px solid #333;
            }
            #groupHeader:hover {
                background-color: #383838;
            }
        """)
        self.header_widget.mousePressEvent = self._toggle_expansion
        
        # Add context menu support
        self.header_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.header_widget.customContextMenuRequested.connect(self._show_group_context_menu)

        header_layout = QHBoxLayout(self.header_widget)
        header_layout.setContentsMargins(8, 6, 8, 6)
        header_layout.setSpacing(6)

        self.arrow_label = QLabel()
        self.arrow_label.setFixedWidth(10)

        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(10)
        self.title_label = QLabel(self.group_name)
        self.title_label.setFont(title_font)
        
        self.item_count_label = QLabel("")
        count_font = QFont()
        count_font.setPointSize(9)
        self.item_count_label.setFont(count_font)
        self.item_count_label.setStyleSheet("color: #888;")

        header_layout.addWidget(self.arrow_label)
        header_layout.addWidget(self.title_label)
        header_layout.addStretch()
        header_layout.addWidget(self.item_count_label)

        self.layer_container = QWidget()
        self.layer_container_layout = QVBoxLayout(self.layer_container)
        self.layer_container_layout.setContentsMargins(0, 0, 0, 0)
        self.layer_container_layout.setSpacing(1)

        layout.addWidget(self.header_widget)
        layout.addWidget(self.layer_container)

        self._update_arrow()

    def _toggle_expansion(self, event=None):
        # Only toggle on left-click, not right-click
        if event and hasattr(event, 'button') and event.button() != Qt.LeftButton:
            return
        self.is_expanded = not self.is_expanded
        self.layer_container.setVisible(self.is_expanded)
        self._update_arrow()

    def _update_arrow(self):
        self.arrow_label.setText("▼" if self.is_expanded else "►")
        
    def _update_item_count(self):
        script_count = len(self.script_groups)
        direct_layer_count = len(self.layer_widgets)
        
        # Count total layers across all script groups
        total_layers = direct_layer_count
        for script_group in self.script_groups:
            total_layers += len(script_group.layer_widgets)
            for run_subgroup in script_group.run_subgroups:
                total_layers += len(run_subgroup.layer_widgets)
        
        if script_count > 0:
            self.item_count_label.setText(f"({script_count} scripts, {total_layers} layers)")
        else:
            self.item_count_label.setText(f"({total_layers} layers)")

    def add_script_group(self, widget: ScriptGroupWidget):
        self.script_groups.append(widget)
        self.layer_container_layout.addWidget(widget)
        self._update_item_count()

    def add_layer_item(self, widget: 'LayerItemWidget'):
        """Add layer directly to processor group (for layers without script info)"""
        self.layer_widgets.append(widget)
        self.layer_container_layout.addWidget(widget)
        self._update_item_count()

    def remove_script_group(self, widget: ScriptGroupWidget):
        if widget in self.script_groups:
            self.script_groups.remove(widget)
            self.layer_container_layout.removeWidget(widget)
            widget.setParent(None)
            self._update_item_count()

    def remove_layer_item(self, widget: 'LayerItemWidget'):
        if widget in self.layer_widgets:
            self.layer_widgets.remove(widget)
            self.layer_container_layout.removeWidget(widget)
            widget.setParent(None)
            self._update_item_count()

    def find_script_group(self, script_name: str) -> Optional[ScriptGroupWidget]:
        """Find existing script group by name"""
        for script_group in self.script_groups:
            if script_group.script_name == script_name:
                return script_group
        return None
    
    def _show_group_context_menu(self, position):
        """Show context menu for group"""
        menu = QMenu()
        
        # Remove group action
        remove_action = menu.addAction("Remove Group")
        remove_action.triggered.connect(self._remove_group)
        
        # Show the menu
        menu.exec(self.header_widget.mapToGlobal(position))
    
    def _remove_group(self):
        """Remove this group and all its layers"""
        # Emit a signal to the parent to handle removal
        parent = self.parent()
        while parent and not hasattr(parent, 'remove_group'):
            parent = parent.parent()
        
        if parent and hasattr(parent, 'remove_group'):
            parent.remove_group(self)


class LayerItemWidget(QWidget):
    """Modern layer item widget with clean design"""
    
    visibility_changed = Signal(str, bool)  # layer_name, visible
    layer_selected = Signal(str)  # layer_name
    layer_double_clicked = Signal(str)  # layer_name
    
    def __init__(self, layer: UXOLayer):
        super().__init__()
        self.layer = layer
        self.setAutoFillBackground(True)
        self.setup_ui()
        self.set_selected(False)  # Initial state
        
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
        self.opacity_label = QLabel()
        self.opacity_label.setStyleSheet("color: #999999; font-size: 8px;")
        layout.addWidget(self.opacity_label)
        self.update_opacity_display()
        
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
        # Don't update layer.is_visible here - let the layer manager do it
        # This maintains the proper signal chain: widget -> layer panel -> layer manager -> map widget
        self.visibility_changed.emit(self.layer.name, not self.layer.is_visible)
            
    def _get_layer_color(self) -> QColor:
        """Get color for layer type and source"""
        # Check if this is processed data - style by source and metadata instead of type
        if self.layer.source.value == "processing":
            # Different colors for different types of processing results
            data_type = self.layer.metadata.get('data_type', '').lower()
            if 'anomaly' in data_type or 'anomalies' in data_type:
                return QColor(231, 76, 60)   # Red for anomalies
            else:
                return QColor(230, 126, 34)  # Orange for other processed data
        
        # Standard colors for layer types
        colors = {
            LayerType.POINTS: QColor(52, 152, 219),      # Blue
            LayerType.RASTER: QColor(46, 204, 113),      # Green
            LayerType.VECTOR: QColor(155, 89, 182),      # Purple
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
        
    def update_opacity_display(self):
        """Update opacity display label"""
        if self.layer.opacity < 1.0:
            self.opacity_label.setText(f"{int(self.layer.opacity * 100)}%")
            self.opacity_label.show()
        else:
            self.opacity_label.hide()

    def set_selected(self, is_selected: bool):
        """Set visual state for selection"""
        self.is_selected = is_selected
        
        # Update name label font
        font = self.name_label.font()
        font.setBold(is_selected)
        self.name_label.setFont(font)
        
        # Trigger a repaint to draw the selection bar
        self.update()

    def paintEvent(self, event):
        """Override paint event to draw a selection bar"""
        super().paintEvent(event)
        if self.is_selected:
            painter = QPainter(self)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor("#3498db"))  # Bright blue
            painter.drawRect(0, 0, 4, self.height())


class ModernLayerControlPanel(QWidget):
    """Modern, clean layer management panel"""
    
    # Signals
    layer_selected = Signal(str)  # layer_name
    visibility_changed = Signal(str, bool)  # layer_name, is_visible
    opacity_changed = Signal(str, float)  # layer_name, opacity
    style_edit_requested = Signal(str)  # layer_name
    zoom_to_layer = Signal(str)  # layer_name
    view_data_file_requested = Signal(str) # file_path
    
    def __init__(self, layer_manager: LayerManager):
        super().__init__()
        self.layer_manager = layer_manager
        self.layer_widgets: Dict[str, LayerItemWidget] = {}
        self.group_widgets: Dict[str, GroupItemWidget] = {}
        self.script_group_widgets: Dict[str, ScriptGroupWidget] = {}
        self.subgroup_widgets: Dict[str, SubgroupItemWidget] = {}
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
        self.layers_layout.setSpacing(0)
        self.layers_layout.addStretch()  # Push items to top

        # Create initial group widgets
        for group_name in self.layer_manager.layer_groups.keys():
            self._create_group_widget(group_name)
        
        scroll.setWidget(self.layers_container)
        layout.addWidget(scroll)
        
    def _create_group_widget(self, group_name: str) -> GroupItemWidget:
        """Creates and registers a group widget if it doesn't exist, but does not add to layout until needed."""
        if group_name in self.group_widgets:
            return self.group_widgets[group_name]

        group_widget = GroupItemWidget(group_name)
        group_widget.setStyleSheet("""
            #groupItem {
                background-color: #f8f9fa;
                border-bottom: 1px solid #dee2e6;
            }
        """)
        self.group_widgets[group_name] = group_widget
        logger.debug(f"Created group widget for '{group_name}'")
        return group_widget
        
    def _add_group_to_layout(self, group_widget: GroupItemWidget):
        """Adds the group widget to the layout if not already present and shows it."""
        if group_widget.parent() is None:  # Not yet added
            self.layers_layout.insertWidget(self.layers_layout.count() - 1, group_widget)
        group_widget.show()

    def _remove_group_from_layout_if_empty(self, group_widget: GroupItemWidget):
        """Hides the group if it has no layers, but keeps it in memory."""
        if len(group_widget.layer_widgets) == 0:
            group_widget.hide()

    def _create_controls(self, layout):
        """Create opacity and other controls"""
        controls = QFrame()
        controls.setFixedHeight(80)
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(12, 8, 12, 8)
        controls_layout.setSpacing(8)
        
        # Opacity control
        self.opacity_group = QGroupBox("Layer Opacity")
        opacity_layout = QVBoxLayout(self.opacity_group)
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
        controls_layout.addWidget(self.opacity_group)
        
        # Hide opacity controls by default
        self.opacity_group.setVisible(False)
        
        layout.addWidget(controls)
        
    def connect_signals(self):
        """Connect layer manager signals"""
        self.layer_manager.layer_added.connect(self._on_layer_added)
        self.layer_manager.layer_removed.connect(self._on_layer_removed)
        self.layer_manager.layer_visibility_changed.connect(self._on_visibility_changed)
        self.layer_manager.layer_style_changed.connect(self._on_style_changed)
        self.layer_manager.layer_display_name_changed.connect(self._on_display_name_changed)
        
    def _on_layer_added(self, layer: UXOLayer):
        """Handle layer addition with 3-level hierarchy: Processor → Script → Run → Layers"""
        if layer.name in self.layer_widgets:
            logger.warning(f"Layer widget for '{layer.name}' already exists. Updating.")
            return

        if not layer.display_name:
            layer.display_name = LayerNameParser.parse(layer)

        # Level 1: Processor Group (e.g., "Magnetic Processing")
        group_name = self._get_group_for_layer(layer)
        group_widget = self._create_group_widget(group_name)

        # Level 2: Script Group (e.g., "Magbase Processing")
        script_name = self._get_script_name_for_layer(layer)
        script_group_key = f"{group_name}:{script_name}"
        
        if script_group_key not in self.script_group_widgets:
            script_group_widget = ScriptGroupWidget(script_name)
            self.script_group_widgets[script_group_key] = script_group_widget
            group_widget.add_script_group(script_group_widget)
        
        script_group_widget = self.script_group_widgets[script_group_key]
        parent_widget = script_group_widget

        # Level 3: Run Subgroup (if processing_run_id exists)
        run_id = layer.processing_run_id
        if run_id:
            run_subgroup_key = f"{script_group_key}:{run_id}"
            if run_subgroup_key not in self.subgroup_widgets:
                run_subgroup_widget = SubgroupItemWidget(run_id, script_name)
                self.subgroup_widgets[run_subgroup_key] = run_subgroup_widget
                script_group_widget.add_run_subgroup(run_subgroup_widget)
            parent_widget = self.subgroup_widgets[run_subgroup_key]
        
        self._add_layer_item_to_parent(layer, parent_widget)
        self._add_group_to_layout(group_widget)

    def _add_layer_item_to_parent(self, layer: UXOLayer, parent_widget):
        """Creates a layer widget and adds it to a parent (group or subgroup)."""
        layer_widget = LayerItemWidget(layer)
        layer_widget.visibility_changed.connect(self._on_layer_visibility_toggled)
        layer_widget.layer_selected.connect(self._on_layer_selected)
        layer_widget.layer_double_clicked.connect(self.zoom_to_layer.emit)
        
        parent_widget.add_layer_item(layer_widget)
        self.layer_widgets[layer.name] = layer_widget
        
        # Context menu
        layer_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        layer_widget.customContextMenuRequested.connect(
            lambda pos, name=layer.name: self._show_context_menu(pos, name, layer_widget)
        )
        
        logger.debug(f"Added layer widget for '{layer.name}' to parent '{parent_widget.objectName()}'")

    def _get_group_for_layer(self, layer: UXOLayer) -> str:
        """Determine the group name for a given layer."""
        for g, layers in self.layer_manager.layer_groups.items():
            if layer.name in layers:
                return g
        return "Other"
    
    def _get_script_name_for_layer(self, layer: UXOLayer) -> str:
        """Get the script name for a layer, with intelligent fallbacks."""
        # First try the source_script field (with backward compatibility)
        source_script = getattr(layer, 'source_script', None)
        if source_script:
            return source_script
        
        # Try to infer from metadata
        if hasattr(layer, 'metadata') and layer.metadata:
            script_name = layer.metadata.get('script', '')
            if script_name:
                return script_name
            
            # Try processor type
            processor_type = layer.metadata.get('processor', '')
            if processor_type:
                return f"{processor_type.title()} Processing"
        
        # Try to infer from layer name patterns
        name_lower = layer.name.lower()
        if 'magbase' in name_lower or 'magnetic' in name_lower:
            return "Magnetic Processing"
        elif 'anomaly' in name_lower:
            return "Anomaly Detection"
        elif 'grid' in name_lower or 'interpolat' in name_lower:
            return "Grid Interpolation"
        elif 'flight' in name_lower or 'path' in name_lower:
            return "Flight Path Processing"
        elif 'gpr' in name_lower:
            return "GPR Processing"
        elif 'gamma' in name_lower:
            return "Gamma Processing"
        
        # Default fallback
        return "Data Processing"
        
    def _on_layer_removed(self, layer_name: str):
        """Handle layer removal from 3-level hierarchy."""
        if layer_name in self.layer_widgets:
            widget_to_remove = self.layer_widgets.pop(layer_name)
            parent = widget_to_remove.parentWidget()
            
            # Find the immediate parent container
            while parent and not isinstance(parent, (SubgroupItemWidget, ScriptGroupWidget, GroupItemWidget)):
                parent = parent.parentWidget()

            if parent:
                parent.remove_layer_item(widget_to_remove)
                
                # Clean up empty containers
                if isinstance(parent, SubgroupItemWidget) and not parent.layer_widgets:
                    # Remove empty run subgroup
                    script_parent = parent.parentWidget()
                    while script_parent and not isinstance(script_parent, ScriptGroupWidget):
                        script_parent = script_parent.parentWidget()
                    
                    if script_parent:
                        script_parent.remove_run_subgroup(parent)
                    
                    # Remove from tracking
                    keys_to_remove = [k for k, v in self.subgroup_widgets.items() if v == parent]
                    for key in keys_to_remove:
                        del self.subgroup_widgets[key]
                    
                    parent.deleteLater()
                
                elif isinstance(parent, ScriptGroupWidget) and not parent.layer_widgets and not parent.run_subgroups:
                    # Remove empty script group
                    group_parent = parent.parentWidget()
                    while group_parent and not isinstance(group_parent, GroupItemWidget):
                        group_parent = group_parent.parentWidget()
                    
                    if group_parent:
                        group_parent.remove_script_group(parent)
                    
                    # Remove from tracking
                    keys_to_remove = [k for k, v in self.script_group_widgets.items() if v == parent]
                    for key in keys_to_remove:
                        del self.script_group_widgets[key]
                    
                    parent.deleteLater()

            widget_to_remove.deleteLater()

            if self.current_layer == layer_name:
                self.current_layer = None
                self._update_controls_for_selection()
                
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

    def _on_display_name_changed(self, layer_name: str, new_display_name: str):
        """Handle display name changes from the manager."""
        if layer_name in self.layer_widgets:
            widget = self.layer_widgets[layer_name]
            widget.name_label.setText(new_display_name)
        
    def _on_layer_visibility_toggled(self, layer_name: str, is_visible: bool):
        """Handle layer visibility toggle from widget"""
        self.layer_manager.set_layer_visibility(layer_name, is_visible)
        self.visibility_changed.emit(layer_name, is_visible)
        
    def _on_layer_selected(self, layer_name: str):
        """Handle layer selection and deselection."""
        # If the clicked layer is already selected, deselect it. Otherwise, select the new one.
        if self.current_layer == layer_name:
            self.current_layer = None
        else:
            self.current_layer = layer_name
        
        self._update_controls_for_selection()
        
        # Emit signal only when a layer is actually selected
        if self.current_layer:
            self.layer_selected.emit(self.current_layer)

    def _update_controls_for_selection(self):
        """Update all UI controls based on the current selection state."""
        # Update visual state for all layer widgets
        for name, widget in self.layer_widgets.items():
            widget.set_selected(name == self.current_layer)
        
        is_layer_selected = self.current_layer is not None
        
        # Update controls visibility and state
        self.opacity_group.setVisible(is_layer_selected)
        self.remove_btn.setEnabled(is_layer_selected)
        self.opacity_slider.setEnabled(is_layer_selected)
        
        if is_layer_selected:
            layer = self.layer_manager.get_layer(self.current_layer)
            if layer:
                opacity_value = int(layer.opacity * 100)
                # Block signals to prevent feedback loops when setting value
                self.opacity_slider.blockSignals(True)
                self.opacity_slider.setValue(opacity_value)
                self.opacity_slider.blockSignals(False)
                self.opacity_label.setText(f"{opacity_value}%")
        else:
            self.opacity_label.setText("--")

    def _on_opacity_changed(self, value: int):
        """Handle opacity slider change"""
        opacity = value / 100.0
        self.opacity_label.setText(f"{value}%")
        
        if self.current_layer:
            # The layer manager will emit the signal that the map widget is listening to.
            self.layer_manager.set_layer_opacity(self.current_layer, opacity)
            
            # Update widget display efficiently
            if self.current_layer in self.layer_widgets:
                widget = self.layer_widgets[self.current_layer]
                widget.layer.opacity = opacity
                widget.update_opacity_display()
                    
    def _show_context_menu(self, position, layer_name: str, widget: LayerItemWidget):
        """Show context menu for a layer."""
        layer = self.layer_manager.get_layer(layer_name)
        if not layer:
            return

        menu = QMenu(self)

        # Rename Action
        rename_action = menu.addAction("Rename")
        rename_action.triggered.connect(lambda: self._rename_layer(layer_name))

        menu.addSeparator()
        
        # Zoom to layer
        zoom_action = menu.addAction("Zoom to Layer")
        zoom_action.triggered.connect(lambda: self.zoom_to_layer.emit(layer_name))
        
        # Properties
        props_action = menu.addAction("Properties...")
        props_action.triggered.connect(lambda: self._show_layer_properties(layer_name))
        
        # Metadata Info
        metadata_action = menu.addAction("Show Metadata...")
        metadata_action.triggered.connect(lambda: self._show_layer_metadata(layer_name))

        # Process Action (if layer has traceability)
        if self._can_process_layer(layer):
            menu.addSeparator()
            process_action = menu.addAction("Process...")
            process_action.triggered.connect(lambda: self._process_layer(layer_name))

        # View Generated Data Action
        if layer.generated_output_files:
            menu.addSeparator()
            view_data_action = menu.addMenu("View Generated Data")
            for fpath in layer.generated_output_files:
                fname = os.path.basename(fpath)
                action = view_data_action.addAction(fname)
                action.triggered.connect(lambda checked, path=fpath: self.view_data_file_requested.emit(path))
        
        menu.addSeparator()
        
        # Remove layer
        remove_action = menu.addAction("Remove Layer")
        remove_action.triggered.connect(lambda: self.layer_manager.remove_layer(layer_name))
        
        # Show menu
        menu.exec_(widget.mapToGlobal(position))

    def _rename_layer(self, layer_name: str):
        """Prompt user for a new layer name and call the manager."""
        layer = self.layer_manager.get_layer(layer_name)
        if not layer:
            return

        current_name = layer.display_name
        new_name, ok = QInputDialog.getText(
            self,
            "Rename Layer",
            "Enter new layer name:",
            text=current_name
        )

        if ok and new_name and new_name != current_name:
            self.layer_manager.rename_layer(layer_name, new_name)
    
    def _show_layer_properties(self, layer_name: str):
        """Show layer properties dialog"""
        layer = self.layer_manager.get_layer(layer_name)
        if not layer:
            return
        
        from .layer_style_editor import LayerStyleEditor
        dialog = LayerStyleEditor(layer, self.layer_manager, self)
        dialog.exec()
    
    def _show_layer_metadata(self, layer_name: str):
        """Show detailed metadata information for a layer"""
        layer = self.layer_manager.get_layer(layer_name)
        if not layer:
            return
        
        from qtpy.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QDialogButtonBox
        from qtpy.QtCore import Qt
        import json
        
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Layer Metadata - {layer.display_name or layer.name}")
        dialog.setModal(True)
        dialog.resize(600, 500)
        
        layout = QVBoxLayout()
        
        # Create metadata text display
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setFont(QFont("Courier", 10))
        
        # Format metadata info
        metadata_info = []
        metadata_info.append(f"=== LAYER INFORMATION ===")
        metadata_info.append(f"Name: {layer.name}")
        metadata_info.append(f"Display Name: {layer.display_name or 'None'}")
        metadata_info.append(f"Type: {layer.layer_type.value}")
        metadata_info.append(f"Geometry: {layer.geometry_type.value}")
        metadata_info.append(f"Source: {layer.source.value}")
        metadata_info.append(f"Visible: {layer.is_visible}")
        metadata_info.append(f"Opacity: {layer.opacity}")
        metadata_info.append("")
        
        metadata_info.append(f"=== TRACEABILITY ===")
        # Use getattr for backward compatibility with older layers
        processing_run_id = getattr(layer, 'processing_run_id', None)
        source_script = getattr(layer, 'source_script', None)
        
        # Extract input/output files from metadata if available
        input_files = getattr(layer, 'source_input_files', [])
        output_files = getattr(layer, 'generated_output_files', [])
        
        # Also check in metadata for file info (enhanced format)
        if hasattr(layer, 'metadata') and isinstance(layer.metadata, dict):
            processing_metadata = layer.metadata.get('processing_metadata', {})
            file_info = processing_metadata.get('file_info', {})
            
            # Extract input file
            if not input_files and file_info.get('input_file'):
                input_files = [file_info['input_file']]
                
            # Extract output file
            if not output_files and file_info.get('output_file'):
                output_files = [file_info['output_file']]
        
        metadata_info.append(f"Processing Run ID: {processing_run_id or 'None'}")
        metadata_info.append(f"Source Script: {source_script or 'None'}")
        
        # Check if traceability data is missing and add note
        if not processing_run_id and not source_script and not input_files and not output_files:
            metadata_info.append("⚠️  Note: This layer was created before traceability features were added.")
            metadata_info.append("   File traceability information is not available.")
        
        # Input Files Section
        metadata_info.append(f"📁 Input Files: {len(input_files)}")
        if input_files:
            for i, file in enumerate(input_files):
                filename = file.split('/')[-1]  # Show just filename
                metadata_info.append(f"   {i+1}. {filename}")
                metadata_info.append(f"      Path: {file}")
        else:
            metadata_info.append("   (None)")
        metadata_info.append("")
        
        # Output Files Section  
        metadata_info.append(f"📄 Output Files: {len(output_files)}")
        if output_files:
            for i, file in enumerate(output_files):
                filename = file.split('/')[-1]  # Show just filename
                metadata_info.append(f"   {i+1}. {filename}")
                metadata_info.append(f"      Path: {file}")
        else:
            metadata_info.append("   (None)")
        metadata_info.append("")
        
        # Generated Figures Section
        figures_info = []
        if hasattr(layer, 'metadata') and layer.metadata:
            if 'figures' in layer.metadata:
                figures_info = layer.metadata['figures']
            elif 'figure_count' in layer.metadata:
                figures_info = [{'description': 'Generated figure'}] * layer.metadata['figure_count']
            elif 'has_pending_figure' in layer.metadata:
                figures_info = [{'description': 'Interactive plot (.mplplot file)'}]
        
        metadata_info.append(f"📊 Generated Figures: {len(figures_info)} files")
        if figures_info:
            for i, fig_info in enumerate(figures_info):
                if isinstance(fig_info, dict):
                    desc = fig_info.get('description', 'Interactive matplotlib plot')
                    file_path = fig_info.get('file_path', 'Generated by pipeline')
                    if file_path and not file_path.startswith('Generated by pipeline'):
                        # Real file path available
                        filename = file_path.split('/')[-1]
                        metadata_info.append(f"   {i+1}. {filename}")
                        metadata_info.append(f"      Path: {file_path}")
                        metadata_info.append(f"      Type: {desc}")
                    else:
                        # Placeholder or auto-generated
                        metadata_info.append(f"   {i+1}. {desc}")
                        metadata_info.append(f"      Status: Auto-saved to project/processed/ directory")
                        metadata_info.append(f"      Format: Interactive .mplplot file")
                else:
                    metadata_info.append(f"   {i+1}. {str(fig_info)}")
        else:
            metadata_info.append("   (None)")
        metadata_info.append("")
        
        metadata_info.append(f"=== PROCESSING HISTORY ===")
        for i, step in enumerate(layer.processing_history):
            metadata_info.append(f"{i+1}. {step}")
        metadata_info.append("")
        
        metadata_info.append(f"=== DATA BOUNDS ===")
        if layer.bounds:
            metadata_info.append(f"Min X: {layer.bounds[0]}")
            metadata_info.append(f"Min Y: {layer.bounds[1]}")
            metadata_info.append(f"Max X: {layer.bounds[2]}")
            metadata_info.append(f"Max Y: {layer.bounds[3]}")
        else:
            metadata_info.append("No bounds available")
        metadata_info.append("")
        
        metadata_info.append(f"=== ADDITIONAL METADATA ===")
        if layer.metadata:
            metadata_info.append(json.dumps(layer.metadata, indent=2))
        else:
            metadata_info.append("No additional metadata")
        
        text_edit.setPlainText('\n'.join(metadata_info))
        layout.addWidget(text_edit)
        
        # Add buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok)
        button_box.accepted.connect(dialog.accept)
        layout.addWidget(button_box)
        
        dialog.setLayout(layout)
        dialog.exec()
    
    def _can_process_layer(self, layer: UXOLayer) -> bool:
        """Check if a layer can be processed (has traceability data)"""
        # Layer can be processed if it has generated output files or source input files
        # Use getattr for backward compatibility with older layers
        generated_files = getattr(layer, 'generated_output_files', [])
        input_files = getattr(layer, 'source_input_files', [])
        return bool(generated_files or input_files)
    
    def _process_layer(self, layer_name: str):
        """Open processing dialog for a layer"""
        layer = self.layer_manager.get_layer(layer_name)
        if not layer:
            return
        
        # Find the most relevant file for processing
        file_to_process = None
        
        # First try generated output files (these are the actual data files)
        # Use getattr for backward compatibility with older layers
        generated_files = getattr(layer, 'generated_output_files', [])
        input_files = getattr(layer, 'source_input_files', [])
        
        if generated_files:
            file_to_process = generated_files[0]  # Use first file
        # Fallback to source input files
        elif input_files:
            file_to_process = input_files[0]
        
        if file_to_process:
            # Import the processing dialog
            try:
                from ..widgets.processing.processing_dialog import ProcessingDialog
                dialog = ProcessingDialog(self, initial_file=file_to_process)
                dialog.exec()
            except ImportError:
                # If processing dialog doesn't exist, show a message
                from qtpy.QtWidgets import QMessageBox
                QMessageBox.information(
                    self,
                    "Processing",
                    f"Processing dialog not available.\nLayer file: {file_to_process}\n\nMetadata:\n"
                    f"- Script: {getattr(layer, 'source_script', None) or 'Unknown'}\n"
                    f"- Input files: {len(getattr(layer, 'source_input_files', []))} files\n"
                    f"- Output files: {len(getattr(layer, 'generated_output_files', []))} files"
                )
        else:
            from qtpy.QtWidgets import QMessageBox
            QMessageBox.warning(
                self,
                "No Data Files",
                "This layer has no associated data files to process."
            )
        
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
    
    def remove_group(self, group_widget: GroupItemWidget):
        """Remove an entire group and all its layers"""
        from qtpy.QtWidgets import QMessageBox
        
        # Count total layers in the group
        total_layers = 0
        for script_group in group_widget.script_groups:
            total_layers += len(script_group.layer_widgets)
            for run_subgroup in script_group.run_subgroups:
                total_layers += len(run_subgroup.layer_widgets)
        total_layers += len(group_widget.layer_widgets)
        
        if total_layers > 0:
            reply = QMessageBox.question(
                self,
                "Remove Group",
                f"Are you sure you want to remove the entire '{group_widget.group_name}' group?\n"
                f"This will remove {total_layers} layers.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply != QMessageBox.Yes:
                return
        
        # Remove all layers in the group
        layers_to_remove = []
        
        # Collect all layer names from script groups
        for script_group in group_widget.script_groups:
            for layer_widget in script_group.layer_widgets:
                layers_to_remove.append(layer_widget.layer.name)
            for run_subgroup in script_group.run_subgroups:
                for layer_widget in run_subgroup.layer_widgets:
                    layers_to_remove.append(layer_widget.layer.name)
        
        # Collect direct layer names
        for layer_widget in group_widget.layer_widgets:
            layers_to_remove.append(layer_widget.layer.name)
        
        # Remove all layers
        for layer_name in layers_to_remove:
            self.layer_manager.remove_layer(layer_name)
    
    def remove_script_group(self, script_group_widget: ScriptGroupWidget):
        """Remove an entire script group and all its layers"""
        from qtpy.QtWidgets import QMessageBox
        
        # Count total layers in the script group
        total_layers = len(script_group_widget.layer_widgets)
        for run_subgroup in script_group_widget.run_subgroups:
            total_layers += len(run_subgroup.layer_widgets)
        
        if total_layers > 0:
            reply = QMessageBox.question(
                self,
                "Remove Script Group",
                f"Are you sure you want to remove the entire '{script_group_widget._format_script_name(script_group_widget.script_name)}' script group?\n"
                f"This will remove {total_layers} layers.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply != QMessageBox.Yes:
                return
        
        # Remove all layers in the script group
        layers_to_remove = []
        
        # Collect layer names from run subgroups
        for run_subgroup in script_group_widget.run_subgroups:
            for layer_widget in run_subgroup.layer_widgets:
                layers_to_remove.append(layer_widget.layer.name)
        
        # Collect direct layer names
        for layer_widget in script_group_widget.layer_widgets:
            layers_to_remove.append(layer_widget.layer.name)
        
        # Remove all layers
        for layer_name in layers_to_remove:
            self.layer_manager.remove_layer(layer_name)


# Keep backwards compatibility
LayerControlPanel = ModernLayerControlPanel 