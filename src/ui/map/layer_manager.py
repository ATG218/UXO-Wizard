"""
Layer Manager - Central registry for all map layers
"""

from PySide6.QtCore import QObject, Signal
from typing import Dict, List, Optional
from .layer_types import UXOLayer, LayerStyle
from loguru import logger


class LayerManager(QObject):
    """Manages all map layers with hierarchical organization"""
    
    # Signals
    layer_added = Signal(UXOLayer)
    layer_removed = Signal(str)  # layer_name
    layer_visibility_changed = Signal(str, bool)  # layer_name, is_visible
    layer_style_changed = Signal(str, LayerStyle)  # layer_name, new_style
    layer_opacity_changed = Signal(str, float)  # layer_name, opacity
    layer_selected = Signal(str)  # layer_name
    layer_order_changed = Signal(list)  # ordered layer names
    layer_bounds_changed = Signal(list)  # [min_x, min_y, max_x, max_y]
    
    def __init__(self):
        super().__init__()
        self.layers: Dict[str, UXOLayer] = {}
        self.layer_groups: Dict[str, List[str]] = {
            "Survey Data": [],
            "Magnetic Processing": [],
            "GPR Processing": [],
            "Gamma Processing": [],
            "Multispectral Processing": [],
            "Annotations": [],
            "Other": []
        }
        self.layer_order: List[str] = []  # Ordered list of layer names
        
    def add_layer(self, layer: UXOLayer, group: Optional[str] = None) -> bool:
        """Add a new layer to the manager"""
        if layer.name in self.layers:
            logger.warning(f"Layer '{layer.name}' already exists, updating instead")
            return self.update_layer(layer)
            
        # Store the layer
        self.layers[layer.name] = layer
        self.layer_order.append(layer.name)
        
        # Assign to group
        if group is None:
            group = self._auto_assign_group(layer)
        if group in self.layer_groups:
            self.layer_groups[group].append(layer.name)
        else:
            self.layer_groups["Other"].append(layer.name)
            
        logger.info(f"Added layer '{layer.name}' to group '{group}'")
        
        # Update global bounds
        self._update_global_bounds()
        
        # Emit signal
        self.layer_added.emit(layer)
        return True
        
    def remove_layer(self, layer_name: str) -> bool:
        """Remove a layer from the manager"""
        if layer_name not in self.layers:
            logger.warning(f"Layer '{layer_name}' not found")
            return False
            
        # Remove from storage
        del self.layers[layer_name]
        self.layer_order.remove(layer_name)
        
        # Remove from groups
        for group_layers in self.layer_groups.values():
            if layer_name in group_layers:
                group_layers.remove(layer_name)
                
        logger.info(f"Removed layer '{layer_name}'")
        
        # Update global bounds
        self._update_global_bounds()
        
        # Emit signal
        self.layer_removed.emit(layer_name)
        return True
        
    def update_layer(self, layer: UXOLayer) -> bool:
        """Update an existing layer"""
        if layer.name not in self.layers:
            logger.warning(f"Layer '{layer.name}' not found, adding instead")
            return self.add_layer(layer)
            
        # Update the layer
        old_layer = self.layers[layer.name]
        self.layers[layer.name] = layer
        
        # Check if visibility changed
        if old_layer.is_visible != layer.is_visible:
            self.layer_visibility_changed.emit(layer.name, layer.is_visible)
            
        # Check if style changed
        if old_layer.style != layer.style:
            self.layer_style_changed.emit(layer.name, layer.style)
            
        # Update global bounds if needed
        if old_layer.bounds != layer.bounds:
            self._update_global_bounds()
            
        logger.info(f"Updated layer '{layer.name}'")
        return True
        
    def get_layer(self, layer_name: str) -> Optional[UXOLayer]:
        """Get a layer by name"""
        return self.layers.get(layer_name)
        
    def get_visible_layers(self) -> List[UXOLayer]:
        """Get all visible layers in z-order"""
        visible = [layer for layer in self.layers.values() if layer.is_visible]
        return sorted(visible, key=lambda l: (l.z_index, self.layer_order.index(l.name)))
        
    def set_layer_visibility(self, layer_name: str, is_visible: bool) -> bool:
        """Set layer visibility"""
        if layer_name not in self.layers:
            logger.warning(f"Layer '{layer_name}' not found")
            return False
            
        layer = self.layers[layer_name]
        if layer.is_visible != is_visible:
            layer.is_visible = is_visible
            self.layer_visibility_changed.emit(layer_name, is_visible)
            logger.debug(f"Set layer '{layer_name}' visibility to {is_visible}")
            
        return True
        
    def set_layer_style(self, layer_name: str, style: LayerStyle) -> bool:
        """Update layer style"""
        if layer_name not in self.layers:
            logger.warning(f"Layer '{layer_name}' not found")
            return False
            
        layer = self.layers[layer_name]
        layer.style = style
        self.layer_style_changed.emit(layer_name, style)
        logger.debug(f"Updated style for layer '{layer_name}'")
        return True
        
    def set_layer_opacity(self, layer_name: str, opacity: float) -> bool:
        """Set layer opacity"""
        if layer_name not in self.layers:
            logger.warning(f"Layer '{layer_name}' not found")
            return False
            
        layer = self.layers[layer_name]
        if layer.opacity != opacity:
            layer.opacity = opacity
            self.layer_opacity_changed.emit(layer_name, opacity)
            logger.debug(f"Set layer '{layer_name}' opacity to {opacity}")
            
        return True
        
    def reorder_layers(self, layer_names: List[str]) -> bool:
        """Reorder layers (updates z-index)"""
        # Validate all layers exist
        for name in layer_names:
            if name not in self.layers:
                logger.warning(f"Layer '{name}' not found in reorder request")
                return False
                
        # Update z-indices
        for i, name in enumerate(layer_names):
            self.layers[name].z_index = i
            
        self.layer_order = layer_names
        self.layer_order_changed.emit(layer_names)
        logger.debug(f"Reordered {len(layer_names)} layers")
        return True
        
    def get_global_bounds(self) -> Optional[List[float]]:
        """Get bounds encompassing all layers"""
        if not self.layers:
            return None
            
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = float('-inf'), float('-inf')
        
        has_bounds = False
        for layer in self.layers.values():
            if layer.bounds:
                has_bounds = True
                min_x = min(min_x, layer.bounds[0])
                min_y = min(min_y, layer.bounds[1])
                max_x = max(max_x, layer.bounds[2])
                max_y = max(max_y, layer.bounds[3])
                
        return [min_x, min_y, max_x, max_y] if has_bounds else None
        
    def get_visible_bounds(self) -> Optional[List[float]]:
        """Get bounds encompassing all VISIBLE layers"""
        visible_layers = [layer for layer in self.layers.values() if layer.is_visible]
        if not visible_layers:
            return None

        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = float('-inf'), float('-inf')

        has_bounds = False
        for layer in visible_layers:
            if layer.bounds:
                has_bounds = True
                min_x = min(min_x, layer.bounds[0])
                min_y = min(min_y, layer.bounds[1])
                max_x = max(max_x, layer.bounds[2])
                max_y = max(max_y, layer.bounds[3])

        return [min_x, min_y, max_x, max_y] if has_bounds else None
        
    def clear_all(self):
        """Remove all layers"""
        layer_names = list(self.layers.keys())
        for name in layer_names:
            self.remove_layer(name)
            
    def _auto_assign_group(self, layer: UXOLayer) -> str:
        """Automatically assign layer to appropriate group with enhanced logic"""
        
        # Check metadata for processor type (enhanced for new layer generation system)
        processor_type = layer.metadata.get('processor_type', '').lower()
        if processor_type:
            if processor_type == 'magnetic':
                return "Magnetic Processing"
            elif processor_type == 'gpr':
                return "GPR Processing" 
            elif processor_type == 'gamma':
                return "Gamma Processing"
            elif processor_type == 'multispectral':
                return "Multispectral Processing"
        
        # Check processing history for group assignment (fallback)
        if layer.processing_history:
            last_process = layer.processing_history[-1].lower()
            if "magnetic" in last_process:
                return "Magnetic Processing"
            elif "gpr" in last_process:
                return "GPR Processing"
            elif "gamma" in last_process:
                return "Gamma Processing"
            elif "multispectral" in last_process:
                return "Multispectral Processing"
        
        # Check for specific data types in metadata (for sub-grouping)
        data_type = layer.metadata.get('data_type', '').lower()
        if data_type:
            if 'anomaly' in data_type or 'anomalies' in data_type:
                # Anomaly layers get special treatment
                if processor_type == 'magnetic':
                    return "Magnetic Processing"  # Could be "Magnetic Anomalies" for sub-groups
                elif processor_type:
                    return f"{processor_type.title()} Processing"
        
        # Check source
        if layer.source.value == "data_viewer":
            return "Survey Data"
        elif layer.source.value == "annotation":
            return "Annotations"
        elif layer.source.value == "processing":
            return "Data Processing"  # Generic processing group
            
        return "Other"
        
    def _update_global_bounds(self):
        """Update and emit global bounds signal"""
        bounds = self.get_global_bounds()
        if bounds:
            self.layer_bounds_changed.emit(bounds) 