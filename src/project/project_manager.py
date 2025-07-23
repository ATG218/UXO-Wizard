"""
Project Manager - Main interface for project operations
"""

import json
import pickle
import zipfile
import tempfile
import shutil
import os
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from PySide6.QtCore import QObject, Signal
from loguru import logger

from .project_schema import UXOProject, UXOProjectMetadata, MapState, ProcessingStep, DataViewerState, DataViewerTabState, LayerVisualState
from .project_history import ProjectHistoryLogger
from ..ui.map.layer_manager import LayerManager
from ..ui.map.layer_types import UXOLayer


class ProjectManager(QObject):
    """Manages project save/load operations and state"""
    
    # Signals
    project_saved = Signal(str)  # file_path
    project_loaded = Signal(str)  # file_path
    project_error = Signal(str, str)  # operation, error_message
    project_created = Signal(str)  # project_name
    project_closed = Signal()
    working_directory_restored = Signal(str)  # working_directory_path
    
    def __init__(self, layer_manager: LayerManager):
        super().__init__()
        self.layer_manager = layer_manager
        self.current_project: Optional[UXOProject] = None
        self.current_file_path: Optional[str] = None
        self.history_logger: Optional[ProjectHistoryLogger] = None
        self._is_dirty = False  # Track unsaved changes
        self._current_working_directory: Optional[str] = None  # Track current working directory
        self._data_viewer = None  # Reference to data viewer widget
        
    def create_new_project(self, name: str, working_directory: str = None) -> UXOProject:
        """Create a new empty project"""
        project = UXOProject(
            name=name,
            working_directory=Path(working_directory) if working_directory else None
        )
        
        self.current_project = project
        self.current_file_path = None
        self._is_dirty = True
        
        # Initialize history logger
        project_root = working_directory or self._current_working_directory
        if project_root:
            self.history_logger = ProjectHistoryLogger(project_root)
            self.history_logger.log_event('project.created', {
                'project_name': name,
                'schema_version': '1.0.0'
            })

        self.project_created.emit(name)
        logger.info(f"Created new project: {name}")
        return project
    
    def save_project(self, file_path: str) -> bool:
        """Save current project to .uxo file"""
        try:
            if not self.current_project:
                # Create project from current layer manager state
                self.current_project = self._create_project_from_current_state()
            
            # Update project with current layer manager state
            self._sync_project_with_layer_manager()
            
            # Save to .uxo file
            success = self._save_uxo_file(file_path, self.current_project)
            
            if success:
                self.current_file_path = file_path
                self._is_dirty = False
                self.project_saved.emit(file_path)
                logger.info(f"Project saved: {file_path}")
                return True
            
        except Exception as e:
            error_msg = f"Failed to save project: {str(e)}"
            logger.error(error_msg)
            self.project_error.emit("save", error_msg)
            
        return False
    
    def load_project(self, file_path: str) -> bool:
        """Load project from .uxo file"""
        try:
            project = self._load_uxo_file(file_path)
            
            if project:
                self.current_project = project
                self.current_file_path = file_path
                self._is_dirty = False
                
                # Initialize history logger with correct working directory (FIXED)
                if self.current_project and self.current_project.working_directory:
                    project_root = str(self.current_project.working_directory)
                else:
                    # Fallback to .uxo file directory if no working directory is set
                    project_root = os.path.dirname(file_path)
                    logger.warning("No working directory found, using .uxo file location for history logger")
                
                self.history_logger = ProjectHistoryLogger(project_root)
                self.history_logger.log_event('project.loaded', {
                    'project_name': self.current_project.name if self.current_project else 'Unknown',
                    'file_path': file_path,
                    'working_directory': project_root
                })

                # Restore to layer manager
                self._restore_project_to_layer_manager(project)
                
                self.project_loaded.emit(file_path)
                logger.info(f"Project loaded: {file_path}")
                return True
                
        except Exception as e:
            error_msg = f"Failed to load project: {str(e)}"
            logger.error(error_msg)
            self.project_error.emit("load", error_msg)
            
        return False
    
    def save_project_as(self, file_path: str) -> bool:
        """Save project with new file path"""
        return self.save_project(file_path)

    def close_project(self):
        """Close the current project and release resources."""
        if self.history_logger:
            self.history_logger.close()
            self.history_logger = None
        
        self.current_project = None
        self.current_file_path = None
        self._is_dirty = False
        self.layer_manager.clear_all()
        self.project_closed.emit()
        logger.info("Project closed.")

    def is_dirty(self) -> bool:
        """Check if project has unsaved changes"""
        return self._is_dirty
    
    def mark_dirty(self):
        """Mark project as having unsaved changes"""
        self._is_dirty = True
        if self.current_project:
            self.current_project.update_modified()
    
    def get_current_project_name(self) -> Optional[str]:
        """Get current project name"""
        return self.current_project.name if self.current_project else None
    
    def get_current_file_path(self) -> Optional[str]:
        """Get current file path"""
        return self.current_file_path
    
    def set_current_working_directory(self, working_directory: str):
        """Set the current working directory and initialize a project if none exists."""
        if self.current_project and self.current_project.working_directory == Path(working_directory):
            return # Avoid re-initializing the same project

        self.close_project() # Close any existing project first

        self._current_working_directory = working_directory
        project_name = os.path.basename(working_directory)

        # Ensure _project directory structure exists
        project_internal_dir = os.path.join(working_directory, '_project')
        os.makedirs(project_internal_dir, exist_ok=True)
        
        # Create a new project object for the opened folder
        self.create_new_project(name=project_name, working_directory=working_directory)
        
        logger.debug(f"Current working directory set to: {working_directory}")
    
    def get_current_working_directory(self) -> Optional[str]:
        """Get current working directory"""
        if self.current_project and self.current_project.working_directory:
            result = str(self.current_project.working_directory)
            return result
        return self._current_working_directory
    
    def set_data_viewer(self, data_viewer):
        """Set reference to the data viewer widget"""
        self._data_viewer = data_viewer
        logger.debug("Data viewer reference set in project manager")
    
    def capture_data_viewer_state(self) -> 'DataViewerState':
        """Capture current data viewer state"""
        from .project_schema import DataViewerState, DataViewerTabState
        
        if not self._data_viewer:
            logger.warning("No data viewer reference - returning empty state")
            return DataViewerState()
        
        try:
            open_tabs = []
            tab_widget = self._data_viewer.tab_widget
            
            # Capture state of each tab
            for i in range(tab_widget.count()):
                widget = tab_widget.widget(i)
                tab_title = tab_widget.tabText(i)
                
                # Skip welcome tab
                if tab_title == "Welcome":
                    continue
                
                # Only process actual DataViewerTab widgets
                if hasattr(widget, 'current_file') and widget.current_file:
                    tab_state = DataViewerTabState(
                        file_path=widget.current_file,
                        tab_title=tab_title,
                        current_column_filter=widget.column_selector.get_selected_columns() if hasattr(widget, 'column_selector') else [],
                        search_text=widget.search_edit.text() if hasattr(widget, 'search_edit') else ""
                    )
                    open_tabs.append(tab_state)
            
            # Get active tab index (adjust for welcome tab)
            current_index = tab_widget.currentIndex()
            active_tab_index = max(0, current_index) if open_tabs else 0
            
            # Check if welcome tab is shown
            has_welcome_tab = (tab_widget.count() == 1 and 
                             tab_widget.tabText(0) == "Welcome")
            
            state = DataViewerState(
                open_tabs=open_tabs,
                active_tab_index=active_tab_index,
                has_welcome_tab=has_welcome_tab
            )
            
            logger.debug(f"Captured data viewer state: {len(open_tabs)} tabs, active: {active_tab_index}")
            return state
            
        except Exception as e:
            logger.error(f"Error capturing data viewer state: {e}")
            return DataViewerState()
    
    def restore_data_viewer_state(self, state: 'DataViewerState'):
        """Restore data viewer state"""
        if not self._data_viewer:
            logger.warning("No data viewer reference - cannot restore state")
            return
        
        try:
            # Close all existing tabs first
            tab_widget = self._data_viewer.tab_widget
            for i in range(tab_widget.count() - 1, -1, -1):
                tab_widget.removeTab(i)
            
            # Restore tabs
            if state.open_tabs:
                for i, tab_state in enumerate(state.open_tabs):
                    try:
                        # Check if file still exists
                        if not os.path.exists(tab_state.file_path):
                            logger.warning(f"File no longer exists: {tab_state.file_path}")
                            continue
                        
                        # Create new tab with the file
                        tab = self._data_viewer.add_file_tab(tab_state.file_path)
                        
                        # Restore tab state
                        if hasattr(tab, 'column_selector') and tab_state.current_column_filter:
                            if isinstance(tab_state.current_column_filter, list):
                                # New format: list of column names
                                tab.column_selector.set_selected_columns(tab_state.current_column_filter)
                            else:
                                # Legacy format: single column name
                                if tab_state.current_column_filter != "All columns":
                                    tab.column_selector.set_selected_columns([tab_state.current_column_filter])
                        elif hasattr(tab, 'column_combo') and tab_state.current_column_filter:
                            # Fallback for legacy tabs
                            combo = tab.column_combo
                            index = combo.findText(tab_state.current_column_filter)
                            if index >= 0:
                                combo.setCurrentIndex(index)
                        
                        if hasattr(tab, 'search_edit') and tab_state.search_text:
                            tab.search_edit.setText(tab_state.search_text)
                        
                    except Exception as e:
                        logger.error(f"Error restoring tab {tab_state.file_path}: {e}")
                
                # Set active tab
                if 0 <= state.active_tab_index < tab_widget.count():
                    tab_widget.setCurrentIndex(state.active_tab_index)
                
                logger.info(f"Restored data viewer state: {len(state.open_tabs)} tabs")
            else:
                # No tabs to restore - let DataViewer show welcome screen
                self._data_viewer.update_ui_state()
                logger.info("No data viewer tabs to restore - showing welcome screen")
                
        except Exception as e:
            logger.error(f"Error restoring data viewer state: {e}")
    
    def capture_map_visual_state(self) -> 'MapState':
        """Capture current map visual state including layer visibility/opacity"""
        from .project_schema import MapState, LayerVisualState
        
        # Get current map state (we'll enhance this later to get actual map center/zoom)
        map_state = MapState()
        
        # Capture layer visual states
        layer_visual_states = []
        for layer_name, layer in self.layer_manager.layers.items():
            visual_state = LayerVisualState(
                layer_name=layer_name,
                is_visible=layer.is_visible,
                opacity=layer.opacity,
                z_index=layer.z_index
            )
            layer_visual_states.append(visual_state)
        
        map_state.layer_visual_states = layer_visual_states
        
        logger.debug(f"Captured visual state for {len(layer_visual_states)} layers")
        return map_state
    
    def restore_map_visual_state(self, map_state: 'MapState'):
        """Restore map visual state including layer visibility/opacity"""
        if not map_state.layer_visual_states:
            logger.debug("No layer visual states to restore")
            return
        
        # Restore layer visual states
        for visual_state in map_state.layer_visual_states:
            layer = self.layer_manager.get_layer(visual_state.layer_name)
            if layer:
                # Check what needs to be updated before changing
                visibility_changed = layer.is_visible != visual_state.is_visible
                opacity_changed = layer.opacity != visual_state.opacity
                
                # Update layer properties
                layer.is_visible = visual_state.is_visible
                layer.opacity = visual_state.opacity
                layer.z_index = visual_state.z_index
                
                # Emit signals to update UI and map
                if visibility_changed:
                    self.layer_manager.layer_visibility_changed.emit(visual_state.layer_name, visual_state.is_visible)
                if opacity_changed:
                    self.layer_manager.layer_opacity_changed.emit(visual_state.layer_name, visual_state.opacity)
                
                logger.debug(f"Restored visual state for layer '{visual_state.layer_name}': visible={visual_state.is_visible}, opacity={visual_state.opacity}")
            else:
                logger.warning(f"Layer '{visual_state.layer_name}' not found during visual state restore")
    
    def _create_project_from_current_state(self) -> UXOProject:
        """Create project from current application state"""
        layers = list(self.layer_manager.layers.values())
        
        project = UXOProject(
            name="Untitled Project",
            layers=layers,
            layer_groups=dict(self.layer_manager.layer_groups),
            layer_order=list(self.layer_manager.layer_order),
            working_directory=Path(self._current_working_directory) if self._current_working_directory else None
        )
        
        return project
    
    def _sync_project_with_layer_manager(self):
        """Update project with current layer manager state"""
        if not self.current_project:
            return
            
        self.current_project.layers = list(self.layer_manager.layers.values())
        self.current_project.layer_groups = dict(self.layer_manager.layer_groups)
        self.current_project.layer_order = list(self.layer_manager.layer_order)
        
        # Update working directory if we have one
        if self._current_working_directory:
            self.current_project.working_directory = Path(self._current_working_directory)
        
        # Capture current data viewer state
        self.current_project.data_viewer_state = self.capture_data_viewer_state()
        
        # Capture current map visual state
        self.current_project.map_state = self.capture_map_visual_state()
        
        self.current_project.update_modified()
    
    def _restore_project_to_layer_manager(self, project: UXOProject):
        """Restore project state to layer manager"""
        # Clear current layers
        self.layer_manager.clear_all()
        
        # Restore layers
        for layer in project.layers:
            self.layer_manager.add_layer(layer)
        
        # Restore grouping
        self.layer_manager.layer_groups = dict(project.layer_groups)
        self.layer_manager.layer_order = list(project.layer_order)
        
        # Update z-indices based on order
        for i, layer_name in enumerate(project.layer_order):
            if layer_name in self.layer_manager.layers:
                self.layer_manager.layers[layer_name].z_index = i
        
        # Restore working directory
        if project.working_directory:
            self._current_working_directory = str(project.working_directory)
            self.working_directory_restored.emit(str(project.working_directory))
            logger.info(f"Working directory restored: {project.working_directory}")
        
        # Restore data viewer state
        if project.data_viewer_state:
            self.restore_data_viewer_state(project.data_viewer_state)
        
        # Restore map visual state
        if project.map_state:
            self.restore_map_visual_state(project.map_state)
    
    def _save_uxo_file(self, file_path: str, project: UXOProject) -> bool:
        """Save project to .uxo ZIP file with enhanced structure"""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # 1. Create manifest for file integrity
                manifest = {
                    "format_version": "2.0",
                    "created": datetime.now().isoformat(),
                    "files": {}
                }
                
                # 2. Save metadata
                metadata = UXOProjectMetadata(
                    name=project.name,
                    version=project.version,
                    created=project.created,
                    modified=project.modified,
                    layer_count=len(project.layers),
                    file_size_mb=0.0  # Will calculate later
                )
                
                with open(temp_path / "metadata.json", 'w') as f:
                    json.dump(metadata.to_dict(), f, indent=2)
                manifest["files"]["metadata.json"] = "project metadata"
                
                # 3. Save individual layers (NEW: Parquet + JSON metadata)
                layers_dir = temp_path / "layers"
                layers_dir.mkdir()
                
                # Create layer registry for tracking
                layer_registry = {
                    "format_version": "2.0",
                    "layers": []
                }
                
                for i, layer in enumerate(project.layers):
                    layer_id = f"layer_{i:03d}"
                    
                    # Save layer data as Parquet (replacing pickle)
                    try:
                        if hasattr(layer, 'data') and layer.data is not None:
                            import pandas as pd
                            logger.debug(f"Saving layer {layer_id} data: {type(layer.data)}")
                            if isinstance(layer.data, pd.DataFrame):
                                data_file = layers_dir / f"{layer_id}_data.parquet"
                                logger.debug(f"Saving DataFrame with shape {layer.data.shape} to {data_file}")
                                layer.data.to_parquet(data_file)
                                has_data = True
                            else:
                                # For non-DataFrame data, still use pickle as fallback
                                data_file = layers_dir / f"{layer_id}_data.pkl"
                                logger.debug(f"Saving non-DataFrame data as pickle to {data_file}")
                                with open(data_file, 'wb') as f:
                                    pickle.dump(layer.data, f)
                                has_data = True
                        else:
                            logger.debug(f"Layer {layer_id} has no data to save")
                            has_data = False
                    except Exception as e:
                        logger.warning(f"Could not save layer data for {layer_id}, using pickle fallback: {e}")
                        data_file = layers_dir / f"{layer_id}_data.pkl"
                        with open(data_file, 'wb') as f:
                            pickle.dump(layer.data if hasattr(layer, 'data') else None, f)
                        has_data = True
                    
                    # Save layer metadata as JSON (handle enum serialization)
                    layer_type = getattr(layer, 'layer_type', 'unknown')
                    geometry_type = getattr(layer, 'geometry_type', 'unknown')
                    
                    # Convert enums to strings for JSON serialization
                    layer_type_str = layer_type.value if hasattr(layer_type, 'value') else str(layer_type)
                    geometry_type_str = geometry_type.value if hasattr(geometry_type, 'value') else str(geometry_type)
                    
                    # Handle style serialization safely
                    style_obj = getattr(layer, 'style', {})
                    if hasattr(style_obj, '__dict__'):
                        style_dict = {}
                        for key, value in style_obj.__dict__.items():
                            # Convert any enum values in style to strings
                            if hasattr(value, 'value'):
                                style_dict[key] = value.value
                            else:
                                style_dict[key] = value
                    else:
                        style_dict = style_obj if isinstance(style_obj, dict) else {}
                    
                    layer_metadata = {
                        "layer_id": layer_id,
                        "name": getattr(layer, 'name', f'Layer {i}'),
                        "layer_type": layer_type_str,
                        "geometry_type": geometry_type_str,
                        "created": getattr(layer, 'created', datetime.now()).isoformat() if hasattr(layer, 'created') else datetime.now().isoformat(),
                        "data_file": data_file.name if has_data else None,
                        "has_data": has_data,
                        "style": style_dict,
                        "source_info": getattr(layer, 'source_info', {}),
                        "metadata": getattr(layer, 'metadata', {}),
                        # Visual state (also saved in map_state.json but kept here for consistency)
                        "is_visible": getattr(layer, 'is_visible', True),
                        "opacity": getattr(layer, 'opacity', 1.0),
                        "z_index": getattr(layer, 'z_index', 0),
                        # Display and traceability info
                        "display_name": getattr(layer, 'display_name', None),
                        "processing_run_id": getattr(layer, 'processing_run_id', None),
                        # Coordinate system info
                        "crs": getattr(layer, 'crs', "EPSG:4326"),
                        "bounds": getattr(layer, 'bounds', None),
                        # Processing lineage
                        "parent_layer": getattr(layer, 'parent_layer', None),
                        "processing_history": getattr(layer, 'processing_history', []),
                        "source_script": getattr(layer, 'source_script', None)
                    }
                    
                    metadata_file = layers_dir / f"{layer_id}.json"
                    with open(metadata_file, 'w') as f:
                        json.dump(layer_metadata, f, indent=2)
                    
                    # Add to registry
                    layer_registry["layers"].append({
                        "layer_id": layer_id,
                        "metadata_file": metadata_file.name,
                        "data_file": data_file.name if has_data else None,
                        "checksum": "placeholder"  # Could add actual checksums later
                    })
                
                # Save layer registry
                with open(layers_dir / "layer_registry.json", 'w') as f:
                    json.dump(layer_registry, f, indent=2)
                manifest["files"]["layers/layer_registry.json"] = "layer registry"
                
                # 4. Save unified logging system (NEW)
                logs_dir = temp_path / "logs"
                logs_dir.mkdir()
                
                # Copy current history.jsonl if it exists
                if self.history_logger and project.working_directory:
                    history_file = Path(project.working_directory) / '_project' / 'logs' / 'history.jsonl'
                    if history_file.exists():
                        import shutil
                        shutil.copy2(history_file, logs_dir / "history.jsonl")
                        manifest["files"]["logs/history.jsonl"] = "complete event log"
                
                # Enhanced processing history with more metadata
                processing_runs = {
                    "format_version": "2.0",
                    "runs": []
                }
                
                for step in project.processing_history:
                    run_data = {
                        "run_id": f"proc_{len(processing_runs['runs']):03d}",
                        "timestamp": step.timestamp.isoformat(),
                        "operation": step.operation,
                        "parameters": step.parameters,
                        "input_layers": step.input_layers,
                        "output_layers": step.output_layers,
                        "success": step.success,
                        "error_message": step.error_message,
                        "metadata": getattr(step, 'metadata', {})
                    }
                    processing_runs["runs"].append(run_data)
                
                with open(logs_dir / "processing_runs.json", 'w') as f:
                    json.dump(processing_runs, f, indent=2)
                manifest["files"]["logs/processing_runs.json"] = "enhanced processing metadata"
                
                # 5. Save annotations system (NEW)
                annotations_dir = temp_path / "annotations"
                annotations_dir.mkdir()
                
                # For now, create empty annotation files - can be enhanced later
                empty_annotations = {
                    "spatial_annotations.json": {"format_version": "2.0", "annotations": []},
                    "user_notes.json": {"format_version": "2.0", "notes": []},
                    "interpretations.json": {"format_version": "2.0", "interpretations": []}
                }
                
                for filename, content in empty_annotations.items():
                    with open(annotations_dir / filename, 'w') as f:
                        json.dump(content, f, indent=2)
                    manifest["files"][f"annotations/{filename}"] = "annotation system"
                
                # 6. Save map state
                map_state_dict = {
                    "center_lat": project.map_state.center_lat,
                    "center_lon": project.map_state.center_lon,
                    "zoom_level": project.map_state.zoom_level,
                    "base_layer": project.map_state.base_layer,
                    "projection": project.map_state.projection,
                    "extent_bounds": project.map_state.extent_bounds,
                    "layer_visual_states": []
                }
                
                # Save layer visual states
                for visual_state in project.map_state.layer_visual_states:
                    map_state_dict["layer_visual_states"].append({
                        "layer_name": visual_state.layer_name,
                        "is_visible": visual_state.is_visible,
                        "opacity": visual_state.opacity,
                        "z_index": visual_state.z_index
                    })
                
                with open(temp_path / "map_state.json", 'w') as f:
                    json.dump(map_state_dict, f, indent=2)
                
                # 7. Save data viewer state
                data_viewer_state = {
                    "open_tabs": [],
                    "active_tab_index": project.data_viewer_state.active_tab_index,
                    "has_welcome_tab": project.data_viewer_state.has_welcome_tab
                }
                
                for tab_state in project.data_viewer_state.open_tabs:
                    data_viewer_state["open_tabs"].append({
                        "file_path": tab_state.file_path,
                        "tab_title": tab_state.tab_title,
                        "current_column_filter": tab_state.current_column_filter,
                        "search_text": tab_state.search_text,
                        "selected_rows": tab_state.selected_rows
                    })
                
                with open(temp_path / "data_viewer_state.json", 'w') as f:
                    json.dump(data_viewer_state, f, indent=2)
                manifest["files"]["data_viewer_state.json"] = "data viewer state"
                
                # 8. Save project structure
                project_structure = {
                    "name": project.name,
                    "description": project.description,
                    "layer_groups": project.layer_groups,
                    "layer_order": project.layer_order,
                    "working_directory": str(project.working_directory) if project.working_directory else None,
                    "metadata": project.metadata
                }
                
                with open(temp_path / "project.json", 'w') as f:
                    json.dump(project_structure, f, indent=2)
                manifest["files"]["project.json"] = "project structure"
                
                # 9. Save manifest (final step)
                with open(temp_path / "manifest.json", 'w') as f:
                    json.dump(manifest, f, indent=2)
                
                # 10. Create ZIP file atomically (use temp file first)
                temp_zip_path = file_path + '.tmp'
                try:
                    with zipfile.ZipFile(temp_zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                        for file_path_temp in temp_path.rglob('*'):
                            if file_path_temp.is_file():
                                arcname = file_path_temp.relative_to(temp_path)
                                zf.write(file_path_temp, arcname)
                    
                    # Atomic move - only replace original if save was successful
                    os.replace(temp_zip_path, file_path)
                    
                except Exception as e:
                    # Clean up temp file if it exists
                    if os.path.exists(temp_zip_path):
                        os.remove(temp_zip_path)
                    raise e
                
                # 11. Calculate file size and log
                file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                logger.info(f"Enhanced project saved as {file_size_mb:.2f} MB file (format v2.0)")
                
                return True
                
        except Exception as e:
            logger.error(f"Error saving .uxo file: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def _load_uxo_file(self, file_path: str) -> Optional[UXOProject]:
        """Load project from .uxo ZIP file (supports both v1.0 and v2.0 formats)"""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Extract ZIP file
                with zipfile.ZipFile(file_path, 'r') as zf:
                    zf.extractall(temp_path)
                
                # Check format version
                format_version = "1.0"  # Default to old format
                if (temp_path / "manifest.json").exists():
                    with open(temp_path / "manifest.json", 'r') as f:
                        manifest = json.load(f)
                        format_version = manifest.get("format_version", "1.0")
                        logger.info(f"Loading UXO project format v{format_version}")
                
                # Load project structure
                with open(temp_path / "project.json", 'r') as f:
                    project_data = json.load(f)
                
                # Load map state
                with open(temp_path / "map_state.json", 'r') as f:
                    map_state_data = json.load(f)
                
                # Reconstruct layer visual states
                layer_visual_states = []
                for lvs_data in map_state_data.get("layer_visual_states", []):
                    visual_state = LayerVisualState(
                        layer_name=lvs_data["layer_name"],
                        is_visible=lvs_data.get("is_visible", True),
                        opacity=lvs_data.get("opacity", 1.0),
                        z_index=lvs_data.get("z_index", 0)
                    )
                    layer_visual_states.append(visual_state)
                
                map_state = MapState(
                    center_lat=map_state_data["center_lat"],
                    center_lon=map_state_data["center_lon"],
                    zoom_level=map_state_data["zoom_level"],
                    base_layer=map_state_data["base_layer"],
                    projection=map_state_data["projection"],
                    extent_bounds=map_state_data.get("extent_bounds"),
                    layer_visual_states=layer_visual_states
                )
                
                # Load layers (handle both old and new formats)
                layers = []
                layers_dir = temp_path / "layers"
                if layers_dir.exists():
                    if format_version == "2.0" and (layers_dir / "layer_registry.json").exists():
                        # New format: Load from registry
                        with open(layers_dir / "layer_registry.json", 'r') as f:
                            layer_registry = json.load(f)
                        
                        for layer_info in layer_registry.get("layers", []):
                            layer_id = layer_info["layer_id"]
                            
                            # Load layer metadata
                            metadata_file = layers_dir / layer_info["metadata_file"]
                            with open(metadata_file, 'r') as f:
                                layer_metadata = json.load(f)
                            
                            # Load layer data
                            layer_data = None
                            if layer_info.get("data_file") and layer_metadata.get("has_data"):
                                data_file = layers_dir / layer_info["data_file"]
                                logger.debug(f"Loading layer data from: {data_file}")
                                if data_file.exists():
                                    try:
                                        if data_file.suffix == '.parquet':
                                            import pandas as pd
                                            layer_data = pd.read_parquet(data_file)
                                            logger.debug(f"Loaded Parquet data with shape: {layer_data.shape if hasattr(layer_data, 'shape') else 'unknown'}")
                                        else:  # Fallback to pickle
                                            with open(data_file, 'rb') as f:
                                                layer_data = pickle.load(f)
                                            logger.debug(f"Loaded pickle data: {type(layer_data)}")
                                    except Exception as e:
                                        logger.error(f"Could not load layer data for {layer_id}: {e}")
                                        layer_data = None
                                else:
                                    logger.warning(f"Layer data file does not exist: {data_file}")
                            else:
                                logger.debug(f"No data file for layer {layer_id}: data_file={layer_info.get('data_file')}, has_data={layer_metadata.get('has_data')}")
                            
                            # Reconstruct layer object (this creates the same UXOLayer objects as before)
                            from ..ui.map.layer_types import UXOLayer, LayerStyle, LayerType, GeometryType, LayerSource
                            
                            # Reconstruct enums from string values safely
                            layer_type_str = layer_metadata.get("layer_type", "unknown")
                            geometry_type_str = layer_metadata.get("geometry_type", "point")
                            
                            try:
                                layer_type_enum = LayerType(layer_type_str)
                            except (ValueError, TypeError):
                                logger.warning(f"Unknown layer type '{layer_type_str}', defaulting to POINTS")
                                layer_type_enum = LayerType.POINTS  # Use POINTS as default
                            
                            try:
                                geometry_type_enum = GeometryType(geometry_type_str)
                            except (ValueError, TypeError):
                                logger.warning(f"Unknown geometry type '{geometry_type_str}', defaulting to POINT")
                                geometry_type_enum = GeometryType.POINT  # This exists
                            
                            # Reconstruct style object safely
                            style_data = layer_metadata.get("style", {})
                            try:
                                if style_data:
                                    layer_style = LayerStyle(**style_data)
                                else:
                                    layer_style = LayerStyle()
                            except (TypeError, ValueError) as e:
                                logger.warning(f"Could not reconstruct layer style for {layer_id}: {e}, using default")
                                layer_style = LayerStyle()
                            
                            # Create UXOLayer object with all preserved state
                            layer = UXOLayer(
                                name=layer_metadata.get("name", f"Layer {layer_id}"),
                                layer_type=layer_type_enum,
                                geometry_type=geometry_type_enum,
                                data=layer_data,
                                style=layer_style,
                                metadata=layer_metadata.get("metadata", {}),  # Required parameter
                                source=LayerSource.IMPORT,
                                # Visual state
                                is_visible=layer_metadata.get("is_visible", True),
                                opacity=layer_metadata.get("opacity", 1.0),
                                z_index=layer_metadata.get("z_index", 0),
                                # Display and traceability info
                                display_name=layer_metadata.get("display_name"),
                                processing_run_id=layer_metadata.get("processing_run_id"),
                                # Coordinate system info
                                crs=layer_metadata.get("crs", "EPSG:4326"),
                                bounds=layer_metadata.get("bounds"),
                                # Processing lineage
                                parent_layer=layer_metadata.get("parent_layer"),
                                processing_history=layer_metadata.get("processing_history", []),
                                source_script=layer_metadata.get("source_script")
                            )
                            
                            # Debug logging for created layer
                            logger.debug(f"Created layer '{layer.name}' with data: {type(layer.data) if layer.data is not None else 'None'}")
                            if hasattr(layer.data, 'shape'):
                                logger.debug(f"Layer data shape: {layer.data.shape}")
                            elif layer.data is not None:
                                logger.debug(f"Layer data length: {len(layer.data) if hasattr(layer.data, '__len__') else 'unknown'}")
                            
                            # Debug logging for layer state
                            logger.debug(f"Layer state - visible: {layer.is_visible}, opacity: {layer.opacity}, z_index: {layer.z_index}")
                            if layer.processing_run_id:
                                logger.debug(f"Layer processing run ID: {layer.processing_run_id}")
                            if layer.parent_layer:
                                logger.debug(f"Layer parent: {layer.parent_layer}")
                            
                            layers.append(layer)
                            
                    else:
                        # Old format: Load pickled layers
                        for layer_file in sorted(layers_dir.glob("layer_*.pkl")):
                            with open(layer_file, 'rb') as f:
                                layer = pickle.load(f)
                                layers.append(layer)
                
                # Load processing history (handle both formats)
                processing_history = []
                if format_version == "2.0" and (temp_path / "logs" / "processing_runs.json").exists():
                    # New format: Enhanced processing runs
                    with open(temp_path / "logs" / "processing_runs.json", 'r') as f:
                        runs_data = json.load(f)
                        for run_data in runs_data.get("runs", []):
                            step = ProcessingStep(
                                timestamp=datetime.fromisoformat(run_data["timestamp"]),
                                operation=run_data["operation"],
                                parameters=run_data["parameters"],
                                input_layers=run_data["input_layers"],
                                output_layers=run_data["output_layers"],
                                success=run_data["success"],
                                error_message=run_data.get("error_message"),
                                metadata=run_data.get("metadata", {})
                            )
                            processing_history.append(step)
                elif (temp_path / "processing_history.json").exists():
                    # Old format: Legacy processing history
                    with open(temp_path / "processing_history.json", 'r') as f:
                        history_data = json.load(f)
                        for step_data in history_data:
                            step = ProcessingStep(
                                timestamp=datetime.fromisoformat(step_data["timestamp"]),
                                operation=step_data["operation"],
                                parameters=step_data["parameters"],
                                input_layers=step_data["input_layers"],
                                output_layers=step_data["output_layers"],
                                success=step_data["success"],
                                error_message=step_data.get("error_message")
                            )
                            processing_history.append(step)
                
                # Load data viewer state
                data_viewer_state = None
                if (temp_path / "data_viewer_state.json").exists():
                    with open(temp_path / "data_viewer_state.json", 'r') as f:
                        dv_data = json.load(f)
                        
                        from .project_schema import DataViewerState, DataViewerTabState
                        
                        # Reconstruct tab states
                        tab_states = []
                        for tab_data in dv_data.get("open_tabs", []):
                            tab_state = DataViewerTabState(
                                file_path=tab_data["file_path"],
                                tab_title=tab_data["tab_title"],
                                current_column_filter=tab_data.get("current_column_filter", "All columns"),
                                search_text=tab_data.get("search_text", ""),
                                selected_rows=tab_data.get("selected_rows", [])
                            )
                            tab_states.append(tab_state)
                        
                        data_viewer_state = DataViewerState(
                            open_tabs=tab_states,
                            active_tab_index=dv_data.get("active_tab_index", 0),
                            has_welcome_tab=dv_data.get("has_welcome_tab", True)
                        )
                
                # Load metadata to get creation date
                metadata = None
                if (temp_path / "metadata.json").exists():
                    with open(temp_path / "metadata.json", 'r') as f:
                        metadata_data = json.load(f)
                        metadata = UXOProjectMetadata.from_dict(metadata_data)
                
                # Create project
                from .project_schema import DataViewerState
                project = UXOProject(
                    name=project_data["name"],
                    description=project_data.get("description", ""),
                    created=metadata.created if metadata else datetime.now(),
                    modified=metadata.modified if metadata else datetime.now(),
                    layers=layers,
                    map_state=map_state,
                    data_viewer_state=data_viewer_state if data_viewer_state else DataViewerState(),
                    layer_groups=project_data.get("layer_groups", {}),
                    layer_order=project_data.get("layer_order", []),
                    working_directory=Path(project_data["working_directory"]) if project_data.get("working_directory") else None,
                    metadata=project_data.get("metadata", {}),
                    processing_history=processing_history
                )
                
                # NEW: Restore history.jsonl to working directory if it exists in the .uxo file
                if format_version == "2.0" and project.working_directory and (temp_path / "logs" / "history.jsonl").exists():
                    working_logs_dir = Path(project.working_directory) / '_project' / 'logs'
                    working_logs_dir.mkdir(parents=True, exist_ok=True)
                    
                    history_source = temp_path / "logs" / "history.jsonl"
                    history_dest = working_logs_dir / "history.jsonl"
                    
                    try:
                        import shutil
                        shutil.copy2(history_source, history_dest)
                        logger.info(f"Restored project history log to {history_dest}")
                    except Exception as e:
                        logger.warning(f"Could not restore history log: {e}")
                
                logger.info(f"Successfully loaded UXO project (format v{format_version}) with {len(layers)} layers")
                return project
                
        except Exception as e:
            logger.error(f"Error loading .uxo file: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def export_project_info(self, file_path: str) -> bool:
        """Export project information as JSON (for debugging/inspection)"""
        try:
            if not self.current_project:
                return False
            
            info = {
                "name": self.current_project.name,
                "description": self.current_project.description,
                "created": self.current_project.created.isoformat(),
                "modified": self.current_project.modified.isoformat(),
                "layer_count": len(self.current_project.layers),
                "layer_groups": self.current_project.layer_groups,
                "layer_order": self.current_project.layer_order,
                "processing_steps": len(self.current_project.processing_history),
                "working_directory": str(self.current_project.working_directory) if self.current_project.working_directory else None,
                "data_viewer_state": {
                    "open_tabs_count": len(self.current_project.data_viewer_state.open_tabs),
                    "active_tab_index": self.current_project.data_viewer_state.active_tab_index,
                    "has_welcome_tab": self.current_project.data_viewer_state.has_welcome_tab,
                    "open_files": [tab.file_path for tab in self.current_project.data_viewer_state.open_tabs]
                },
                "map_state": {
                    "center": [self.current_project.map_state.center_lat, self.current_project.map_state.center_lon],
                    "zoom": self.current_project.map_state.zoom_level,
                    "projection": self.current_project.map_state.projection
                }
            }
            
            with open(file_path, 'w') as f:
                json.dump(info, f, indent=2)
            
            logger.info(f"Project info exported to: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting project info: {e}")
            return False 