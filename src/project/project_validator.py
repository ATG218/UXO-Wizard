"""
Project Validator - Validation and migration utilities for .uxo files
"""

import json
import zipfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from loguru import logger


class ProjectValidator:
    """Validates and migrates .uxo project files"""
    
    SUPPORTED_VERSIONS = ["1.0"]
    CURRENT_VERSION = "1.0"
    
    @classmethod
    def validate_uxo_file(cls, file_path: str) -> Tuple[bool, List[str]]:
        """
        Validate a .uxo file structure and content
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        try:
            if not Path(file_path).exists():
                errors.append(f"File does not exist: {file_path}")
                return False, errors
            
            if not file_path.endswith('.uxo'):
                errors.append("File must have .uxo extension")
                return False, errors
            
            # Check if it's a valid ZIP file
            try:
                with zipfile.ZipFile(file_path, 'r') as zf:
                    file_list = zf.namelist()
            except zipfile.BadZipFile:
                errors.append("Invalid ZIP file format")
                return False, errors
            
            # Check required files
            required_files = ['metadata.json', 'project.json', 'map_state.json']
            
            for required_file in required_files:
                if required_file not in file_list:
                    errors.append(f"Missing required file: {required_file}")
            
            # Check optional files (different files for different format versions)
            optional_v1_files = ['data_viewer_state.json', 'processing_history.json']
            optional_v2_files = ['data_viewer_state.json', 'logs/processing_runs.json', 'manifest.json']
            
            # Determine format version based on presence of manifest
            is_v2_format = 'manifest.json' in file_list
            
            if is_v2_format:
                logger.debug("Detected UXO format v2.0")
                for optional_file in optional_v2_files:
                    if optional_file not in file_list:
                        logger.debug(f"Optional file missing: {optional_file} (v2.0 format)")
            else:
                logger.debug("Detected UXO format v1.0")
                for optional_file in optional_v1_files:
                    if optional_file not in file_list:
                        logger.debug(f"Optional file missing: {optional_file} (v1.0 format)")
            
            # Validate file contents
            with zipfile.ZipFile(file_path, 'r') as zf:
                # Validate metadata.json
                try:
                    metadata_content = zf.read('metadata.json').decode('utf-8')
                    metadata = json.loads(metadata_content)
                    cls._validate_metadata(metadata, errors)
                except Exception as e:
                    errors.append(f"Error reading metadata.json: {str(e)}")
                
                # Validate project.json
                try:
                    project_content = zf.read('project.json').decode('utf-8')
                    project = json.loads(project_content)
                    cls._validate_project_structure(project, errors)
                except Exception as e:
                    errors.append(f"Error reading project.json: {str(e)}")
                
                # Validate map_state.json
                try:
                    map_state_content = zf.read('map_state.json').decode('utf-8')
                    map_state = json.loads(map_state_content)
                    cls._validate_map_state(map_state, errors)
                except Exception as e:
                    errors.append(f"Error reading map_state.json: {str(e)}")
                
                # Check layers directory (support both old and new formats)
                layer_pkl_files = [f for f in file_list if f.startswith('layers/') and f.endswith('.pkl')]
                layer_json_files = [f for f in file_list if f.startswith('layers/') and f.endswith('.json') and not f.endswith('layer_registry.json')]
                layer_registry = [f for f in file_list if f == 'layers/layer_registry.json']
                
                # Old format: has .pkl files
                # New format: has layer_registry.json and .json metadata files
                has_old_format = len(layer_pkl_files) > 0
                has_new_format = len(layer_registry) > 0 and len(layer_json_files) > 0
                
                if not has_old_format and not has_new_format:
                    errors.append("No layer files found in layers/ directory (neither old .pkl nor new registry format)")
            
            return len(errors) == 0, errors
            
        except Exception as e:
            errors.append(f"Unexpected error during validation: {str(e)}")
            return False, errors
    
    @classmethod
    def _validate_metadata(cls, metadata: Dict[str, Any], errors: List[str]):
        """Validate metadata structure"""
        required_fields = ['name', 'version', 'created', 'modified', 'layer_count']
        
        for field in required_fields:
            if field not in metadata:
                errors.append(f"Missing metadata field: {field}")
        
        if 'version' in metadata:
            version = metadata['version']
            if version not in cls.SUPPORTED_VERSIONS:
                errors.append(f"Unsupported version: {version}. Supported: {cls.SUPPORTED_VERSIONS}")
        
        # Validate timestamps
        for timestamp_field in ['created', 'modified']:
            if timestamp_field in metadata:
                try:
                    datetime.fromisoformat(metadata[timestamp_field])
                except ValueError:
                    errors.append(f"Invalid timestamp format in {timestamp_field}")
    
    @classmethod
    def _validate_project_structure(cls, project: Dict[str, Any], errors: List[str]):
        """Validate project structure"""
        required_fields = ['name', 'layer_groups', 'layer_order']
        
        for field in required_fields:
            if field not in project:
                errors.append(f"Missing project field: {field}")
        
        # Validate layer_groups structure
        if 'layer_groups' in project:
            layer_groups = project['layer_groups']
            if not isinstance(layer_groups, dict):
                errors.append("layer_groups must be a dictionary")
            else:
                for group_name, layer_list in layer_groups.items():
                    if not isinstance(layer_list, list):
                        errors.append(f"Layer group '{group_name}' must contain a list")
        
        # Validate layer_order
        if 'layer_order' in project:
            layer_order = project['layer_order']
            if not isinstance(layer_order, list):
                errors.append("layer_order must be a list")
    
    @classmethod
    def _validate_map_state(cls, map_state: Dict[str, Any], errors: List[str]):
        """Validate map state structure"""
        required_fields = ['center_lat', 'center_lon', 'zoom_level']
        
        for field in required_fields:
            if field not in map_state:
                errors.append(f"Missing map_state field: {field}")
        
        # Validate coordinate ranges
        if 'center_lat' in map_state:
            lat = map_state['center_lat']
            if not isinstance(lat, (int, float)) or not -90 <= lat <= 90:
                errors.append("center_lat must be between -90 and 90")
        
        if 'center_lon' in map_state:
            lon = map_state['center_lon']
            if not isinstance(lon, (int, float)) or not -180 <= lon <= 180:
                errors.append("center_lon must be between -180 and 180")
        
        if 'zoom_level' in map_state:
            zoom = map_state['zoom_level']
            if not isinstance(zoom, int) or not 0 <= zoom <= 25:
                errors.append("zoom_level must be between 0 and 25")
    
    @classmethod
    def get_project_info(cls, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Get basic project information without fully loading the project
        
        Returns:
            Dictionary with project info or None if error
        """
        try:
            with zipfile.ZipFile(file_path, 'r') as zf:
                # Read metadata
                metadata_content = zf.read('metadata.json').decode('utf-8')
                metadata = json.loads(metadata_content)
                
                # Read project structure
                project_content = zf.read('project.json').decode('utf-8')
                project = json.loads(project_content)
                
                # Calculate file size
                file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)
                
                return {
                    "name": metadata.get("name", "Unknown"),
                    "version": metadata.get("version", "Unknown"),
                    "created": metadata.get("created"),
                    "modified": metadata.get("modified"),
                    "layer_count": metadata.get("layer_count", 0),
                    "file_size_mb": round(file_size_mb, 2),
                    "description": project.get("description", ""),
                    "layer_groups": project.get("layer_groups", {}),
                    "working_directory": project.get("working_directory")
                }
                
        except Exception as e:
            logger.error(f"Error reading project info from {file_path}: {e}")
            return None
    
    @classmethod
    def migrate_project(cls, file_path: str, target_version: str = None) -> bool:
        """
        Migrate project to newer version
        
        Args:
            file_path: Path to .uxo file
            target_version: Target version (defaults to current)
            
        Returns:
            True if migration successful
        """
        if target_version is None:
            target_version = cls.CURRENT_VERSION
        
        try:
            # Get current version
            info = cls.get_project_info(file_path)
            if not info:
                logger.error(f"Cannot read project info for migration: {file_path}")
                return False
            
            current_version = info.get("version", "1.0")
            
            if current_version == target_version:
                logger.info(f"Project already at target version {target_version}")
                return True
            
            logger.info(f"Migrating project from {current_version} to {target_version}")
            
            # Create backup
            backup_path = file_path + f".backup_{current_version}"
            import shutil
            shutil.copy2(file_path, backup_path)
            logger.info(f"Created backup: {backup_path}")
            
            # Perform migration based on versions
            if current_version == "1.0" and target_version == "1.0":
                # No migration needed
                return True
            
            # Add future migration logic here
            logger.warning(f"Migration from {current_version} to {target_version} not implemented")
            return False
            
        except Exception as e:
            logger.error(f"Error during project migration: {e}")
            return False
    
    @classmethod
    def repair_project(cls, file_path: str) -> bool:
        """
        Attempt to repair a corrupted .uxo file
        
        Returns:
            True if repair successful
        """
        try:
            is_valid, errors = cls.validate_uxo_file(file_path)
            
            if is_valid:
                logger.info("Project file is valid, no repair needed")
                return True
            
            logger.info(f"Attempting to repair project file with {len(errors)} errors")
            
            # Create backup
            backup_path = file_path + ".backup_before_repair"
            import shutil
            shutil.copy2(file_path, backup_path)
            
            # Attempt basic repairs
            with zipfile.ZipFile(file_path, 'a') as zf:
                file_list = zf.namelist()
                
                # Add missing required files with defaults
                if 'metadata.json' not in file_list:
                    default_metadata = {
                        "name": "Repaired Project",
                        "version": cls.CURRENT_VERSION,
                        "created": datetime.now().isoformat(),
                        "modified": datetime.now().isoformat(),
                        "layer_count": 0,
                        "file_size_mb": 0.0
                    }
                    zf.writestr('metadata.json', json.dumps(default_metadata, indent=2))
                    logger.info("Added missing metadata.json")
                
                if 'project.json' not in file_list:
                    default_project = {
                        "name": "Repaired Project",
                        "description": "Project repaired by validator",
                        "layer_groups": {
                            "Survey Data": [],
                            "Other": []
                        },
                        "layer_order": [],
                        "working_directory": None,
                        "metadata": {}
                    }
                    zf.writestr('project.json', json.dumps(default_project, indent=2))
                    logger.info("Added missing project.json")
                
                if 'map_state.json' not in file_list:
                    default_map_state = {
                        "center_lat": 63.8167,
                        "center_lon": 9.3667,
                        "zoom_level": 12,
                        "base_layer": "OpenStreetMap",
                        "projection": "EPSG:4326",
                        "extent_bounds": None
                    }
                    zf.writestr('map_state.json', json.dumps(default_map_state, indent=2))
                    logger.info("Added missing map_state.json")
            
            # Validate again
            is_valid_after, remaining_errors = cls.validate_uxo_file(file_path)
            
            if is_valid_after:
                logger.info("Project file successfully repaired")
                return True
            else:
                logger.warning(f"Repair partially successful. Remaining errors: {remaining_errors}")
                return False
                
        except Exception as e:
            logger.error(f"Error during project repair: {e}")
            return False 