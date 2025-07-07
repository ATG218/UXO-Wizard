"""
Base classes for all data processors
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable
from PySide6.QtCore import Signal, QThread
import pandas as pd
from loguru import logger
import time
import os
import glob
import importlib.util
import sys
import numpy as np

# Import layer system components for layer generation
try:
    from ..ui.map.layer_types import UXOLayer, LayerType, GeometryType, LayerStyle, LayerSource, NORWEGIAN_CRS
    LAYER_SYSTEM_AVAILABLE = True
except ImportError:
    logger.warning("Layer system not available - layer_types module not found")
    LAYER_SYSTEM_AVAILABLE = False


class ProcessingError(Exception):
    """Custom exception for processing errors"""
    pass


@dataclass
class OutputFile:
    """Container for script-generated output files"""
    file_path: str                      # Path to generated file
    file_type: str                      # "png", "geotiff", "csv", "html", etc.
    description: str                    # "Flight path visualization", "Grid data", etc.
    metadata: Dict[str, Any] = field(default_factory=dict)  # File-specific metadata


@dataclass
class LayerOutput:
    """Container for layer data ready for future layer system consumption"""
    layer_type: str                     # "flight_lines", "grid_visualization", "points", etc.
    data: Any                          # Layer data (DataFrame, coordinates, etc.)
    style_info: Dict[str, Any] = field(default_factory=dict)  # Styling information
    metadata: Dict[str, Any] = field(default_factory=dict)    # Layer-specific metadata


@dataclass
class ProcessingResult:
    """Container for processing results with flexible output support"""
    success: bool
    data: Optional[pd.DataFrame] = None
    metadata: Dict[str, Any] = None
    error_message: Optional[str] = None
    processing_time: float = 0.0
    output_file_path: Optional[str] = None        # Primary generated file path (backward compatibility)
    processing_script: Optional[str] = None       # Script/algorithm name
    input_file_path: Optional[str] = None         # Original input file
    export_format: str = "csv"                    # Primary output format (backward compatibility)
    
    # New flexible output system
    processor_type: Optional[str] = None          # "magnetic", "gamma", etc.
    script_id: Optional[str] = None              # Which script was used
    output_files: List[OutputFile] = field(default_factory=list)      # All generated files
    layer_outputs: List[LayerOutput] = field(default_factory=list)    # Data for future layer system
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def add_output_file(self, file_path: str, file_type: str, description: str, metadata: Dict[str, Any] = None):
        """Add an output file to the result"""
        self.output_files.append(OutputFile(
            file_path=file_path,
            file_type=file_type,
            description=description,
            metadata=metadata or {}
        ))
    
    def add_layer_output(self, layer_type: str, data: Any, style_info: Dict[str, Any] = None, metadata: Dict[str, Any] = None):
        """Add layer data for future layer system consumption"""
        self.layer_outputs.append(LayerOutput(
            layer_type=layer_type,
            data=data,
            style_info=style_info or {},
            metadata=metadata or {}
        ))


class ScriptInterface(ABC):
    """Abstract interface that all processing scripts must implement"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable script name for UI display"""
        pass
    
    @property  
    @abstractmethod
    def description(self) -> str:
        """Script description for UI display"""
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Return script-specific parameters for UI generation
        
        Returns:
            Dict with parameter structure compatible with ParameterWidget
            Example:
            {
                'processing_options': {
                    'threshold': {
                        'value': 2.0,
                        'type': 'float',
                        'min': 0.1,
                        'max': 10.0,
                        'description': 'Detection threshold'
                    },
                    'output_format': {
                        'value': 'png',
                        'type': 'choice',
                        'choices': ['png', 'geotiff', 'csv'],
                        'description': 'Output file format'
                    }
                }
            }
        """
        pass
    
    @abstractmethod
    def execute(self, data: pd.DataFrame, params: Dict[str, Any], 
                progress_callback: Optional[Callable] = None, input_file_path: Optional[str] = None) -> ProcessingResult:
        """Execute the script with provided data and parameters
        
        Args:
            data: Input DataFrame to process
            params: Processing parameters from UI
            progress_callback: Optional callback for progress updates
            input_file_path: Optional path to the original input file
            
        Returns:
            ProcessingResult with success status, output files, and layer data
        """
        pass
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate that data is suitable for this script
        
        Args:
            data: Input DataFrame to validate
            
        Returns:
            True if data is valid for this script
            
        Raises:
            ProcessingError: If data is invalid with description
        """
        # Default implementation - override in scripts if needed
        if data.empty:
            raise ProcessingError("No data provided")
        return True


class ProcessingWorker(QThread):
    """Worker thread for background processing"""
    
    # Signals
    progress = Signal(int)  # Progress percentage (0-100)
    status = Signal(str)    # Status message
    finished = Signal(ProcessingResult)  # Final result
    error = Signal(str)     # Error message
    layer_created = Signal(object)  # UXOLayer for automatic layer registration
    
    def __init__(self, processor_func: Callable, data: pd.DataFrame, params: Dict[str, Any], 
                 input_file_path: Optional[str] = None, processor_instance: 'BaseProcessor' = None):
        super().__init__()
        self.processor_func = processor_func
        self.data = data
        self.params = params
        self.input_file_path = input_file_path
        self.processor_instance = processor_instance  # Reference to processor for layer creation
        self._is_cancelled = False
        
    def run(self):
        """Run the processing in background thread"""
        try:
            start_time = time.time()
            
            # Create progress callback
            def progress_callback(value: int, message: str = ""):
                if self._is_cancelled:
                    raise ProcessingError("Processing cancelled by user")
                self.progress.emit(value)
                if message:
                    self.status.emit(message)
            
            # Run the processor with progress callback
            result = self.processor_func(
                self.data, 
                self.params,
                progress_callback=progress_callback,
                input_file_path=self.input_file_path
            )
            
            result.processing_time = time.time() - start_time
            
            # Convert LayerOutput objects to UXOLayer objects and emit them
            if result.success and result.layer_outputs and LAYER_SYSTEM_AVAILABLE:
                logger.info(f"Processing {len(result.layer_outputs)} layer outputs")
                if self.processor_instance:
                    for layer_output in result.layer_outputs:
                        try:
                            uxo_layer = self._convert_layer_output_to_uxo_layer(layer_output, result)
                            if uxo_layer:
                                self.layer_created.emit(uxo_layer)
                                logger.info(f"Created and emitted layer: {uxo_layer.name}")
                        except Exception as e:
                            logger.error(f"Failed to create layer: {str(e)}")
                else:
                    logger.warning("No processor instance available for layer creation")
            
            self.finished.emit(result)
            
        except Exception as e:
            logger.error(f"Processing error: {str(e)}")
            self.error.emit(str(e))
            self.finished.emit(ProcessingResult(
                success=False,
                error_message=str(e)
            ))
    
    def _convert_layer_output_to_uxo_layer(self, layer_output: LayerOutput, result: ProcessingResult) -> Optional['UXOLayer']:
        """Convert LayerOutput to UXOLayer using processor's layer creation methods"""
        try:
            # Use layer_name from metadata if provided, otherwise auto-generate
            if layer_output.metadata and 'layer_name' in layer_output.metadata:
                name = layer_output.metadata['layer_name']
            else:
                name = f"{result.processor_type or 'Processing'} - {layer_output.layer_type.replace('_', ' ').title()}"
            
            # Convert style_info to proper style_overrides format
            style_overrides = {}
            if layer_output.style_info:
                # Map common style_info keys to LayerStyle attributes
                style_mapping = {
                    'color': 'point_color',
                    'size': 'point_size',
                    'opacity': 'point_opacity',
                    'line_color': 'line_color',
                    'line_width': 'line_width',
                    'line_opacity': 'line_opacity',
                    'fill_color': 'fill_color',
                    'color_field': 'color_field',
                    'color_scheme': 'color_ramp',
                    'use_graduated_colors': 'use_graduated_colors'
                }
                
                for old_key, new_key in style_mapping.items():
                    if old_key in layer_output.style_info:
                        style_overrides[new_key] = layer_output.style_info[old_key]
            
            # Enhanced metadata from layer output and processing result
            enhanced_metadata = {
                'processing_script': result.script_id or result.processing_script,
                'input_file': result.input_file_path,
                'layer_output_metadata': layer_output.metadata,
                'processing_metadata': result.metadata or {}
            }
            
            # Create the UXOLayer using processor's method
            uxo_layer = self.processor_instance.create_layer(
                layer_type=layer_output.layer_type,
                data=layer_output.data,
                name=name,
                style_overrides=style_overrides,
                metadata=enhanced_metadata
            )
            
            return uxo_layer
            
        except Exception as e:
            logger.error(f"Failed to convert LayerOutput to UXOLayer: {str(e)}")
            return None
    
    def cancel(self):
        """Cancel the processing"""
        self._is_cancelled = True


class BaseProcessor(ABC):
    """Abstract base class for all data processors with script framework support"""
    
    def __init__(self):
        self.name = "Base Processor"
        self.description = "Base processor class"
        self.supported_columns = []
        self.required_columns = []
        
        # Script framework attributes
        self.processor_type = self._get_processor_type()
        self.available_scripts = self._discover_scripts()
        self.current_script = None
        
        # Generate parameters (now includes scripts)
        self.parameters = self._define_parameters()
    
    def _get_processor_type(self) -> str:
        """Return processor type for script discovery"""
        # Extract processor type from class name (e.g., MagneticProcessor -> magnetic)
        class_name = self.__class__.__name__
        if class_name.endswith('Processor'):
            return class_name[:-9].lower()  # Remove 'Processor' and lowercase
        return class_name.lower()
    
    def _discover_scripts(self) -> Dict[str, 'ScriptInterface']:
        """Auto-discover scripts specific to this processor type"""
        scripts = {}
        
        try:
            # Get path to processor-specific scripts directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            scripts_dir = os.path.join(current_dir, 'scripts', self.processor_type)
            
            if not os.path.exists(scripts_dir):
                logger.debug(f"Scripts directory not found: {scripts_dir}")
                return scripts
            
            # Find all Python files in the scripts directory
            script_files = glob.glob(os.path.join(scripts_dir, '*.py'))
            
            for script_file in script_files:
                # Skip __init__.py files
                if os.path.basename(script_file).startswith('__'):
                    continue
                
                try:
                    # Import the script module
                    script_name = os.path.splitext(os.path.basename(script_file))[0]
                    
                    # Create a unique module name that reflects its path from the src directory
                    module_name = f"src.processing.scripts.{self.processor_type}.{script_name}"
                    
                    # Load module from file with the unique name
                    spec = importlib.util.spec_from_file_location(module_name, script_file)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        sys.modules[module_name] = module
                        spec.loader.exec_module(module)
                        
                        # Look for SCRIPT_CLASS attribute
                        if hasattr(module, 'SCRIPT_CLASS'):
                            script_instance = module.SCRIPT_CLASS()
                            if isinstance(script_instance, ScriptInterface):
                                scripts[script_name] = script_instance
                                logger.debug(f"Loaded script: {script_name} for {self.processor_type}")
                            else:
                                logger.warning(f"Script {script_name} does not implement ScriptInterface")
                        else:
                            logger.debug(f"Script {script_name} has no SCRIPT_CLASS attribute")
                            
                except Exception as e:
                    logger.error(f"Failed to load script {script_file}: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Failed to discover scripts for {self.processor_type}: {str(e)}")
        
        return scripts
    
    def get_available_scripts(self) -> List[Dict[str, str]]:
        """Return list of scripts with metadata for UI dropdown"""
        return [
            {
                'id': script_id,
                'name': script.name,
                'description': script.description
            }
            for script_id, script in self.available_scripts.items()
        ]
    
    def set_script(self, script_id: str) -> None:
        """Set current script and update parameters"""
        if script_id in self.available_scripts:
            self.current_script = script_id
            self.parameters = self._generate_script_parameters(script_id)
            logger.debug(f"Set active script to: {script_id}")
        else:
            logger.warning(f"Script not found: {script_id}")
    
    def _generate_script_parameters(self, script_id: str) -> Dict[str, Any]:
        """Generate combined base + script-specific parameters"""
        # Start with base parameters
        base_params = self._define_base_parameters()
        
        # Add script selection parameter
        if self.available_scripts:
            base_params['script_selection'] = {
                'script_name': {
                    'value': script_id,
                    'type': 'choice',
                    'choices': list(self.available_scripts.keys()),
                    'description': 'Select processing script'
                }
            }
        
        # Get script-specific parameters
        script = self.available_scripts.get(script_id)
        if script:
            try:
                script_params = script.get_parameters()
                # Merge parameters - script parameters take precedence
                base_params.update(script_params)
            except Exception as e:
                logger.error(f"Failed to get parameters for script {script_id}: {str(e)}")
        
        return base_params
    
    def _define_base_parameters(self) -> Dict[str, Any]:
        """Define base parameters common to all scripts"""
        return {
            'output_settings': {
                'export_format': {
                    'value': 'csv',
                    'type': 'choice',
                    'choices': ['csv', 'xlsx', 'json'],
                    'description': 'Primary output file format'
                }
            }
        }
        
    def _define_parameters(self) -> Dict[str, Any]:
        """Define processing parameters with defaults and metadata"""
        # If we have scripts, set up script-based parameters
        if self.available_scripts:
            # Use first available script as default
            default_script = list(self.available_scripts.keys())[0]
            return self._generate_script_parameters(default_script)
        else:
            # Fallback to base parameters if no scripts available
            return self._define_base_parameters()
    
    @abstractmethod
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate that data has required columns and format"""
        # Check required columns
        missing_cols = [col for col in self.required_columns if col not in data.columns]
        if missing_cols:
            raise ProcessingError(f"Missing required columns: {missing_cols}")
        return True
    
    def process(self, data: pd.DataFrame, params: Dict[str, Any], 
                progress_callback: Optional[Callable] = None, input_file_path: Optional[str] = None) -> ProcessingResult:
        """Enhanced process method that handles script execution"""
        try:
            if progress_callback:
                progress_callback(0, "Starting processing...")
            
            # Validate data
            self.validate_data(data)
            
            # Get selected script
            script_id = params.get('script_selection', {}).get('script_name', {}).get('value')
            
            # If no script specified, try current script or first available
            if not script_id:
                if self.current_script:
                    script_id = self.current_script
                elif self.available_scripts:
                    script_id = list(self.available_scripts.keys())[0]
            
            # Execute script if available
            if script_id and script_id in self.available_scripts:
                if progress_callback:
                    progress_callback(10, f"Executing script: {script_id}")
                
                script = self.available_scripts[script_id]
                
                # Validate data against script requirements
                script.validate_data(data)
                
                # Execute script
                result = script.execute(data, params, progress_callback, input_file_path)
                
                # Ensure result has processor metadata
                result.processor_type = self.processor_type
                result.script_id = script_id
                
                # Register any output files
                if result.output_files:
                    for output_file in result.output_files:
                        self._register_output_file(output_file)
                
                if progress_callback:
                    progress_callback(100, "Processing complete!")
                
                return result
            else:
                # Fallback to legacy processing if no scripts
                return self._legacy_process(data, params, progress_callback)
                
        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            return ProcessingResult(
                success=False,
                error_message=str(e),
                processor_type=self.processor_type,
                script_id=script_id
            )
    
    def _register_output_file(self, output_file: OutputFile):
        """Register output file with processor for tracking"""
        logger.info(f"Generated {output_file.file_type}: {output_file.description} at {output_file.file_path}")
    
    def _legacy_process(self, data: pd.DataFrame, params: Dict[str, Any], 
                       progress_callback: Optional[Callable] = None, input_file_path: Optional[str] = None) -> ProcessingResult:
        """Fallback processing method for processors without scripts"""
        # This will be overridden by processors that need legacy support
        _ = data, params, progress_callback, input_file_path  # Suppress unused parameter warnings
        raise ProcessingError("No processing scripts available and no legacy processing implemented")
    
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Common preprocessing steps"""
        # Remove NaN values
        data = data.dropna()
        
        # Sort by timestamp if available
        if 'timestamp' in data.columns:
            data = data.sort_values('timestamp')
            
        return data
    
    def detect_columns(self, data: pd.DataFrame) -> Dict[str, str]:
        """Auto-detect relevant columns in the data"""
        detected = {}
        
        # Common patterns for different data types
        patterns = {
            'latitude': ['lat', 'latitude', 'y', 'northing', 'utm_northing'],
            'longitude': ['lon', 'lng', 'longitude', 'x', 'easting', 'utm_easting'],
            'timestamp': ['time', 'timestamp', 'datetime', 'date'],
            'altitude': ['alt', 'altitude', 'height', 'z'],
            'magnetic': ['mag', 'magnetic', 'btotal', 'field'],
            'gamma': ['gamma', 'counts', 'CountRate', 'U238'],
            'gpr': ['gpr', 'radar', 'amplitude', 'signal']
        }
        
        for col_type, keywords in patterns.items():
            for col in data.columns:
                col_lower = col.lower()
                if any(kw in col_lower for kw in keywords):
                    detected[col_type] = col
                    break
                    
        return detected
    
    # ===== UNIVERSAL LAYER GENERATION METHODS =====
    # These methods can be inherited by all processor types
    
    def create_layer(self, layer_type: str, data: Any, name: str = None, 
                    style_overrides: Dict[str, Any] = None, metadata: Dict[str, Any] = None,
                    geometry_type: GeometryType = None) -> 'UXOLayer':
        """
        Universal layer creation method inherited by all processors
        
        Args:
            layer_type: Type of layer ('points', 'raster', 'vector', etc.)
            data: Layer data (DataFrame, numpy array, etc.)
            name: Layer name (auto-generated if None)
            style_overrides: Custom styling options
            metadata: Additional metadata
            geometry_type: Specific geometry type (auto-detected if None)
            
        Returns:
            UXOLayer object ready for map display
        """
        if not LAYER_SYSTEM_AVAILABLE:
            logger.warning("Layer system not available - cannot create layer")
            return None
            
        # Auto-generate name if not provided
        if name is None:
            name = f"{self.name} - {layer_type.replace('_', ' ').title()}"
        
        # Auto-detect geometry type if not specified
        if geometry_type is None:
            geometry_type = self._detect_layer_geometry(data, layer_type)
        
        # Map layer types to UXOLayer types
        layer_type_mapping = {
            'points': LayerType.POINTS,
            'point_data': LayerType.POINTS,
            'raster': LayerType.RASTER,
            'grid_visualization': LayerType.RASTER,
            'heatmap': LayerType.RASTER,
            'vector': LayerType.VECTOR,
            'flight_lines': LayerType.VECTOR,
            'flight_path': LayerType.VECTOR,
            'processed': LayerType.POINTS,  # Processed data is still point data
            'annotation': LayerType.ANNOTATION
        }
        
        uxo_layer_type = layer_type_mapping.get(layer_type, LayerType.POINTS)
        
        # Generate appropriate styling
        style = self._generate_layer_style(data, layer_type, style_overrides)
        
        # Calculate bounds for the data
        bounds = self._calculate_layer_bounds(data, geometry_type)
        
        # Create comprehensive metadata
        layer_metadata = {
            'processor_type': self.processor_type,
            'data_type': type(data).__name__,
            'creation_timestamp': time.time(),
            'coordinate_system': self._detect_coordinate_system(data),
            'data_summary': self._generate_data_summary(data)
        }
        
        # Merge with provided metadata
        if metadata:
            layer_metadata.update(metadata)
        
        # Create UXOLayer
        return UXOLayer(
            name=name,
            layer_type=uxo_layer_type,
            data=data,
            geometry_type=geometry_type,
            style=style,
            metadata=layer_metadata,
            source=LayerSource.PROCESSING,
            bounds=bounds,
            processing_history=[self.processor_type]
        )
    
    def create_point_layer(self, data: pd.DataFrame, name: str = None, 
                          color_field: str = None, style_overrides: Dict[str, Any] = None,
                          **kwargs) -> 'UXOLayer':
        """
        Create point layer with coordinate auto-detection
        
        Args:
            data: DataFrame with coordinate columns
            name: Layer name
            color_field: Column for graduated colors
            style_overrides: Custom styling
            
        Returns:
            UXOLayer with point geometry
        """
        # Detect coordinate columns
        coord_info = self.detect_columns(data)
        
        if 'latitude' not in coord_info or 'longitude' not in coord_info:
            logger.warning("Cannot create point layer - coordinate columns not found")
            return None
        
        # Create enhanced metadata
        metadata = {
            'coordinate_columns': coord_info,
            'total_points': len(data),
            'color_field': color_field
        }
        metadata.update(kwargs.get('metadata', {}))
        
        # Enhanced styling for points
        if style_overrides is None:
            style_overrides = {}
        
        if color_field and color_field in data.columns:
            style_overrides.update({
                'use_graduated_colors': True,
                'color_field': color_field
            })
        
        return self.create_layer(
            layer_type='points',
            data=data,
            name=name,
            geometry_type=GeometryType.POINT,
            style_overrides=style_overrides,
            metadata=metadata
        )
    
    def create_raster_layer(self, data: np.ndarray, bounds: List[float], 
                           name: str = None, style_overrides: Dict[str, Any] = None,
                           **kwargs) -> 'UXOLayer':
        """
        Create raster layer for interpolated grids
        
        Args:
            data: Numpy array with grid data
            bounds: Spatial bounds [min_x, min_y, max_x, max_y]
            name: Layer name
            style_overrides: Custom styling
            
        Returns:
            UXOLayer with raster geometry
        """
        metadata = {
            'grid_shape': data.shape,
            'data_range': [float(np.nanmin(data)), float(np.nanmax(data))],
            'bounds': bounds
        }
        metadata.update(kwargs.get('metadata', {}))
        
        # Enhanced styling for rasters
        if style_overrides is None:
            style_overrides = {}
        
        style_overrides.update({
            'use_graduated_colors': True,
            'fill_opacity': 0.7
        })
        
        return self.create_layer(
            layer_type='raster',
            data=data,
            name=name,
            geometry_type=GeometryType.RASTER,
            style_overrides=style_overrides,
            metadata=metadata
        )
    
    def create_vector_layer(self, data: pd.DataFrame, geometry_type: GeometryType,
                           name: str = None, style_overrides: Dict[str, Any] = None,
                           **kwargs) -> 'UXOLayer':
        """
        Create vector layer for flight paths, boundaries, etc.
        
        Args:
            data: DataFrame with coordinate sequences
            geometry_type: LINE, POLYGON, etc.
            name: Layer name
            style_overrides: Custom styling
            
        Returns:
            UXOLayer with vector geometry
        """
        metadata = {
            'vector_type': geometry_type.value,
            'feature_count': len(data)
        }
        metadata.update(kwargs.get('metadata', {}))
        
        return self.create_layer(
            layer_type='vector',
            data=data,
            name=name,
            geometry_type=geometry_type,
            style_overrides=style_overrides,
            metadata=metadata
        )
    
    def _detect_layer_geometry(self, data: Any, layer_type: str) -> GeometryType:
        """Auto-detect geometry type from data structure"""
        if isinstance(data, np.ndarray) and data.ndim >= 2:
            return GeometryType.RASTER
        elif isinstance(data, pd.DataFrame):
            if layer_type in ['flight_lines', 'flight_path']:
                return GeometryType.LINE
            elif any(col in str(data.columns).lower() for col in ['lat', 'lon', 'x', 'y', 'northing', 'easting']):
                return GeometryType.POINT
        elif isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], (list, tuple)) and len(data[0]) == 2:
                return GeometryType.LINE
        
        # Default fallback
        return GeometryType.POINT
    
    def _generate_layer_style(self, data: Any, layer_type: str, 
                             style_overrides: Dict[str, Any] = None) -> LayerStyle:
        """Generate appropriate styling based on processor type and data"""
        style = LayerStyle()
        
        # Processor-specific styling
        if self.processor_type == 'magnetic':
            style = self._create_magnetic_style(data, layer_type)
        elif self.processor_type == 'gamma':
            style = self._create_gamma_style(data, layer_type)
        elif self.processor_type == 'gpr':
            style = self._create_gpr_style(data, layer_type)
        else:
            # Default styling
            style = self._create_default_style(layer_type)
        
        # Apply overrides
        if style_overrides:
            for key, value in style_overrides.items():
                if hasattr(style, key):
                    setattr(style, key, value)
        
        return style
    
    def _create_magnetic_style(self, data: Any, layer_type: str) -> LayerStyle:
        """Create magnetic-specific styling"""
        if layer_type in ['points', 'point_data']:
            return LayerStyle(
                point_color="#0066CC",
                point_size=4,
                point_opacity=0.8,
                enable_clustering=True,
                cluster_distance=25,
                use_graduated_colors=True,
                color_ramp=["#000080", "#0000FF", "#00FFFF", "#00FF00", "#FFFF00", "#FF8000", "#FF0000"]
            )
        elif layer_type in ['raster', 'grid_visualization', 'heatmap']:
            return LayerStyle(
                use_graduated_colors=True,
                color_ramp=["#000080", "#0000FF", "#00FFFF", "#00FF00", "#FFFF00", "#FF8000", "#FF0000"],
                fill_opacity=0.8
            )
        elif layer_type in ['flight_lines', 'flight_path']:
            return LayerStyle(
                line_color="#FF6600",
                line_width=2,
                line_opacity=1.0,
                show_labels=True
            )
        
        return LayerStyle(point_color="#0066CC", point_size=4)
    
    def _create_gamma_style(self, data: Any, layer_type: str) -> LayerStyle:
        """Create gamma radiation-specific styling"""
        return LayerStyle(
            point_color="#00CC66",
            point_size=5,
            use_graduated_colors=True,
            color_ramp=["#004400", "#008800", "#00CC00", "#CCCC00", "#CC8800", "#CC0000"]
        )
    
    def _create_gpr_style(self, data: Any, layer_type: str) -> LayerStyle:
        """Create GPR-specific styling"""
        return LayerStyle(
            point_color="#CC6600",
            point_size=4,
            use_graduated_colors=True,
            color_ramp=["#000066", "#0066CC", "#66CCFF", "#CCCCCC", "#FFCC66", "#CC6600"]
        )
    
    def _create_default_style(self, layer_type: str) -> LayerStyle:
        """Create default styling for unknown processor types"""
        if layer_type in ['points', 'point_data']:
            return LayerStyle(point_color="#666666", point_size=4)
        elif layer_type in ['raster', 'grid_visualization']:
            return LayerStyle(use_graduated_colors=True, fill_opacity=0.7)
        elif layer_type in ['flight_lines', 'vector']:
            return LayerStyle(line_color="#333333", line_width=2)
        
        return LayerStyle()
    
    def _calculate_layer_bounds(self, data: Any, geometry_type: GeometryType) -> Optional[List[float]]:
        """Calculate spatial bounds for Norwegian UTM zones"""
        # Handle raster data (dict with bounds)
        if isinstance(data, dict) and 'bounds' in data:
            return data['bounds']
        
        if isinstance(data, pd.DataFrame):
            coord_info = self.detect_columns(data)
            lat_col = coord_info.get('latitude')
            lon_col = coord_info.get('longitude')
            
            if lat_col and lon_col:
                try:
                    min_lat, max_lat = data[lat_col].min(), data[lat_col].max()
                    min_lon, max_lon = data[lon_col].min(), data[lon_col].max()
                    return [float(min_lon), float(min_lat), float(max_lon), float(max_lat)]
                except:
                    pass
            
            # Try UTM columns
            utm_east_cols = [col for col in data.columns if 'easting' in col.lower()]
            utm_north_cols = [col for col in data.columns if 'northing' in col.lower()]
            
            if utm_east_cols and utm_north_cols:
                try:
                    min_east, max_east = data[utm_east_cols[0]].min(), data[utm_east_cols[0]].max()
                    min_north, max_north = data[utm_north_cols[0]].min(), data[utm_north_cols[0]].max()
                    return [float(min_east), float(min_north), float(max_east), float(max_north)]
                except:
                    pass
        
        return None
    
    def _detect_coordinate_system(self, data: Any) -> str:
        """Detect coordinate system from data"""
        if isinstance(data, pd.DataFrame):
            coord_info = self.detect_columns(data)
            
            # Check for UTM columns
            utm_cols = [col for col in data.columns if any(utm_word in col.lower() 
                       for utm_word in ['utm', 'easting', 'northing'])]
            if utm_cols:
                # Try to detect UTM zone from data bounds or column names
                for zone in [32, 33, 34, 35]:  # Norwegian UTM zones
                    if str(zone) in ' '.join(data.columns):
                        return f"EPSG:258{zone}"
                return "EPSG:25833"  # Default to UTM 33N for Norway
            
            # Check for lat/lon
            if 'latitude' in coord_info and 'longitude' in coord_info:
                return "EPSG:4326"  # WGS84
        
        return "EPSG:4326"  # Default
    
    def _generate_data_summary(self, data: Any) -> Dict[str, Any]:
        """Generate summary statistics for data"""
        summary = {}
        
        if isinstance(data, pd.DataFrame):
            summary['rows'] = len(data)
            summary['columns'] = len(data.columns)
            summary['numeric_columns'] = len(data.select_dtypes(include=[np.number]).columns)
        elif isinstance(data, np.ndarray):
            summary['shape'] = data.shape
            summary['dtype'] = str(data.dtype)
            summary['size'] = data.size
        
        return summary
    
    def create_animation_frames(self, data: pd.DataFrame, 
                              num_frames: int = 10) -> List[pd.DataFrame]:
        """Create animation frames for progressive visualization"""
        frames = []
        step = max(1, len(data) // num_frames)
        
        for i in range(0, len(data), step):
            frames.append(data.iloc[:i+step])
            
        return frames 