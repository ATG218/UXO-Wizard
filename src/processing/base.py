"""
Base classes for all data processors
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable
from matplotlib.figure import Figure
from PySide6.QtCore import Signal, QThread
import pandas as pd
from loguru import logger
import time
import os
import glob
import importlib.util
import sys
import numpy as np
from scipy.interpolate import griddata
from scipy.spatial.distance import pdist

# Try to import Numba for JIT compilation
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    # Define a no-op decorator if Numba is not available
    def jit(*_args, **_kwargs):
        def decorator(func):
            return func
        return decorator
    NUMBA_AVAILABLE = False

# Additional imports for boundary masking
try:
    from scipy.spatial import ConvexHull
    from scipy.spatial.distance import cdist
    from matplotlib.path import Path as MplPath
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Import layer system components for layer generation
try:
    # Import directly from the module to avoid ui package cascade
    import sys
    import os
    import importlib.util
    
    # Get the absolute path to layer_types.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    layer_types_path = os.path.join(current_dir, '..', 'ui', 'map', 'layer_types.py')
    layer_types_path = os.path.normpath(layer_types_path)
    
    # Import layer_types module directly
    spec = importlib.util.spec_from_file_location("layer_types", layer_types_path)
    layer_types_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(layer_types_module)
    
    # Extract the classes we need
    UXOLayer = layer_types_module.UXOLayer
    LayerType = layer_types_module.LayerType
    GeometryType = layer_types_module.GeometryType
    LayerStyle = layer_types_module.LayerStyle
    LayerSource = layer_types_module.LayerSource
    NORWEGIAN_CRS = layer_types_module.NORWEGIAN_CRS
    
    LAYER_SYSTEM_AVAILABLE = True
    logger.info("✓ Layer system successfully loaded via direct import")
    logger.info(f"✓ Available layer types: {list(LayerType)}")
    logger.info(f"✓ Available geometry types: {list(GeometryType)}")
    logger.info(f"✓ UXOLayer class: {UXOLayer}")
    logger.info("✓ Layer creation should work properly")
except Exception as e:
    logger.error(f"✗ Layer system not available - {str(e)}")
    logger.error("✗ This will cause layer creation to be SKIPPED")
    logger.error("✗ Scripts will create layer_outputs but they won't appear on the map")
    import traceback
    logger.error(f"✗ Full traceback: {traceback.format_exc()}")
    LAYER_SYSTEM_AVAILABLE = False
    # Define dummy classes for type hints when layer system not available
    GeometryType = str
    LayerType = str
    UXOLayer = object
    LayerStyle = object
    LayerSource = object
    NORWEGIAN_CRS = "EPSG:25833"


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
    layer_outputs: List[LayerOutput] = field(default_factory=list)
    figure: Optional[Figure] = None  # Matplotlib figure for direct display
    
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
    
    def add_figure_file(self, file_path: str, file_type: str, description: str):
        """Add a generated figure file to the metadata for tracking"""
        if 'figures' not in self.metadata:
            self.metadata['figures'] = []
        
        figure_info = {
            'file_path': file_path,
            'file_type': file_type,
            'description': description
        }
        self.metadata['figures'].append(figure_info)


class ScriptInterface(ABC):
    """Abstract interface that all processing scripts must implement"""
    
    def __init__(self, project_manager=None):
        """Initialize script with optional project manager reference"""
        self.project_manager = project_manager
    
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
    
    def get_project_working_directory(self) -> Optional[str]:
        """Get the current project working directory"""
        if self.project_manager:
            working_dir = self.project_manager.get_current_working_directory()
            print(f"DEBUG: ScriptInterface.get_project_working_directory() returning: {working_dir}")
            return working_dir
        print(f"DEBUG: ScriptInterface.get_project_working_directory() - no project_manager")
        return None


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
        logger.info(f"DEBUG: ProcessingWorker.run() started")
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
            
            # DEBUG: Check processing result
            logger.info(f"DEBUG ProcessingWorker: result.success={result.success}")
            logger.info(f"DEBUG ProcessingWorker: result.layer_outputs count={len(result.layer_outputs) if result.layer_outputs else 'None'}")
            logger.info(f"DEBUG ProcessingWorker: processor_instance={self.processor_instance is not None}")
            
            # Convert LayerOutput objects to UXOLayer objects and emit them
            if result.success and result.layer_outputs:
                logger.info(f"Processing result has {len(result.layer_outputs)} layer outputs, LAYER_SYSTEM_AVAILABLE={LAYER_SYSTEM_AVAILABLE}")
                if not LAYER_SYSTEM_AVAILABLE:
                    logger.error("LAYER SYSTEM NOT AVAILABLE - This is why layers are not appearing on the map!")
                    logger.error("Layer creation skipped because layer system imports failed")
                    return
                    
                if self.processor_instance:
                    for layer_output in result.layer_outputs:
                        try:
                            logger.info(f"Converting layer output: {layer_output.layer_type}")
                            uxo_layer = self._convert_layer_output_to_uxo_layer(layer_output, result)
                            if uxo_layer:
                                self.layer_created.emit(uxo_layer)
                                logger.info(f"✓ Created and emitted layer: {uxo_layer.name}")
                            else:
                                logger.error(f"✗ _convert_layer_output_to_uxo_layer returned None for layer_type: {layer_output.layer_type}")
                        except Exception as e:
                            logger.error(f"✗ Failed to create layer for {layer_output.layer_type}: {str(e)}")
                            import traceback
                            logger.error(traceback.format_exc())
                else:
                    logger.warning("No processor instance available for layer creation")
            elif result.success and not result.layer_outputs:
                logger.info("Processing successful but no layer outputs generated")
            elif not result.success:
                logger.info("Processing failed, no layer creation attempted")
            
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
            # Update processor's output files for traceability
            if result.output_files:
                output_file_paths = [of.file_path for of in result.output_files]
                current_input_files = getattr(self.processor_instance, '_current_input_files', [])
                self.processor_instance.set_current_files(current_input_files, output_file_paths)
            
            # Add figures information from result metadata if available
            additional_metadata = {}
            if result.metadata and 'figures' in result.metadata:
                additional_metadata['figures'] = result.metadata['figures']
                additional_metadata['figure_count'] = len(result.metadata['figures'])
            elif result.figure is not None:
                # If there's a figure but no figures metadata yet, create placeholder info
                # The pipeline will generate the actual files after this layer is created
                additional_metadata['figures'] = [
                    {'description': 'Interactive matplotlib plot (.mplplot)', 'file_path': 'Generated by pipeline after layer creation'}
                ]
                additional_metadata['figure_count'] = 1
            
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
            
            # Add figures information to enhanced metadata
            enhanced_metadata.update(additional_metadata)
            
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
    
    def __init__(self, project_manager=None):
        self.name = "Base Processor"
        self.description = "Base processor class"
        self.supported_columns = []
        self.required_columns = []
        self.project_manager = project_manager
        
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
            script_files = sorted(glob.glob(os.path.join(scripts_dir, '*.py')))
            
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
                            try:
                                # Try to create script instance with project_manager parameter
                                script_instance = module.SCRIPT_CLASS(project_manager=self.project_manager)
                            except TypeError:
                                # Fallback for older scripts that don't accept project_manager
                                script_instance = module.SCRIPT_CLASS()
                                # Manually set project_manager if the instance has this attribute
                                if hasattr(script_instance, 'project_manager'):
                                    script_instance.project_manager = self.project_manager
                            
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
            
            # Set input file for traceability before processing
            if input_file_path:
                self.set_current_files([input_file_path])
            
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
        
        # Create UXOLayer with traceability
        return UXOLayer(
            name=name,
            layer_type=uxo_layer_type,
            data=data,
            geometry_type=geometry_type,
            style=style,
            metadata=layer_metadata,
            source=LayerSource.PROCESSING,
            bounds=bounds,
            processing_history=[self.processor_type],
            # Add traceability fields
            processing_run_id=self._generate_run_id(),
            source_script=self._get_current_script_path(),
            source_input_files=getattr(self, '_current_input_files', []),
            generated_output_files=getattr(self, '_current_output_files', [])
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
    
    def _generate_run_id(self) -> str:
        """Generate a unique run ID for this processing session"""
        if not hasattr(self, '_current_run_id'):
            from datetime import datetime
            self._current_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self._current_run_id
    
    def _get_current_script_path(self) -> Optional[str]:
        """Get the path of the current script being executed"""
        if hasattr(self, 'current_script') and self.current_script:
            script_instance = self.available_scripts.get(self.current_script)
            if script_instance:
                # Try to get the script file path
                import inspect
                try:
                    return inspect.getfile(script_instance.__class__)
                except:
                    pass
        return None
    
    def set_current_files(self, input_files: List[str], output_files: List[str] = None):
        """Set the current input and output files for traceability"""
        self._current_input_files = input_files if input_files else []
        self._current_output_files = output_files if output_files else []
    
    def get_project_working_directory(self) -> Optional[str]:
        """Get the current project working directory"""
        if self.project_manager:
            working_dir = self.project_manager.get_current_working_directory()
            return working_dir
        return None
    
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
    
    def _create_gamma_style(self, data: Any, layer_type: str, provided_style: Optional[Dict[str, Any]] = None) -> LayerStyle:
        """Create gamma radiation-specific styling, respecting custom ramps"""
        default_ramp = ["#004400", "#008800", "#00CC00", "#CCCC00", "#CC8800", "#CC0000"]
        color_ramp = provided_style.get('color_ramp', default_ramp) if provided_style else default_ramp
        return LayerStyle(
            point_color="#00CC66",
            point_size=5,
            use_graduated_colors=True,
            color_ramp=color_ramp
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
    
    # ===== MINIMUM CURVATURE INTERPOLATION METHODS =====
    # Universal interpolation methods inherited by all processors
    
    def minimum_curvature_interpolation(self, x, y, z, grid_resolution=150, 
                                        max_iterations=1000, tolerance=1e-6, 
                                        omega=1.8, constraint_mode='soft',
                                        influence_radius=None, progress_callback=None):
        """
        Universal minimum curvature interpolation method for all processors
        
        Args:
            x, y, z: Data coordinates and values
            grid_resolution: Number of grid points per axis
            max_iterations: Maximum iterations for convergence
            tolerance: Convergence tolerance
            omega: Relaxation factor for successive over-relaxation
            constraint_mode: 'soft' (gamma style) or 'hard' (traditional)
            influence_radius: For soft constraints, auto-calculated if None
            progress_callback: Progress reporting function
        
        Returns:
            grid_x, grid_y, grid_z: Interpolated grid coordinates and values
        """
        if progress_callback:
            progress_callback(5, "Setting up interpolation grid...")
        
        # Create regular grid
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        
        # Add minimal padding to grid bounds
        x_range = x_max - x_min
        y_range = y_max - y_min
        padding = 0.01  # 1% padding
        
        x_min -= padding * x_range
        x_max += padding * x_range
        y_min -= padding * y_range
        y_max += padding * y_range
        
        grid_x = np.linspace(x_min, x_max, grid_resolution)
        grid_y = np.linspace(y_min, y_max, grid_resolution)
        grid_X, grid_Y = np.meshgrid(grid_x, grid_y)
        
        # Initialize grid with linear interpolation first
        if progress_callback:
            progress_callback(10, "Creating initial grid with linear interpolation...")
        
        # Create initial grid using linear interpolation to fill entire domain
        grid_z_initial = griddata((x, y), z, (grid_X, grid_Y), method='linear', fill_value=np.nan)
        
        if progress_callback:
            progress_callback(15, "Filling gaps with nearest neighbor interpolation...")
        
        # Fill any remaining NaN values with nearest neighbor interpolation
        nan_mask = np.isnan(grid_z_initial)
        if np.any(nan_mask):
            grid_z_nearest = griddata((x, y), z, (grid_X, grid_Y), method='nearest', fill_value=np.mean(z))
            grid_z_initial[nan_mask] = grid_z_nearest[nan_mask]
        
        grid_z = grid_z_initial.copy()
        
        # Apply constraint mode
        if constraint_mode == 'soft':
            return self._minimum_curvature_soft_constraints(
                x, y, z, grid_X, grid_Y, grid_z, grid_x, grid_y,
                max_iterations, tolerance, omega, influence_radius, progress_callback
            )
        else:
            return self._minimum_curvature_hard_constraints(
                x, y, z, grid_X, grid_Y, grid_z,
                max_iterations, tolerance, omega, progress_callback
            )
    
    def _minimum_curvature_soft_constraints(self, x, y, z, grid_X, grid_Y, grid_z, 
                                           grid_x, grid_y, max_iterations, tolerance, 
                                           omega, influence_radius, progress_callback):
        """Minimum curvature with soft constraints (gamma interpolator style)"""
        if progress_callback:
            progress_callback(20, "Creating soft data constraints...")
        
        # Use soft constraints instead of hard constraints
        ny, nx = grid_z.shape
        x_min, x_max = np.min(grid_x), np.max(grid_x)
        y_min, y_max = np.min(grid_y), np.max(grid_y)
        dx = (x_max - x_min) / (nx - 1)
        dy = (y_max - y_min) / (ny - 1)
        
        # Calculate influence radius if not provided
        if influence_radius is None:
            x_range = x_max - x_min
            y_range = y_max - y_min
            data_density = len(x) / (x_range * y_range)
            influence_radius = max(2 * dx, 2 * dy, 1.0 / np.sqrt(data_density))
            
            # Limit influence radius for performance with large grids
            max_reasonable_radius = min(x_range, y_range) * 0.1  # 10% of data range
            influence_radius = min(influence_radius, max_reasonable_radius)
        
        if progress_callback:
            progress_callback(22, f"Influence radius: {influence_radius:.4f} for {len(x)} data points on {ny}x{nx} grid")
        
        # Convert to numpy arrays for JIT compilation
        x_array = np.asarray(x, dtype=np.float64)
        y_array = np.asarray(y, dtype=np.float64)
        z_array = np.asarray(z, dtype=np.float64)
        
        # Create influence weights and target values
        if progress_callback:
            if NUMBA_AVAILABLE:
                progress_callback(25, "🚀 Using JIT-compiled influence zones...")
            else:
                progress_callback(25, "Creating influence zones...")
        
        influence_weights, target_values = self._create_influence_weights_numba(
            x_array, y_array, z_array, grid_x, grid_y, influence_radius
        )
        
        # Normalize influence weights to [0, 1] range for stability
        max_weight = np.max(influence_weights)
        if max_weight > 0:
            influence_weights = influence_weights / max_weight
        
        if progress_callback:
            progress_callback(35, f"Influence zones created successfully (max weight: {max_weight:.3f})")
        
        # Value bounds for stability
        data_range = np.ptp(z)
        data_mean = np.mean(z)
        min_allowed = data_mean - 3 * data_range
        max_allowed = data_mean + 3 * data_range
        
        if progress_callback:
            progress_callback(40, "Starting minimum curvature iterations with soft constraints...")
        
        # Use stable omega value for soft constraints
        omega = 0.5  # Lower omega for stability
        
        if progress_callback:
            if NUMBA_AVAILABLE:
                progress_callback(45, f"🚀 Starting {max_iterations} JIT-compiled iterations...")
            else:
                progress_callback(45, f"Starting {max_iterations} iterations...")
        
        # Iterative minimum curvature with soft constraints
        for iteration in range(max_iterations):
            # Use JIT-compiled function for the iteration
            max_change = self._minimum_curvature_iteration_numba(
                grid_z, influence_weights, target_values, omega, min_allowed, max_allowed
            )
            
            # Apply boundary conditions
            self._apply_boundary_conditions_numba(grid_z)
            
            # Early detection of instability
            if max_change > data_range:
                if progress_callback:
                    progress_callback(80, f"Large change detected at iteration {iteration+1}, stopping early")
                break
            
            # Progress update
            if progress_callback:
                progress = 45 + 35 * ((iteration + 1) / max_iterations)
                convergence_info = f"max change: {max_change:.2e}"
                if max_change < tolerance * 10:
                    convergence_info += " (converging)"
                progress_callback(int(progress), f"Iteration {iteration+1}/{max_iterations}, {convergence_info}")
            
            # Check for convergence
            if max_change < tolerance:
                if progress_callback:
                    progress_callback(80, f"Converged after {iteration+1} iterations (change: {max_change:.2e})")
                break
        
        if progress_callback:
            if NUMBA_AVAILABLE:
                progress_callback(85, "🚀 JIT-compiled minimum curvature interpolation complete")
            else:
                progress_callback(85, "Minimum curvature interpolation complete")
        
        return grid_X, grid_Y, grid_z
    
    def _minimum_curvature_hard_constraints(self, x, y, z, grid_X, grid_Y, grid_z,
                                           max_iterations, tolerance, omega, progress_callback):
        """Minimum curvature with hard constraints (traditional style)"""
        if progress_callback:
            progress_callback(20, "Setting up hard constraints...")
        
        # Create mask for data points
        ny, nx = grid_z.shape
        constraint_mask = np.zeros((ny, nx), dtype=bool)
        
        # Find nearest grid points for each data point
        x_min, x_max = np.min(grid_X), np.max(grid_X)
        y_min, y_max = np.min(grid_Y), np.max(grid_Y)
        
        for i in range(len(x)):
            # Convert data point to grid indices
            j = int(round((x[i] - x_min) / (x_max - x_min) * (nx - 1)))
            i_idx = int(round((y[i] - y_min) / (y_max - y_min) * (ny - 1)))
            
            # Clamp to valid range
            j = max(0, min(nx - 1, j))
            i_idx = max(0, min(ny - 1, i_idx))
            
            # Set constraint
            constraint_mask[i_idx, j] = True
            grid_z[i_idx, j] = z[i]
        
        if progress_callback:
            progress_callback(40, "Starting minimum curvature iterations with hard constraints...")
        
        # Iterative minimum curvature with hard constraints
        for iteration in range(max_iterations):
            max_change = 0.0
            
            # Update interior points using 5-point stencil
            for i in range(1, ny-1):
                for j in range(1, nx-1):
                    if not constraint_mask[i, j]:  # Skip constrained points
                        old_value = grid_z[i, j]
                        
                        # 5-point stencil for Laplacian
                        neighbors = (grid_z[i+1, j] + grid_z[i-1, j] + 
                                   grid_z[i, j+1] + grid_z[i, j-1]) / 4.0
                        
                        # Successive over-relaxation
                        new_value = old_value + omega * (neighbors - old_value)
                        grid_z[i, j] = new_value
                        
                        # Track maximum change
                        change = abs(new_value - old_value)
                        if change > max_change:
                            max_change = change
            
            # Apply boundary conditions
            self._apply_boundary_conditions_numba(grid_z)
            
            # Progress update
            if progress_callback:
                progress = 40 + 40 * ((iteration + 1) / max_iterations)
                progress_callback(int(progress), f"Iteration {iteration+1}/{max_iterations}, max change: {max_change:.2e}")
            
            # Check for convergence
            if max_change < tolerance:
                if progress_callback:
                    progress_callback(80, f"Converged after {iteration+1} iterations")
                break
        
        if progress_callback:
            progress_callback(85, "Minimum curvature interpolation complete")
        
        return grid_X, grid_Y, grid_z
    
    def create_boundary_mask(self, x, y, grid_x, grid_y, method='convex_hull'):
        """
        Create a boundary mask to prevent interpolation outside data coverage.
        
        Args:
            x, y: Data point coordinates
            grid_x, grid_y: Grid meshgrid coordinates
            method: 'convex_hull' or 'alpha_shape'
        
        Returns:
            mask: Boolean array where True indicates points inside the data boundary
        """
        if not SCIPY_AVAILABLE:
            # Return all True mask if scipy is not available
            return np.ones(grid_x.shape, dtype=bool)
        
        try:
            # Get data points as array
            data_points = np.column_stack([x, y])
            
            # Get grid points as array
            grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
            
            if method == 'convex_hull':
                # Create convex hull
                hull = ConvexHull(data_points)
                
                # Check which grid points are inside the convex hull
                hull_path = MplPath(data_points[hull.vertices])
                mask_1d = hull_path.contains_points(grid_points)
                
            elif method == 'alpha_shape':
                # Simple distance-based approach as alpha shape approximation
                # For each grid point, check if it's within reasonable distance of data
                distances = cdist(grid_points, data_points)
                min_distances = np.min(distances, axis=1)
                
                # Use median nearest neighbor distance as threshold
                nn_distances = []
                for i in range(len(data_points)):
                    dists = cdist([data_points[i]], data_points)[0]
                    dists = dists[dists > 0]  # Remove self-distance
                    if len(dists) > 0:
                        nn_distances.append(np.min(dists))
                
                threshold = np.median(nn_distances) * 2.0 if nn_distances else 0.01
                mask_1d = min_distances <= threshold
            
            else:
                raise ValueError(f"Unknown boundary method: {method}")
            
            # Reshape to grid shape
            mask = mask_1d.reshape(grid_x.shape)
            
            return mask
            
        except Exception:
            # Fallback to no masking
            return np.ones(grid_x.shape, dtype=bool)
    
    @jit(nopython=True, cache=True)
    def _create_influence_weights_numba(self, x, y, z, grid_x_coords, grid_y_coords, influence_radius):
        """
        JIT-compiled function to create influence weights and target values for soft constraints.
        
        Args:
            x, y, z: Data point coordinates and values (1D arrays)
            grid_x_coords, grid_y_coords: Grid coordinate arrays (1D arrays)
            influence_radius: Influence radius for soft constraints
        
        Returns:
            influence_weights: 2D array of influence weights
            target_values: 2D array of target values
        """
        ny = len(grid_y_coords)
        nx = len(grid_x_coords)
        
        influence_weights = np.zeros((ny, nx))
        target_values = np.zeros((ny, nx))
        
        # Grid spacing
        dx = grid_x_coords[1] - grid_x_coords[0] if nx > 1 else 1.0
        dy = grid_y_coords[1] - grid_y_coords[0] if ny > 1 else 1.0
        
        # For each data point, create influence zone
        for data_idx in range(len(x)):
            xi, yi, zi = x[data_idx], y[data_idx], z[data_idx]
            
            # Convert to grid coordinates
            gi = (yi - grid_y_coords[0]) / dy
            gj = (xi - grid_x_coords[0]) / dx
            
            # Calculate grid range that could be influenced
            i_min = max(0, int(gi - influence_radius/dy))
            i_max = min(ny, int(gi + influence_radius/dy) + 1)
            j_min = max(0, int(gj - influence_radius/dx))
            j_max = min(nx, int(gj + influence_radius/dx) + 1)
            
            # Create influence zone around this data point
            for i in range(i_min, i_max):
                for j in range(j_min, j_max):
                    # Calculate distance from grid point to data point
                    grid_y_coord = grid_y_coords[i]
                    grid_x_coord = grid_x_coords[j]
                    distance = np.sqrt((grid_x_coord - xi)**2 + (grid_y_coord - yi)**2)
                    
                    if distance <= influence_radius:
                        # Gaussian-like weight function
                        weight = np.exp(-(distance / (influence_radius * 0.3))**2)
                        
                        # Accumulate weighted influence
                        old_weight = influence_weights[i, j]
                        new_weight = old_weight + weight
                        
                        if new_weight > 0:
                            # Weighted average of target values
                            target_values[i, j] = (target_values[i, j] * old_weight + zi * weight) / new_weight
                            influence_weights[i, j] = new_weight
        
        return influence_weights, target_values
    
    @jit(nopython=True, cache=True)
    def _minimum_curvature_iteration_numba(self, grid_z, influence_weights, target_values, 
                                          omega, min_allowed, max_allowed):
        """
        JIT-compiled function for a single minimum curvature iteration with soft constraints.
        
        Args:
            grid_z: Current grid values (modified in-place)
            influence_weights: Influence weights for soft constraints
            target_values: Target values for soft constraints
            omega: Relaxation factor
            min_allowed, max_allowed: Value bounds for stability
        
        Returns:
            max_change: Maximum change in this iteration
        """
        ny, nx = grid_z.shape
        max_change = 0.0
        
        # Update interior points using 5-point stencil with soft constraints
        for i in range(1, ny-1):
            for j in range(1, nx-1):
                old_value = grid_z[i, j]
                
                # 5-point stencil for Laplacian
                neighbors = (grid_z[i+1, j] + grid_z[i-1, j] + 
                           grid_z[i, j+1] + grid_z[i, j-1]) / 4.0
                
                # Successive over-relaxation
                new_value = old_value + omega * (neighbors - old_value)
                
                # Apply soft data constraint if there's influence at this point
                if influence_weights[i, j] > 0.01:  # Only apply if significant influence
                    constraint_strength = influence_weights[i, j] * 0.1  # Reduce constraint strength
                    target = target_values[i, j]
                    # Blend the relaxed value with the target value
                    new_value = new_value * (1 - constraint_strength) + target * constraint_strength
                
                # Clamp values for stability
                new_value = max(min_allowed, min(max_allowed, new_value))
                
                grid_z[i, j] = new_value
                
                # Track maximum change
                change = abs(new_value - old_value)
                if change > max_change:
                    max_change = change
        
        return max_change
    
    @jit(nopython=True, cache=True)
    def _apply_boundary_conditions_numba(self, grid_z):
        """
        JIT-compiled function to apply boundary conditions (zero second derivative).
        
        Args:
            grid_z: Grid values (modified in-place)
        """
        ny, nx = grid_z.shape
        
        # Apply boundary conditions
        if ny >= 3:
            # Top and bottom boundaries (zero second derivative)
            for j in range(nx):
                grid_z[0, j] = 2*grid_z[1, j] - grid_z[2, j]
                grid_z[ny-1, j] = 2*grid_z[ny-2, j] - grid_z[ny-3, j]
        
        if nx >= 3:
            # Left and right boundaries (zero second derivative)
            for i in range(ny):
                grid_z[i, 0] = 2*grid_z[i, 1] - grid_z[i, 2]
                grid_z[i, nx-1] = 2*grid_z[i, nx-2] - grid_z[i, nx-3] 