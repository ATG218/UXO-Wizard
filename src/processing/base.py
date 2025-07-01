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
    
    def __init__(self, processor_func: Callable, data: pd.DataFrame, params: Dict[str, Any], input_file_path: Optional[str] = None):
        super().__init__()
        self.processor_func = processor_func
        self.data = data
        self.params = params
        self.input_file_path = input_file_path
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
            self.finished.emit(result)
            
        except Exception as e:
            logger.error(f"Processing error: {str(e)}")
            self.error.emit(str(e))
            self.finished.emit(ProcessingResult(
                success=False,
                error_message=str(e)
            ))
    
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
            'latitude': ['lat', 'latitude', 'y', 'northing'],
            'longitude': ['lon', 'lng', 'longitude', 'x', 'easting'],
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
    
    def create_animation_frames(self, data: pd.DataFrame, 
                              num_frames: int = 10) -> List[pd.DataFrame]:
        """Create animation frames for progressive visualization"""
        frames = []
        step = max(1, len(data) // num_frames)
        
        for i in range(0, len(data), step):
            frames.append(data.iloc[:i+step])
            
        return frames 