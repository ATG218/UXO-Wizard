"""
Base classes for all data processors
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Callable
from PySide6.QtCore import Signal, QThread
import pandas as pd
from loguru import logger
import time


class ProcessingError(Exception):
    """Custom exception for processing errors"""
    pass


@dataclass
class ProcessingResult:
    """Container for processing results"""
    success: bool
    data: Optional[pd.DataFrame] = None
    metadata: Dict[str, Any] = None
    error_message: Optional[str] = None
    processing_time: float = 0.0
    output_file_path: Optional[str] = None        # Generated file path
    processing_script: Optional[str] = None       # Script/algorithm name
    input_file_path: Optional[str] = None         # Original input file
    export_format: str = "csv"                    # Output format
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ProcessingWorker(QThread):
    """Worker thread for background processing"""
    
    # Signals
    progress = Signal(int)  # Progress percentage (0-100)
    status = Signal(str)    # Status message
    finished = Signal(ProcessingResult)  # Final result
    error = Signal(str)     # Error message
    
    def __init__(self, processor_func: Callable, data: pd.DataFrame, params: Dict[str, Any]):
        super().__init__()
        self.processor_func = processor_func
        self.data = data
        self.params = params
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
                progress_callback=progress_callback
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
    """Abstract base class for all data processors"""
    
    def __init__(self):
        self.name = "Base Processor"
        self.description = "Base processor class"
        self.supported_columns = []
        self.required_columns = []
        self.parameters = self._define_parameters()
        
    @abstractmethod
    def _define_parameters(self) -> Dict[str, Any]:
        """Define processing parameters with defaults and metadata"""
        return {}
    
    @abstractmethod
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate that data has required columns and format"""
        # Check required columns
        missing_cols = [col for col in self.required_columns if col not in data.columns]
        if missing_cols:
            raise ProcessingError(f"Missing required columns: {missing_cols}")
        return True
    
    @abstractmethod
    def process(self, data: pd.DataFrame, params: Dict[str, Any], 
                progress_callback: Optional[Callable] = None) -> ProcessingResult:
        """Main processing method - must be implemented by subclasses"""
        pass
    
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