"""
Processing pipeline manager for coordinating data processing
"""

from typing import Dict, List, Type, Optional
from PySide6.QtCore import QObject, Signal
import pandas as pd
from loguru import logger

from .base import BaseProcessor, ProcessingResult, ProcessingWorker
from .magnetic import MagneticProcessor
from .gpr import GPRProcessor
from .gamma import GammaProcessor
from .multispectral import MultispectralProcessor


class ProcessingPipeline(QObject):
    """Manages and coordinates all data processors"""
    
    # Signals
    processing_started = Signal(str)  # Processor name
    processing_finished = Signal(ProcessingResult)
    progress_updated = Signal(int, str)  # Progress percentage, message
    error_occurred = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.processors: Dict[str, BaseProcessor] = {}
        
        # Try to create each processor and catch any import errors
        try:
            logger.debug("Creating MagneticProcessor...")
            self.processors['magnetic'] = MagneticProcessor()
            logger.debug("MagneticProcessor created successfully")
        except Exception as e:
            logger.error(f"Failed to create MagneticProcessor: {e}")
            
        try:
            logger.debug("Creating GPRProcessor...")
            self.processors['gpr'] = GPRProcessor()
            logger.debug("GPRProcessor created successfully")
        except Exception as e:
            logger.error(f"Failed to create GPRProcessor: {e}")
            
        try:
            logger.debug("Creating GammaProcessor...")
            self.processors['gamma'] = GammaProcessor()
            logger.debug("GammaProcessor created successfully")
        except Exception as e:
            logger.error(f"Failed to create GammaProcessor: {e}")
            
        try:
            logger.debug("Creating MultispectralProcessor...")
            self.processors['multispectral'] = MultispectralProcessor()
            logger.debug("MultispectralProcessor created successfully")
        except Exception as e:
            logger.error(f"Failed to create MultispectralProcessor: {e}")
            
        logger.info(f"ProcessingPipeline initialized with {len(self.processors)} processors")
        self.current_worker: Optional[ProcessingWorker] = None
        
    def get_available_processors(self) -> List[Dict[str, str]]:
        """Get list of available processors with metadata"""
        return [
            {
                'id': proc_id,
                'name': processor.name,
                'description': processor.description,
                'icon': self._get_processor_icon(proc_id)
            }
            for proc_id, processor in self.processors.items()
        ]
    
    def get_processor(self, processor_id: str) -> Optional[BaseProcessor]:
        """Get a specific processor by ID"""
        return self.processors.get(processor_id)
    
    def detect_data_type(self, data: pd.DataFrame) -> str:
        """Auto-detect the most likely data type based on columns"""
        logger.debug(f"Auto-detecting data type for DataFrame with columns: {list(data.columns)}")
        scores = {}
        
        for proc_id, processor in self.processors.items():
            try:
                # Try to validate data with each processor
                processor.validate_data(data)
                scores[proc_id] = 1.0
                logger.debug(f"Processor {proc_id} validates data successfully (score: 1.0)")
            except Exception as e:
                # Check column detection
                detected = processor.detect_columns(data)
                score = len(detected) / 10.0  # Normalize score
                scores[proc_id] = score
                logger.debug(f"Processor {proc_id} detected {len(detected)} columns (score: {score:.2f}): {detected}")
        
        # Return processor with highest score
        if scores:
            best_processor = max(scores.items(), key=lambda x: x[1])
            logger.info(f"Auto-detected data type: {best_processor[0]} (score: {best_processor[1]:.2f})")
            return best_processor[0]
        
        logger.warning("No suitable processor found, defaulting to magnetic")
        return 'magnetic'  # Default fallback
    
    def process_data(self, processor_id: str, data: pd.DataFrame, 
                     params: Optional[Dict] = None) -> None:
        """Process data using specified processor in background thread"""
        logger.info(f"Starting data processing with processor: {processor_id}")
        logger.debug(f"Data shape: {data.shape}, Parameters provided: {params is not None}")
        
        processor = self.processors.get(processor_id)
        if not processor:
            error_msg = f"Unknown processor: {processor_id}"
            logger.error(error_msg)
            self.error_occurred.emit(error_msg)
            return
            
        # Use default parameters if none provided
        if params is None:
            params = processor.parameters
            logger.debug("Using default processor parameters")
        else:
            logger.debug("Using custom parameters")
            
        # Stop any existing processing
        if self.current_worker and self.current_worker.isRunning():
            logger.debug("Cancelling existing processing worker")
            self.current_worker.cancel()
            self.current_worker.wait()
            
        # Create and configure worker
        logger.debug("Creating new processing worker thread")
        self.current_worker = ProcessingWorker(
            processor.process,
            data,
            params
        )
        
        # Connect signals
        self.current_worker.progress.connect(
            lambda val: self.progress_updated.emit(val, "")
        )
        self.current_worker.status.connect(
            lambda msg: self.progress_updated.emit(-1, msg)
        )
        self.current_worker.finished.connect(self._on_processing_finished)
        self.current_worker.error.connect(self.error_occurred.emit)
        
        # Start processing
        logger.info(f"Starting {processor.name} processing in background thread")
        self.processing_started.emit(processor.name)
        self.current_worker.start()
        
    def cancel_processing(self) -> None:
        """Cancel current processing operation"""
        if self.current_worker and self.current_worker.isRunning():
            self.current_worker.cancel()
            logger.info("Processing cancelled by user")
            
    def _on_processing_finished(self, result: ProcessingResult):
        """Handle processing completion"""
        self.processing_finished.emit(result)
        if result.success:
            logger.info(f"Processing completed successfully in {result.processing_time:.2f}s")
        else:
            logger.error(f"Processing failed: {result.error_message}")
            
    def _get_processor_icon(self, processor_id: str) -> str:
        """Get icon for processor type"""
        icons = {
            'magnetic': '🧲',
            'gpr': '📡',
            'gamma': '☢️',
            'multispectral': '🌈'
        }
        return icons.get(processor_id, '📊') 