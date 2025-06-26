"""
Processing pipeline manager for coordinating data processing
"""

from typing import Dict, List, Type, Optional
from PySide6.QtCore import QObject, Signal
import pandas as pd
import os
import json
from datetime import datetime
from pathlib import Path
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
        self.current_input_file: Optional[str] = None  # Track current input file
        
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
                     params: Optional[Dict] = None, input_file_path: Optional[str] = None) -> None:
        """Process data using specified processor in background thread"""
        logger.info(f"Starting data processing with processor: {processor_id}")
        logger.debug(f"Data shape: {data.shape}, Parameters provided: {params is not None}")
        
        # Store input file path for output generation
        self.current_input_file = input_file_path
        
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
        """Handle processing completion with auto file generation"""
        if result.success and result.data is not None:
            try:
                # Set input file path in result
                result.input_file_path = self.current_input_file
                
                # Generate output file automatically
                output_path = self._generate_output_file(result)
                result.output_file_path = output_path
                logger.info(f"Processed data saved to: {output_path}")
                
                # Generate metadata sidecar file
                self._generate_metadata_file(result)
                
            except Exception as e:
                logger.error(f"Failed to generate output file: {str(e)}")
                result.error_message = f"Processing succeeded but file generation failed: {str(e)}"
                result.success = False
        
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
    
    def _generate_output_file(self, result: ProcessingResult) -> str:
        """Generate output file with processed data"""
        # Create output directory structure
        output_dir = self._create_output_directory(result)
        
        # Generate filename
        filename = self._generate_filename(result)
        output_path = os.path.join(output_dir, filename)
        
        # Save data in specified format
        if result.export_format.lower() == 'csv':
            result.data.to_csv(output_path, index=False)
        elif result.export_format.lower() in ['xlsx', 'excel']:
            result.data.to_excel(output_path, index=False)
        elif result.export_format.lower() == 'json':
            result.data.to_json(output_path, orient='records', indent=2)
        else:
            # Default to CSV
            result.data.to_csv(output_path, index=False)
            
        logger.debug(f"Generated output file: {output_path}")
        return output_path
    
    def _create_output_directory(self, result: ProcessingResult) -> str:
        """Create and return output directory path"""
        if result.input_file_path:
            # Create processed/ directory next to input file
            input_dir = os.path.dirname(result.input_file_path)
            base_output_dir = os.path.join(input_dir, "processed")
        else:
            # Fallback to current working directory
            base_output_dir = os.path.join(os.getcwd(), "processed")
        
        # Create processor-specific subdirectory
        processor_type = result.metadata.get('processor', 'unknown')
        output_dir = os.path.join(base_output_dir, processor_type)
        
        # Create directories if they don't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        return output_dir
    
    def _generate_filename(self, result: ProcessingResult) -> str:
        """Generate filename based on input and processing info"""
        # Get base name from input file
        if result.input_file_path:
            base_name = Path(result.input_file_path).stem
        else:
            base_name = "processed_data"
        
        # Get processor and script info
        processor_type = result.metadata.get('processor', 'unknown')
        script_name = result.processing_script or 'default'
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get file extension
        ext = result.export_format.lower()
        if ext == 'excel':
            ext = 'xlsx'
        
        # Format: {input_name}_{processor}_{script}_{timestamp}.{ext}
        filename = f"{base_name}_{processor_type}_{script_name}_{timestamp}.{ext}"
        
        return filename
    
    def _generate_metadata_file(self, result: ProcessingResult) -> str:
        """Generate metadata sidecar file"""
        if not result.output_file_path:
            return None
            
        # Create metadata filename (same as output but with .json extension)
        output_path = Path(result.output_file_path)
        metadata_path = output_path.with_suffix('.json')
        
        # Prepare metadata
        metadata = {
            'processing_info': {
                'processor_type': result.metadata.get('processor', 'unknown'),
                'processing_script': result.processing_script,
                'processing_time': result.processing_time,
                'timestamp': datetime.now().isoformat(),
                'success': result.success
            },
            'file_info': {
                'input_file': result.input_file_path,
                'output_file': result.output_file_path,
                'export_format': result.export_format,
                'data_shape': list(result.data.shape) if result.data is not None else None
            },
            'parameters': result.metadata.get('parameters', {}),
            'results': {
                key: value for key, value in result.metadata.items() 
                if key not in ['processor', 'parameters']
            }
        }
        
        # Save metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
            
        logger.debug(f"Generated metadata file: {metadata_path}")
        return str(metadata_path) 