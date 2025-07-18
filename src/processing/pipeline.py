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
    layer_created = Signal(object)  # UXOLayer created during processing
    
    def __init__(self, project_manager=None):
        super().__init__()
        self.processors: Dict[str, BaseProcessor] = {}
        self.current_input_file: Optional[str] = None  # Track current input file
        self.project_manager = project_manager  # Reference to project manager for working directory
        
        # Try to create each processor and catch any import errors
        try:
            logger.debug("Creating MagneticProcessor...")
            print(f"DEBUG: Creating MagneticProcessor with project_manager: {project_manager}")
            self.processors['magnetic'] = MagneticProcessor(project_manager=project_manager)
            logger.debug("MagneticProcessor created successfully")
        except Exception as e:
            logger.error(f"Failed to create MagneticProcessor: {e}")
            
        try:
            logger.debug("Creating GPRProcessor...")
            self.processors['gpr'] = GPRProcessor(project_manager=project_manager)
            logger.debug("GPRProcessor created successfully")
        except Exception as e:
            logger.error(f"Failed to create GPRProcessor: {e}")
            
        try:
            logger.debug("Creating GammaProcessor...")
            self.processors['gamma'] = GammaProcessor(project_manager=project_manager)
            logger.debug("GammaProcessor created successfully")
        except Exception as e:
            logger.error(f"Failed to create GammaProcessor: {e}")
            
        try:
            logger.debug("Creating MultispectralProcessor...")
            self.processors['multispectral'] = MultispectralProcessor(project_manager=project_manager)
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
            params,
            input_file_path=self.current_input_file,
            processor_instance=processor  # Pass processor instance for layer creation
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
        
        # Connect layer creation signal to forward to main application
        self.current_worker.layer_created.connect(self.layer_created.emit)
        
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
        logger.info(f"Pipeline processing finished - success: {result.success}, has_data: {result.data is not None}, has_figure: {result.figure is not None}")
        if result.success:
            try:
                # Set input file path in result
                result.input_file_path = self.current_input_file
                
                # Generate output file automatically (only if there's data)
                if result.data is not None:
                    output_path = self._generate_output_file(result)
                    result.output_file_path = output_path
                    logger.info(f"Processed data saved to: {output_path}")
                    
                    # Generate metadata sidecar file
                    self._generate_metadata_file(result)
                
                # Auto-save any generated plots (independent of data)
                self._save_plot_files(result)
                
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
        """Create and return output directory path - always use project working directory"""
        # Use project working directory if available
        if self.project_manager and self.project_manager.get_current_working_directory():
            project_dir = self.project_manager.get_current_working_directory()
            base_output_dir = os.path.join(project_dir, "processed")
        elif result.input_file_path:
            # Find project root by looking for the first "processed" directory in the path
            input_path = Path(result.input_file_path)
            project_dir = None
            
            # Walk up the path to find project root (before any "processed" directory)
            for parent in input_path.parents:
                if parent.name == "processed" and parent.parent:
                    project_dir = parent.parent
                    break
            
            # If no processed directory found, use directory containing input file
            if project_dir is None:
                project_dir = input_path.parent
                
            base_output_dir = os.path.join(project_dir, "processed")
        else:
            # Last resort: current working directory
            base_output_dir = os.path.join(os.getcwd(), "processed")
        
        # Create processor-specific subdirectory
        processor_type = result.metadata.get('processor', 'unknown')
        output_dir = os.path.join(base_output_dir, processor_type)
        
        # Create directories if they don't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Processing output directory: {output_dir}")
        return output_dir
    
    def _generate_filename(self, result: ProcessingResult) -> str:
        """Generate filename based on input and processing info"""
        # Check if the script has already provided a specific filename
        if hasattr(result, 'output_files') and result.output_files:
            for output_file in result.output_files:
                if output_file.file_type.lower() in ['csv', 'xlsx', 'json']:
                    # Use the script-generated filename
                    return os.path.basename(output_file.file_path)
        
        # Fallback to default filename generation
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
        if script_name == 'magbase_processing':
            filename = f"{base_name}_magbase.{ext}"
        else:
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
    
    def _save_plot_files(self, result: ProcessingResult):
        """Automatically save any generated matplotlib figures"""
        logger.info("_save_plot_files called")
        if not result.figure:
            logger.info("No figure found in result - skipping plot save")
            return
        
        logger.info(f"Found figure in result - proceeding with auto-save")
            
        try:
            # Create output directory structure (same as data files)
            output_dir = self._create_output_directory(result)
            
            # Generate unique plot filename based on processing info
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            script_name = result.processing_script or "plot"
            processor_type = result.metadata.get('processor', 'unknown')
            
            # Get base name from input file if available
            if result.input_file_path:
                input_name = Path(result.input_file_path).stem
                base_name = f"{input_name}_{script_name}_{timestamp}"
            else:
                base_name = f"{processor_type}_{script_name}_{timestamp}"
            
            logger.info(f"Auto-saving plot to directory: {output_dir}")
            
            # Priority: Save interactive .mplplot file with unique naming
            plot_files = []
            
            # Save interactive plot file (.mplplot) - primary format
            mplplot_path = Path(output_dir) / f"{base_name}.mplplot"
            try:
                import pickle
                with open(mplplot_path, 'wb') as f:
                    pickle.dump(result.figure, f)
                plot_files.append(('mplplot', str(mplplot_path), 'Interactive matplotlib plot'))
                logger.info(f"✓ Auto-saved interactive plot: {mplplot_path}")
            except Exception as e:
                logger.error(f"✗ Failed to save interactive plot: {e}")
            
            # Optionally save static PNG file (secondary format)
            """
            png_path = Path(output_dir) / f"{base_name}.png"
            try:
                result.figure.savefig(png_path, dpi=300, bbox_inches='tight')
                plot_files.append(('png', str(png_path), 'Static plot image'))
                logger.info(f"✓ Auto-saved static plot: {png_path}")
            except Exception as e:
                logger.warning(f"✗ Failed to save static plot: {e}")
            """
            # Add plot files to result outputs
            for file_type, file_path, description in plot_files:
                result.add_output_file(file_path, file_type, description)
            
            logger.info(f"Successfully auto-saved {len(plot_files)} plot files")
                
        except Exception as e:
            logger.error(f"Failed to save plot files: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")