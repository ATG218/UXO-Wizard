"""
Magnetic data processing for UXO detection - Script Integration Framework
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Callable, List
from loguru import logger

from .base import BaseProcessor, ProcessingResult, ProcessingError


class MagneticProcessor(BaseProcessor):
    """Processor for magnetic survey data with script framework support"""
    
    def __init__(self):
        super().__init__()
        self.name = "Magnetic Processor"
        self.description = "Process magnetic field data using various processing scripts"
        self.required_columns = []  # Will be detected automatically
        
    def _define_parameters(self) -> Dict[str, Any]:
        """Define parameters - now handled by script framework"""
        # Let the base class handle script-based parameter generation
        if self.available_scripts:
            # Use first available script as default
            default_script = list(self.available_scripts.keys())[0]
            return self._generate_script_parameters(default_script)
        else:
            # Fallback to base parameters if no scripts available
            return self._define_base_parameters()
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate magnetic data - basic checks for script integration"""
        if data.empty:
            raise ProcessingError("No data provided")
            
        # Basic column detection for magnetic data
        detected = self.detect_columns(data)
        logger.debug(f"Detected columns: {detected}")
        
        # Check for basic data structure
        if len(data.columns) < 2:
            raise ProcessingError("Data must have at least 2 columns")
            
        return True
    
    def _legacy_process(self, data: pd.DataFrame, params: Dict[str, Any], 
                       progress_callback: Optional[Callable] = None) -> ProcessingResult:
        """Legacy processing method for backward compatibility"""
        try:
            if progress_callback:
                progress_callback(0, "Starting legacy magnetic processing...")
            
            # Basic preprocessing
            if progress_callback:
                progress_callback(50, "Preprocessing data...")
            result_data = self.preprocess(data)
            
            # Simple placeholder processing
            if progress_callback:
                progress_callback(80, "Executing basic processing...")
            
            # Add a simple processed column as example
            if len(data.columns) > 0:
                first_numeric_col = None
                for col in data.columns:
                    if pd.api.types.is_numeric_dtype(data[col]):
                        first_numeric_col = col
                        break
                
                if first_numeric_col:
                    result_data[f'{first_numeric_col}_processed'] = data[first_numeric_col]
                    logger.debug(f"Added processed column based on: {first_numeric_col}")
            
            if progress_callback:
                progress_callback(100, "Legacy processing complete!")
                
            # Prepare metadata
            metadata = {
                'processor': 'magnetic',
                'script_used': 'legacy_basic',
                'data_shape': result_data.shape,
                'parameters': params
            }
            
            return ProcessingResult(
                success=True,
                data=result_data,
                metadata=metadata,
                processing_script='legacy_basic',
                export_format=params.get('output_settings', {}).get('export_format', {}).get('value', 'csv'),
                processor_type='magnetic'
            )
            
        except Exception as e:
            logger.error(f"Legacy magnetic processing failed: {str(e)}")
            return ProcessingResult(
                success=False,
                error_message=str(e),
                processor_type='magnetic'
            )
    
 