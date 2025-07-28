"""
Ground Penetrating Radar (GPR) data processing for UXO detection - Script Integration Framework
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Callable
from loguru import logger

from .base import BaseProcessor, ProcessingResult, ProcessingError


class GPRProcessor(BaseProcessor):
    """Processor for Ground Penetrating Radar data with script framework support"""
    
    def __init__(self, project_manager=None):
        super().__init__(project_manager=project_manager)
        self.name = "GPR Processor"
        self.description = "Process GPR data using various processing scripts for UXO detection"
        self.processor_type = "gpr"  # Essential for script discovery
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
        
    def _define_base_parameters(self) -> Dict[str, Any]:
        """Define base GPR parameters for fallback"""
        # No base parameters needed for GPR as scripts define everything.
        return {}
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate GPR data - basic checks for script integration"""
        if data.empty:
            raise ProcessingError("No data provided")
            
        # Basic column detection for GPR data
        detected = self.detect_columns(data)
        logger.debug(f"Detected columns: {detected}")
        
        # Check for basic data structure
        if len(data.columns) < 2:
            raise ProcessingError("Data must have at least 2 columns")
        
        # Look for GPR-specific indicators
        gpr_indicators = ['gpr', 'trace', 'amplitude', 'sample', 'radar']
        has_gpr_data = any(
            any(indicator in str(col).lower() for indicator in gpr_indicators)
            for col in data.columns
        )
        
        # If no clear GPR indicators, check if data structure suggests trace data
        if not has_gpr_data:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 10:  # GPR data typically has many traces/samples
                logger.warning("Data may not be GPR trace data - consider checking format")
                
        return True
    
    # Note: The process method is now handled by the BaseProcessor script framework
    # Individual processing scripts implement their own execute methods 