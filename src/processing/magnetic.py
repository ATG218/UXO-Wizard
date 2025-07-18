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
    
    def __init__(self, project_manager=None):
        super().__init__(project_manager=project_manager)
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