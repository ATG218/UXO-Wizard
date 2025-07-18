"""
Gamma Ray Spectrometer data processing
"""

import pandas as pd
import numpy as np
from scipy import signal
from scipy.stats import poisson
from typing import Dict, Any, Optional, Callable
from loguru import logger

from .base import BaseProcessor, ProcessingResult, ProcessingError


class GammaProcessor(BaseProcessor):
    """Processor for Gamma Ray Spectrometer data with script framework support"""
    
    def __init__(self, project_manager=None):
        super().__init__(project_manager=project_manager)
        self.name = "Gamma Processor"
        self.description = "Process gamma ray spectrometer data using various processing scripts"
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
        """Validate gamma data - basic checks for script integration"""
        if data.empty:
            raise ProcessingError("No data provided")
            
        # Basic column detection for gamma data
        detected = self.detect_columns(data)
        logger.debug(f"Detected columns: {detected}")
        
        # Check for basic data structure
        if len(data.columns) < 2:
            raise ProcessingError("Data must have at least 2 columns")
            
        return True
    
    def _detect_peaks(self, data: np.ndarray, threshold: float = 3.0) -> tuple:
        """Detect peaks in gamma spectrum"""
        # Calculate noise level
        noise_level = np.median(np.abs(data - np.median(data))) * 1.4826  # MAD estimator
        
        # Find peaks
        peak_threshold = np.median(data) + threshold * noise_level
        peaks, properties = signal.find_peaks(
            data, 
            height=peak_threshold,
            distance=5,  # Minimum distance between peaks
            prominence=noise_level
        )
        
        return peaks, properties
    
    def _calculate_significance(self, counts: np.ndarray) -> np.ndarray:
        """Calculate statistical significance of gamma counts"""
        # Assuming Poisson statistics
        background = np.median(counts)
        
        # Calculate z-score for Poisson distribution
        if background > 0:
            z_scores = (counts - background) / np.sqrt(background)
        else:
            z_scores = np.zeros_like(counts)
            
        return z_scores
    
    def _identify_isotopes(self, energy_spectrum: np.ndarray, peaks: np.ndarray) -> Dict[str, Any]:
        """Basic isotope identification based on peak energies"""
        # Simplified isotope library (energy in keV)
        isotope_library = {
            'K-40': [1461],
            'Cs-137': [662],
            'Co-60': [1173, 1333],
            'U-238': [1001, 766, 1120],
            'Th-232': [2614, 583, 911]
        }
        
        identified = {}
        # This would require energy calibration to work properly
        # Placeholder for now
        
        return identified 