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
    """Processor for Gamma Ray Spectrometer data"""
    
    def __init__(self):
        super().__init__()
        self.name = "Gamma Processor"
        self.description = "Process gamma ray spectrometer data for radiation anomalies"
        
    def _define_parameters(self) -> Dict[str, Any]:
        """Define gamma processing parameters"""
        return {
            'calibration': {
                'energy_calibration': {
                    'value': True,
                    'type': 'bool',
                    'description': 'Apply energy calibration'
                },
                'background_subtract': {
                    'value': True,
                    'type': 'bool',
                    'description': 'Subtract background radiation'
                },
                'dead_time_correction': {
                    'value': True,
                    'type': 'bool',
                    'description': 'Apply dead time correction'
                }
            },
            'spectral_analysis': {
                'peak_detection': {
                    'value': True,
                    'type': 'bool',
                    'description': 'Detect spectral peaks'
                },
                'peak_threshold': {
                    'value': 3.0,
                    'type': 'float',
                    'min': 1.0,
                    'max': 10.0,
                    'description': 'Peak detection threshold (sigma)'
                },
                'smoothing_window': {
                    'value': 5,
                    'type': 'int',
                    'min': 3,
                    'max': 21,
                    'description': 'Spectral smoothing window size'
                }
            },
            'dose_calculation': {
                'calculate_dose': {
                    'value': True,
                    'type': 'bool',
                    'description': 'Calculate dose rate'
                },
                'dose_conversion_factor': {
                    'value': 0.0117,
                    'type': 'float',
                    'min': 0.001,
                    'max': 0.1,
                    'description': 'Counts to dose conversion factor'
                }
            },
            'isotope_identification': {
                'identify_isotopes': {
                    'value': True,
                    'type': 'bool',
                    'description': 'Attempt isotope identification'
                },
                'isotope_library': {
                    'value': 'standard',
                    'type': 'choice',
                    'choices': ['standard', 'extended', 'custom'],
                    'description': 'Isotope reference library'
                }
            }
        }
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate gamma spectrometer data"""
        detected = self.detect_columns(data)
        
        # Look for gamma-related columns
        gamma_keywords = ['gamma', 'counts', 'cps', 'spectrum', 'channel']
        gamma_cols = [col for col in data.columns if any(kw in col.lower() for kw in gamma_keywords)]
        
        if not gamma_cols and 'gamma' not in detected:
            raise ProcessingError("No gamma ray data detected")
            
        return True
    
    def process(self, data: pd.DataFrame, params: Dict[str, Any], 
                progress_callback: Optional[Callable] = None) -> ProcessingResult:
        """Process gamma ray data"""
        try:
            if progress_callback:
                progress_callback(0, "Starting gamma ray processing...")
                
            # Validate data
            self.validate_data(data)
            
            # Prepare data
            result_data = data.copy()
            
            # Find gamma data columns
            gamma_cols = [col for col in data.columns if any(
                kw in col.lower() for kw in ['gamma', 'counts', 'cps', 'spectrum']
            )]
            
            if not gamma_cols:
                raise ProcessingError("No gamma data columns found")
                
            # Use the first gamma column as primary
            primary_col = gamma_cols[0]
            
            # Background subtraction
            if progress_callback:
                progress_callback(20, "Subtracting background...")
                
            if params.get('calibration', {}).get('background_subtract', {}).get('value', True):
                # Simple background estimation using percentile
                background = np.percentile(result_data[primary_col].values, 10)
                result_data['gamma_corrected'] = result_data[primary_col] - background
                result_data['gamma_corrected'] = result_data['gamma_corrected'].clip(lower=0)
            else:
                result_data['gamma_corrected'] = result_data[primary_col]
            
            # Smoothing
            if progress_callback:
                progress_callback(40, "Smoothing spectrum...")
                
            window = params.get('spectral_analysis', {}).get('smoothing_window', {}).get('value', 5)
            result_data['gamma_smoothed'] = signal.savgol_filter(
                result_data['gamma_corrected'].values, window, 2
            )
            
            # Peak detection
            if progress_callback:
                progress_callback(60, "Detecting spectral peaks...")
                
            if params.get('spectral_analysis', {}).get('peak_detection', {}).get('value', True):
                threshold = params.get('spectral_analysis', {}).get('peak_threshold', {}).get('value', 3.0)
                peaks, properties = self._detect_peaks(
                    result_data['gamma_smoothed'].values, 
                    threshold=threshold
                )
                result_data['is_peak'] = False
                result_data.loc[peaks, 'is_peak'] = True
            
            # Dose rate calculation
            if progress_callback:
                progress_callback(80, "Calculating dose rates...")
                
            if params.get('dose_calculation', {}).get('calculate_dose', {}).get('value', True):
                factor = params.get('dose_calculation', {}).get('dose_conversion_factor', {}).get('value', 0.0117)
                result_data['dose_rate'] = result_data['gamma_corrected'] * factor
                
                # Mark high dose areas
                dose_threshold = result_data['dose_rate'].mean() + 3 * result_data['dose_rate'].std()
                result_data['is_anomaly'] = result_data['dose_rate'] > dose_threshold
            else:
                # Simple anomaly detection based on count rate
                mean_counts = result_data['gamma_corrected'].mean()
                std_counts = result_data['gamma_corrected'].std()
                result_data['is_anomaly'] = (
                    result_data['gamma_corrected'] > mean_counts + 3 * std_counts
                )
            
            # Statistical significance
            result_data['statistical_significance'] = self._calculate_significance(
                result_data['gamma_corrected'].values
            )
            
            if progress_callback:
                progress_callback(100, "Gamma processing complete!")
                
            # Prepare metadata
            metadata = {
                'processor': 'gamma',
                'mean_counts': float(result_data['gamma_corrected'].mean()),
                'max_counts': float(result_data['gamma_corrected'].max()),
                'anomalies_found': int(result_data['is_anomaly'].sum()),
                'peaks_found': int(result_data.get('is_peak', pd.Series([False])).sum()),
                'parameters': params
            }
            
            return ProcessingResult(
                success=True,
                data=result_data,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Gamma processing failed: {str(e)}")
            return ProcessingResult(
                success=False,
                error_message=str(e)
            )
    
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