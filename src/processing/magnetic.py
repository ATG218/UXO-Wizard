"""
Magnetic data processing algorithms for UXO detection
"""

import pandas as pd
import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from typing import Dict, Any, Optional, Callable
from loguru import logger

from .base import BaseProcessor, ProcessingResult, ProcessingError


class MagneticProcessor(BaseProcessor):
    """Processor for magnetic survey data"""
    
    def __init__(self):
        super().__init__()
        self.name = "Magnetic Processor"
        self.description = "Process magnetic field data for anomaly detection"
        self.required_columns = []  # Will be detected automatically
        
    def _define_parameters(self) -> Dict[str, Any]:
        """Define magnetic processing parameters"""
        return {
            'anomaly_detection': {
                'threshold_std': {
                    'value': 3.0,
                    'type': 'float',
                    'min': 1.0,
                    'max': 10.0,
                    'description': 'Anomaly threshold in standard deviations'
                },
                'window_size': {
                    'value': 50,
                    'type': 'int',
                    'min': 10,
                    'max': 500,
                    'description': 'Moving window size for background estimation'
                }
            },
            'filters': {
                'apply_kalman': {
                    'value': True,
                    'type': 'bool',
                    'description': 'Apply Kalman filter for noise reduction'
                },
                'apply_wavelet': {
                    'value': False,
                    'type': 'bool',
                    'description': 'Apply wavelet denoising'
                },
                'gaussian_sigma': {
                    'value': 2.0,
                    'type': 'float',
                    'min': 0.5,
                    'max': 10.0,
                    'description': 'Gaussian filter sigma'
                }
            },
            'advanced': {
                'reduction_to_pole': {
                    'value': False,
                    'type': 'bool',
                    'description': 'Apply Reduction to Pole transformation'
                },
                'remove_diurnal': {
                    'value': True,
                    'type': 'bool',
                    'description': 'Remove diurnal variations'
                }
            }
        }
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate magnetic data"""
        # Auto-detect magnetic field column
        detected = self.detect_columns(data)
        if 'magnetic' not in detected:
            # Try common column names
            mag_cols = ['Btotal1 [nT]', 'Btotal2 [nT]', 'Mag', 'Field', 'nT']
            found = False
            for col in data.columns:
                if any(mc.lower() in col.lower() for mc in mag_cols):
                    found = True
                    break
            if not found:
                raise ProcessingError("No magnetic field column detected")
        
        # Check for coordinate columns
        if 'latitude' not in detected or 'longitude' not in detected:
            raise ProcessingError("Latitude and longitude columns required")
            
        return True
    
    def process(self, data: pd.DataFrame, params: Dict[str, Any], 
                progress_callback: Optional[Callable] = None) -> ProcessingResult:
        """Process magnetic data"""
        try:
            if progress_callback:
                progress_callback(0, "Starting magnetic processing...")
            
            # Validate data
            self.validate_data(data)
            
            # Detect columns
            cols = self.detect_columns(data)
            mag_col = cols.get('magnetic', self._find_mag_column(data))
            
            # Copy data
            result_data = data.copy()
            
            # Preprocessing
            if progress_callback:
                progress_callback(10, "Preprocessing data...")
            result_data = self.preprocess(result_data)
            
            # Apply filters
            if params.get('filters', {}).get('apply_kalman', {}).get('value', True):
                if progress_callback:
                    progress_callback(30, "Applying Kalman filter...")
                result_data['mag_filtered'] = self._kalman_filter(result_data[mag_col].values)
            else:
                result_data['mag_filtered'] = result_data[mag_col]
            
            if params.get('filters', {}).get('gaussian_sigma', {}).get('value', 0) > 0:
                if progress_callback:
                    progress_callback(40, "Applying Gaussian filter...")
                sigma = params['filters']['gaussian_sigma']['value']
                result_data['mag_filtered'] = gaussian_filter1d(
                    result_data['mag_filtered'].values, sigma=sigma
                )
            
            # Anomaly detection
            if progress_callback:
                progress_callback(60, "Detecting anomalies...")
            
            threshold = params.get('anomaly_detection', {}).get('threshold_std', {}).get('value', 3.0)
            window = params.get('anomaly_detection', {}).get('window_size', {}).get('value', 50)
            
            result_data['anomaly_score'] = self._detect_anomalies(
                result_data['mag_filtered'].values, 
                threshold=threshold,
                window_size=window
            )
            
            # Calculate gradient (useful for edge detection)
            if progress_callback:
                progress_callback(80, "Calculating gradients...")
            result_data['mag_gradient'] = np.gradient(result_data['mag_filtered'].values)
            
            # Mark anomalies
            mean_val = result_data['mag_filtered'].mean()
            std_val = result_data['mag_filtered'].std()
            result_data['is_anomaly'] = (
                np.abs(result_data['mag_filtered'] - mean_val) > threshold * std_val
            )
            
            if progress_callback:
                progress_callback(100, "Processing complete!")
                
            # Prepare metadata
            metadata = {
                'processor': 'magnetic',
                'anomalies_found': int(result_data['is_anomaly'].sum()),
                'mean_field': float(mean_val),
                'std_field': float(std_val),
                'parameters': params
            }
            
            return ProcessingResult(
                success=True,
                data=result_data,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Magnetic processing failed: {str(e)}")
            return ProcessingResult(
                success=False,
                error_message=str(e)
            )
    
    def _find_mag_column(self, data: pd.DataFrame) -> str:
        """Find the magnetic field column"""
        mag_keywords = ['btotal', 'mag', 'field', 'nt', 'tesla']
        for col in data.columns:
            if any(kw in col.lower() for kw in mag_keywords):
                return col
        raise ProcessingError("Could not find magnetic field column")
    
    def _kalman_filter(self, data: np.ndarray) -> np.ndarray:
        """Simple Kalman filter implementation"""
        # Initialize
        n = len(data)
        filtered = np.zeros(n)
        
        # Initial estimates
        x_est = data[0]
        p_est = 1.0
        
        # Process and measurement noise
        q = 0.01  # Process noise
        r = 0.1   # Measurement noise
        
        for i in range(n):
            # Prediction
            x_pred = x_est
            p_pred = p_est + q
            
            # Update
            k_gain = p_pred / (p_pred + r)
            x_est = x_pred + k_gain * (data[i] - x_pred)
            p_est = (1 - k_gain) * p_pred
            
            filtered[i] = x_est
            
        return filtered
    
    def _detect_anomalies(self, data: np.ndarray, threshold: float = 3.0, 
                         window_size: int = 50) -> np.ndarray:
        """Detect anomalies using moving window statistics"""
        n = len(data)
        anomaly_scores = np.zeros(n)
        
        for i in range(n):
            # Define window
            start = max(0, i - window_size // 2)
            end = min(n, i + window_size // 2)
            
            # Calculate local statistics
            window_data = data[start:end]
            local_mean = np.mean(window_data)
            local_std = np.std(window_data)
            
            # Calculate anomaly score
            if local_std > 0:
                anomaly_scores[i] = abs(data[i] - local_mean) / local_std
            else:
                anomaly_scores[i] = 0
                
        return anomaly_scores 