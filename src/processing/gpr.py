"""
Ground Penetrating Radar (GPR) data processing
"""

import pandas as pd
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
from typing import Dict, Any, Optional, Callable
from loguru import logger

from .base import BaseProcessor, ProcessingResult, ProcessingError


class GPRProcessor(BaseProcessor):
    """Processor for Ground Penetrating Radar data"""
    
    def __init__(self, project_manager=None):
        super().__init__(project_manager=project_manager)
        self.name = "GPR Processor"
        self.description = "Process GPR data for subsurface anomaly detection"
        
    def _define_parameters(self) -> Dict[str, Any]:
        """Define GPR processing parameters"""
        return {
            'preprocessing': {
                'remove_dc_bias': {
                    'value': True,
                    'type': 'bool',
                    'description': 'Remove DC component from traces'
                },
                'time_zero_correction': {
                    'value': True,
                    'type': 'bool',
                    'description': 'Adjust time zero position'
                },
                'background_removal': {
                    'value': True,
                    'type': 'bool',
                    'description': 'Remove horizontal banding'
                }
            },
            'filtering': {
                'bandpass_low': {
                    'value': 10.0,
                    'type': 'float',
                    'min': 1.0,
                    'max': 100.0,
                    'description': 'Low frequency cutoff (MHz)'
                },
                'bandpass_high': {
                    'value': 500.0,
                    'type': 'float',
                    'min': 50.0,
                    'max': 2000.0,
                    'description': 'High frequency cutoff (MHz)'
                },
                'dewow': {
                    'value': True,
                    'type': 'bool',
                    'description': 'Apply dewow filter'
                }
            },
            'gain': {
                'apply_gain': {
                    'value': True,
                    'type': 'bool',
                    'description': 'Apply time-varying gain'
                },
                'gain_type': {
                    'value': 'exponential',
                    'type': 'choice',
                    'choices': ['linear', 'exponential', 'agc'],
                    'description': 'Type of gain function'
                }
            },
            'migration': {
                'apply_migration': {
                    'value': False,
                    'type': 'bool',
                    'description': 'Apply Kirchhoff migration'
                },
                'velocity': {
                    'value': 0.1,
                    'type': 'float',
                    'min': 0.05,
                    'max': 0.3,
                    'description': 'EM wave velocity (m/ns)'
                }
            }
        }
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate GPR data format"""
        # Check for required columns
        detected = self.detect_columns(data)
        
        # GPR data might have trace/sample columns or be in wide format
        if 'gpr' not in detected:
            # Look for trace data columns
            trace_cols = [col for col in data.columns if 'trace' in col.lower() or 'sample' in col.lower()]
            if not trace_cols:
                raise ProcessingError("No GPR trace data detected")
                
        return True
    
    def process(self, data: pd.DataFrame, params: Dict[str, Any], 
                progress_callback: Optional[Callable] = None) -> ProcessingResult:
        """Process GPR data"""
        try:
            if progress_callback:
                progress_callback(0, "Starting GPR processing...")
                
            # Validate data
            self.validate_data(data)
            
            # Prepare data
            result_data = data.copy()
            
            # Detect GPR data columns
            gpr_cols = [col for col in data.columns if any(
                kw in col.lower() for kw in ['gpr', 'trace', 'amplitude', 'sample']
            )]
            
            if not gpr_cols:
                raise ProcessingError("No GPR data columns found")
                
            # Process each trace
            if progress_callback:
                progress_callback(20, "Preprocessing traces...")
                
            # Apply preprocessing
            if params.get('preprocessing', {}).get('remove_dc_bias', {}).get('value', True):
                for col in gpr_cols:
                    if col in result_data.columns:
                        result_data[col] = result_data[col] - result_data[col].mean()
            
            # Apply filtering
            if progress_callback:
                progress_callback(40, "Applying filters...")
                
            if params.get('filtering', {}).get('dewow', {}).get('value', True):
                for col in gpr_cols:
                    if col in result_data.columns:
                        result_data[col] = self._dewow_filter(result_data[col].values)
            
            # Apply gain
            if progress_callback:
                progress_callback(60, "Applying gain correction...")
                
            if params.get('gain', {}).get('apply_gain', {}).get('value', True):
                gain_type = params.get('gain', {}).get('gain_type', {}).get('value', 'exponential')
                for col in gpr_cols:
                    if col in result_data.columns:
                        result_data[col] = self._apply_gain(result_data[col].values, gain_type)
            
            # Detect anomalies (simplified - look for high amplitude reflections)
            if progress_callback:
                progress_callback(80, "Detecting subsurface anomalies...")
                
            # Calculate envelope for anomaly detection
            if gpr_cols:
                primary_col = gpr_cols[0]
                envelope = np.abs(signal.hilbert(result_data[primary_col].values))
                result_data['gpr_envelope'] = envelope
                
                # Simple anomaly detection based on envelope threshold
                threshold = np.percentile(envelope, 95)
                result_data['is_anomaly'] = envelope > threshold
                
            if progress_callback:
                progress_callback(100, "GPR processing complete!")
                
            # Prepare metadata
            metadata = {
                'processor': 'gpr',
                'traces_processed': len(gpr_cols),
                'anomalies_found': int(result_data.get('is_anomaly', pd.Series([False])).sum()),
                'parameters': params
            }
            
            return ProcessingResult(
                success=True,
                data=result_data,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"GPR processing failed: {str(e)}")
            return ProcessingResult(
                success=False,
                error_message=str(e)
            )
    
    def _dewow_filter(self, trace: np.ndarray, window: int = 10) -> np.ndarray:
        """Remove low-frequency wow from GPR trace"""
        # Simple high-pass filter using running average
        running_avg = np.convolve(trace, np.ones(window)/window, mode='same')
        return trace - running_avg
    
    def _apply_gain(self, trace: np.ndarray, gain_type: str = 'exponential') -> np.ndarray:
        """Apply time-varying gain to compensate for signal attenuation"""
        n_samples = len(trace)
        t = np.arange(n_samples)
        
        if gain_type == 'linear':
            gain = 1 + t / n_samples * 5  # Linear gain up to 6x
        elif gain_type == 'exponential':
            gain = np.exp(t / n_samples * 2)  # Exponential gain
        elif gain_type == 'agc':  # Automatic Gain Control
            window = 50
            gain = np.ones_like(trace)
            for i in range(n_samples):
                start = max(0, i - window // 2)
                end = min(n_samples, i + window // 2)
                local_max = np.max(np.abs(trace[start:end]))
                if local_max > 0:
                    gain[i] = 1.0 / local_max
        else:
            gain = np.ones_like(trace)
            
        return trace * gain 