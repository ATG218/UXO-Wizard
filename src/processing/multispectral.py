"""
Multispectral image data processing
"""

import pandas as pd
import numpy as np
from scipy import ndimage
from typing import Dict, Any, Optional, Callable, List
from loguru import logger

from .base import BaseProcessor, ProcessingResult, ProcessingError


class MultispectralProcessor(BaseProcessor):
    """Processor for Multispectral imaging data"""
    
    def __init__(self):
        super().__init__()
        self.name = "Multispectral Processor"
        self.description = "Process multispectral imagery for anomaly detection"
        
    def _define_parameters(self) -> Dict[str, Any]:
        """Define multispectral processing parameters"""
        return {
            'band_math': {
                'calculate_ndvi': {
                    'value': True,
                    'type': 'bool',
                    'description': 'Calculate NDVI (vegetation index)'
                },
                'calculate_ndwi': {
                    'value': False,
                    'type': 'bool',
                    'description': 'Calculate NDWI (water index)'
                },
                'calculate_bai': {
                    'value': True,
                    'type': 'bool',
                    'description': 'Calculate BAI (bare soil index)'
                }
            },
            'enhancement': {
                'histogram_equalization': {
                    'value': True,
                    'type': 'bool',
                    'description': 'Apply histogram equalization'
                },
                'contrast_stretch': {
                    'value': True,
                    'type': 'bool',
                    'description': 'Apply contrast stretching'
                },
                'stretch_percentile': {
                    'value': 2,
                    'type': 'int',
                    'min': 0,
                    'max': 5,
                    'description': 'Percentile for contrast stretch'
                }
            },
            'classification': {
                'anomaly_detection': {
                    'value': True,
                    'type': 'bool',
                    'description': 'Detect spectral anomalies'
                },
                'anomaly_method': {
                    'value': 'rxd',
                    'type': 'choice',
                    'choices': ['rxd', 'matched_filter', 'ace'],
                    'description': 'Anomaly detection algorithm'
                },
                'pca_components': {
                    'value': 3,
                    'type': 'int',
                    'min': 1,
                    'max': 10,
                    'description': 'Number of PCA components'
                }
            },
            'spatial_analysis': {
                'texture_analysis': {
                    'value': True,
                    'type': 'bool',
                    'description': 'Calculate texture features'
                },
                'edge_detection': {
                    'value': True,
                    'type': 'bool',
                    'description': 'Detect edges and boundaries'
                },
                'morphology': {
                    'value': False,
                    'type': 'bool',
                    'description': 'Apply morphological operations'
                }
            }
        }
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate multispectral data"""
        # Look for band columns
        band_keywords = ['band', 'channel', 'wavelength', 'nm', 'red', 'green', 'blue', 'nir', 'swir']
        band_cols = [col for col in data.columns if any(kw in col.lower() for kw in band_keywords)]
        
        if len(band_cols) < 2:
            raise ProcessingError("At least 2 spectral bands required for multispectral processing")
            
        return True
    
    def process(self, data: pd.DataFrame, params: Dict[str, Any], 
                progress_callback: Optional[Callable] = None) -> ProcessingResult:
        """Process multispectral data"""
        try:
            if progress_callback:
                progress_callback(0, "Starting multispectral processing...")
                
            # Validate data
            self.validate_data(data)
            
            # Prepare data
            result_data = data.copy()
            
            # Find band columns
            band_cols = self._detect_bands(data)
            
            if len(band_cols) < 2:
                raise ProcessingError("Insufficient spectral bands found")
                
            # Calculate band indices
            if progress_callback:
                progress_callback(20, "Calculating spectral indices...")
                
            if params.get('band_math', {}).get('calculate_ndvi', {}).get('value', True):
                if 'red' in band_cols and 'nir' in band_cols:
                    result_data['ndvi'] = self._calculate_ndvi(
                        result_data[band_cols['red']].values,
                        result_data[band_cols['nir']].values
                    )
            
            # Enhancement
            if progress_callback:
                progress_callback(40, "Enhancing spectral data...")
                
            if params.get('enhancement', {}).get('contrast_stretch', {}).get('value', True):
                percentile = params.get('enhancement', {}).get('stretch_percentile', {}).get('value', 2)
                for band_name, band_col in band_cols.items():
                    if band_col in result_data.columns:
                        result_data[f'{band_col}_enhanced'] = self._contrast_stretch(
                            result_data[band_col].values, percentile
                        )
            
            # Anomaly detection
            if progress_callback:
                progress_callback(60, "Detecting spectral anomalies...")
                
            if params.get('classification', {}).get('anomaly_detection', {}).get('value', True):
                method = params.get('classification', {}).get('anomaly_method', {}).get('value', 'rxd')
                
                # Create band matrix
                band_matrix = np.column_stack([
                    result_data[col].values for col in band_cols.values() 
                    if col in result_data.columns
                ])
                
                if method == 'rxd':
                    anomaly_scores = self._rxd_anomaly_detection(band_matrix)
                    result_data['anomaly_score'] = anomaly_scores
                    
                    # Threshold for anomalies
                    threshold = np.percentile(anomaly_scores, 95)
                    result_data['is_anomaly'] = anomaly_scores > threshold
            
            # Texture analysis
            if progress_callback:
                progress_callback(80, "Analyzing spatial features...")
                
            if params.get('spatial_analysis', {}).get('texture_analysis', {}).get('value', True):
                # Calculate texture for first band
                first_band = list(band_cols.values())[0]
                if first_band in result_data.columns:
                    result_data['texture_variance'] = self._calculate_texture(
                        result_data[first_band].values
                    )
            
            if progress_callback:
                progress_callback(100, "Multispectral processing complete!")
                
            # Prepare metadata
            metadata = {
                'processor': 'multispectral',
                'bands_processed': len(band_cols),
                'anomalies_found': int(result_data.get('is_anomaly', pd.Series([False])).sum()),
                'indices_calculated': [],
                'parameters': params
            }
            
            if 'ndvi' in result_data.columns:
                metadata['indices_calculated'].append('NDVI')
                metadata['ndvi_mean'] = float(result_data['ndvi'].mean())
            
            return ProcessingResult(
                success=True,
                data=result_data,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Multispectral processing failed: {str(e)}")
            return ProcessingResult(
                success=False,
                error_message=str(e)
            )
    
    def _detect_bands(self, data: pd.DataFrame) -> Dict[str, str]:
        """Detect spectral band columns"""
        band_mapping = {}
        
        # Common band names and keywords
        band_definitions = {
            'blue': ['blue', 'b1', 'band1', '450', '490'],
            'green': ['green', 'b2', 'band2', '520', '560'],
            'red': ['red', 'b3', 'band3', '630', '690'],
            'nir': ['nir', 'near_infrared', 'b4', 'band4', '770', '860'],
            'swir1': ['swir1', 'shortwave1', 'b5', 'band5', '1550', '1650'],
            'swir2': ['swir2', 'shortwave2', 'b6', 'band6', '2100', '2300']
        }
        
        for band_name, keywords in band_definitions.items():
            for col in data.columns:
                col_lower = col.lower()
                if any(kw in col_lower for kw in keywords):
                    band_mapping[band_name] = col
                    break
                    
        return band_mapping
    
    def _calculate_ndvi(self, red: np.ndarray, nir: np.ndarray) -> np.ndarray:
        """Calculate Normalized Difference Vegetation Index"""
        with np.errstate(divide='ignore', invalid='ignore'):
            ndvi = (nir - red) / (nir + red)
            ndvi[~np.isfinite(ndvi)] = 0
        return ndvi
    
    def _contrast_stretch(self, data: np.ndarray, percentile: int = 2) -> np.ndarray:
        """Apply percentile-based contrast stretching"""
        p_low = np.percentile(data, percentile)
        p_high = np.percentile(data, 100 - percentile)
        
        stretched = (data - p_low) / (p_high - p_low)
        return np.clip(stretched, 0, 1)
    
    def _rxd_anomaly_detection(self, data: np.ndarray) -> np.ndarray:
        """Reed-Xiaoli Detector for spectral anomaly detection"""
        # Remove invalid values
        valid_mask = np.all(np.isfinite(data), axis=1)
        valid_data = data[valid_mask]
        
        if len(valid_data) < 10:
            return np.zeros(len(data))
            
        # Calculate mean and covariance
        mean = np.mean(valid_data, axis=0)
        cov = np.cov(valid_data.T)
        
        # Add regularization to avoid singular matrix
        cov += np.eye(cov.shape[0]) * 1e-6
        
        try:
            inv_cov = np.linalg.inv(cov)
        except:
            # Fallback to pseudo-inverse
            inv_cov = np.linalg.pinv(cov)
        
        # Calculate RXD scores
        scores = np.zeros(len(data))
        for i in range(len(data)):
            if valid_mask[i]:
                diff = data[i] - mean
                scores[i] = np.sqrt(diff @ inv_cov @ diff.T)
                
        return scores
    
    def _calculate_texture(self, data: np.ndarray, window_size: int = 5) -> np.ndarray:
        """Calculate local texture variance"""
        # Simple texture measure using local variance
        texture = np.zeros_like(data)
        half_window = window_size // 2
        
        for i in range(len(data)):
            start = max(0, i - half_window)
            end = min(len(data), i + half_window + 1)
            window = data[start:end]
            texture[i] = np.var(window) if len(window) > 1 else 0
            
        return texture 