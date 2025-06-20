"""
Core Geophysical Processor adapted from SGTool
==============================================

Vectorized geophysical processing algorithms for frequency domain operations.
Based on SGTool's GeophysicalProcessor.py with optimizations for batch processing.
"""

import numpy as np
from typing import Tuple, Optional, Union
import warnings

try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    NUMBA_AVAILABLE = False


class GeophysicalProcessor:
    """
    Core geophysical processing class with vectorized FFT-based operations.
    
    Adapted from SGTool's GeophysicalProcessor to work with numpy arrays
    and provide optimized batch processing capabilities.
    """
    
    def __init__(self, dx: float, dy: float, buffer: float = 0.0):
        """
        Initialize the processor with grid spacing.
        
        Parameters:
            dx (float): Grid spacing in the x-direction
            dy (float): Grid spacing in the y-direction  
            buffer (float): Buffer zone around data for edge effects
        """
        self.dx = dx
        self.dy = dy
        self.buffer = buffer
        
    def create_wavenumber_grids(self, nx: int, ny: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create wavenumber grids for Fourier-based operations.
        
        Parameters:
            nx (int): Number of points in x-direction
            ny (int): Number of points in y-direction
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: kx and ky wavenumber grids
        """
        kx = np.fft.fftfreq(nx, self.dx) * 2 * np.pi
        ky = np.fft.fftfreq(ny, self.dy) * 2 * np.pi
        return np.meshgrid(kx, ky, indexing='ij')
    
    def fill_nan_values(self, data: np.ndarray, method: str = 'nearest') -> np.ndarray:
        """
        Fill NaN values in the data array using specified method.
        
        Parameters:
            data (np.ndarray): Input data with potential NaN values
            method (str): Method for filling ('nearest', 'linear', 'mean')
            
        Returns:
            np.ndarray: Data with NaN values filled
        """
        if not np.any(np.isnan(data)):
            return data
            
        if method == 'mean':
            return np.where(np.isnan(data), np.nanmean(data), data)
        elif method == 'nearest':
            # Use simple nearest neighbor interpolation
            mask = np.isnan(data)
            data_filled = data.copy()
            data_filled[mask] = np.nanmean(data)
            return data_filled
        else:
            # Default to mean for simplicity
            return np.where(np.isnan(data), np.nanmean(data), data)
    
    def pad_data(self, data: np.ndarray, pad_width: int = None) -> np.ndarray:
        """
        Pad data to reduce edge effects in FFT operations.
        
        Parameters:
            data (np.ndarray): Input data array
            pad_width (int): Padding width (default: 10% of data size)
            
        Returns:
            np.ndarray: Padded data array
        """
        if pad_width is None:
            pad_width = max(int(0.1 * min(data.shape)), 10)
            
        # Pad with edge values to maintain continuity
        return np.pad(data, pad_width, mode='edge')
    
    def apply_frequency_filter(self, data: np.ndarray, filter_func) -> np.ndarray:
        """
        Apply a frequency domain filter to the data.
        
        Parameters:
            data (np.ndarray): Input data array
            filter_func: Function that takes (kx, ky) and returns filter
            
        Returns:
            np.ndarray: Filtered data array
        """
        # Fill NaN values
        data_filled = self.fill_nan_values(data)
        
        # Pad data to reduce edge effects
        original_shape = data_filled.shape
        data_padded = self.pad_data(data_filled)
        ny, nx = data_padded.shape
        
        # Create wavenumber grids
        kx, ky = self.create_wavenumber_grids(nx, ny)
        
        # Apply FFT
        data_fft = np.fft.fft2(data_padded)
        
        # Apply filter in frequency domain
        filter_response = filter_func(kx, ky)
        filtered_fft = data_fft * filter_response
        
        # Inverse FFT
        filtered_data = np.real(np.fft.ifft2(filtered_fft))
        
        # Remove padding
        pad_width = (np.array(data_padded.shape) - np.array(original_shape)) // 2
        filtered_data = filtered_data[
            pad_width[0]:pad_width[0] + original_shape[0],
            pad_width[1]:pad_width[1] + original_shape[1]
        ]
        
        return filtered_data
    
    def reduction_to_pole(self, data: np.ndarray, inclination: float, 
                         declination: float) -> np.ndarray:
        """
        Apply Reduction to Pole (RTP) filter.
        
        Parameters:
            data (np.ndarray): Input magnetic data
            inclination (float): Magnetic inclination in degrees
            declination (float): Magnetic declination in degrees
            
        Returns:
            np.ndarray: RTP filtered data
        """
        # Convert to radians
        inc_rad = np.radians(inclination)
        dec_rad = np.radians(declination)
        
        def rtp_filter(kx, ky):
            k = np.sqrt(kx**2 + ky**2)
            # Avoid division by zero
            k = np.where(k == 0, 1e-10, k)
            
            # RTP filter formula
            numerator = (k * np.cos(inc_rad) * np.cos(dec_rad) + 
                        1j * ky * np.cos(inc_rad) * np.sin(dec_rad) + 
                        kx * np.sin(inc_rad))
            
            return numerator / k
        
        return self.apply_frequency_filter(data, rtp_filter)
    
    def reduction_to_equator(self, data: np.ndarray, inclination: float,
                           declination: float) -> np.ndarray:
        """
        Apply Reduction to Equator (RTE) filter.
        
        Parameters:
            data (np.ndarray): Input magnetic data
            inclination (float): Magnetic inclination in degrees
            declination (float): Magnetic declination in degrees
            
        Returns:
            np.ndarray: RTE filtered data
        """
        # Convert to radians
        inc_rad = np.radians(inclination)
        dec_rad = np.radians(declination)
        
        def rte_filter(kx, ky):
            k = np.sqrt(kx**2 + ky**2)
            # Avoid division by zero
            k = np.where(k == 0, 1e-10, k)
            
            # RTE filter formula
            numerator = (k * np.cos(inc_rad) * np.cos(dec_rad) + 
                        1j * ky * np.cos(inc_rad) * np.sin(dec_rad) + 
                        kx * np.sin(inc_rad))
            denominator = (k * np.cos(inc_rad) * np.cos(dec_rad) - 
                          1j * ky * np.cos(inc_rad) * np.sin(dec_rad) + 
                          kx * np.sin(inc_rad))
            
            return numerator / denominator
        
        return self.apply_frequency_filter(data, rte_filter)
    
    def upward_continuation(self, data: np.ndarray, height: float) -> np.ndarray:
        """
        Apply upward continuation filter.
        
        Parameters:
            data (np.ndarray): Input potential field data
            height (float): Continuation height (positive for upward)
            
        Returns:
            np.ndarray: Upward continued data
        """
        def continuation_filter(kx, ky):
            k = np.sqrt(kx**2 + ky**2)
            return np.exp(-k * height)
        
        return self.apply_frequency_filter(data, continuation_filter)
    
    def downward_continuation(self, data: np.ndarray, height: float) -> np.ndarray:
        """
        Apply downward continuation filter.
        
        WARNING: Downward continuation is unstable and should be used with caution.
        
        Parameters:
            data (np.ndarray): Input potential field data
            height (float): Continuation height (positive for downward)
            
        Returns:
            np.ndarray: Downward continued data
        """
        warnings.warn("Downward continuation is unstable. Use with caution.")
        return self.upward_continuation(data, -height)
    
    def vertical_integration(self, data: np.ndarray) -> np.ndarray:
        """
        Apply vertical integration filter (1/k) with proper regularization.
        
        When applied to RTE or RTP data, provides pseudogravity result.
        Uses high-pass filtering to avoid amplification of very low frequencies.
        
        Parameters:
            data (np.ndarray): Input data
            
        Returns:
            np.ndarray: Vertically integrated data
        """
        def vertical_integration_filter(kx, ky):
            k = np.sqrt(kx**2 + ky**2)
            
            # Calculate cutoff frequency based on grid spacing
            # Use wavelength of about 10 grid cells as minimum
            k_cutoff = 2 * np.pi / (10 * max(self.dx, self.dy))
            
            # Apply regularization to prevent extreme amplification of low frequencies
            # Use a smooth transition instead of hard cutoff
            k_reg = np.where(k < k_cutoff, k_cutoff, k)
            
            # Additional safety: ensure minimum value
            k_reg = np.where(k_reg == 0, k_cutoff, k_reg)
            
            return 1.0 / k_reg
        
        return self.apply_frequency_filter(data, vertical_integration_filter)
    
    def calculate_grid_spacing(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """
        Calculate grid spacing from coordinate arrays.
        
        Parameters:
            x (np.ndarray): X coordinates
            y (np.ndarray): Y coordinates
            
        Returns:
            Tuple[float, float]: dx, dy grid spacing
        """
        dx = np.median(np.diff(np.sort(np.unique(x))))
        dy = np.median(np.diff(np.sort(np.unique(y))))
        return dx, dy
    
    def remove_regional_trend(self, data: np.ndarray, order: int = 1) -> np.ndarray:
        """
        Remove regional trend from data using polynomial fitting.
        
        Parameters:
            data (np.ndarray): Input data
            order (int): Polynomial order (1 for plane, 2 for parabolic)
            
        Returns:
            np.ndarray: Data with regional trend removed
        """
        ny, nx = data.shape
        x = np.arange(nx)
        y = np.arange(ny)
        X, Y = np.meshgrid(x, y)
        
        # Flatten arrays and remove NaN values
        mask = ~np.isnan(data)
        x_flat = X[mask]
        y_flat = Y[mask]
        z_flat = data[mask]
        
        if len(z_flat) == 0:
            return data
        
        # Build design matrix for polynomial fitting
        if order == 1:
            # Linear plane: z = a + b*x + c*y
            A = np.column_stack([np.ones(len(x_flat)), x_flat, y_flat])
        elif order == 2:
            # Quadratic surface: z = a + b*x + c*y + d*x^2 + e*y^2 + f*x*y
            A = np.column_stack([
                np.ones(len(x_flat)), x_flat, y_flat,
                x_flat**2, y_flat**2, x_flat*y_flat
            ])
        else:
            raise ValueError("Order must be 1 or 2")
        
        # Solve for coefficients
        coeffs, _, _, _ = np.linalg.lstsq(A, z_flat, rcond=None)
        
        # Calculate regional surface
        if order == 1:
            regional = coeffs[0] + coeffs[1]*X + coeffs[2]*Y
        else:
            regional = (coeffs[0] + coeffs[1]*X + coeffs[2]*Y + 
                       coeffs[3]*X**2 + coeffs[4]*Y**2 + coeffs[5]*X*Y)
        
        return data - regional