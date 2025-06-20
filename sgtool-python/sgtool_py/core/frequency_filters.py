"""
Frequency Domain Filters
========================

Vectorized frequency domain filtering operations based on SGTool.
Includes high-pass, low-pass, band-pass filters with cosine rolloff.
"""

import numpy as np
from typing import Tuple, Union, Optional
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


class FrequencyFilters:
    """
    Frequency domain filtering operations with cosine rolloff transitions.
    
    Based on SGTool's frequency filtering with optimizations for reduced ringing.
    """
    
    def __init__(self, dx: float, dy: float):
        """
        Initialize frequency filters.
        
        Parameters:
            dx (float): Grid spacing in x-direction
            dy (float): Grid spacing in y-direction
        """
        self.dx = dx
        self.dy = dy
    
    def create_wavenumber_grids(self, nx: int, ny: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create wavenumber grids for frequency operations."""
        kx = np.fft.fftfreq(nx, self.dx) * 2 * np.pi
        ky = np.fft.fftfreq(ny, self.dy) * 2 * np.pi
        return np.meshgrid(kx, ky, indexing='ij')
    
    def wavelength_to_wavenumber(self, wavelength: float) -> float:
        """Convert wavelength to wavenumber."""
        return 2 * np.pi / wavelength
    
    def high_pass_filter(self, data: np.ndarray, cutoff_wavelength: float,
                        transition_width: Optional[float] = None) -> np.ndarray:
        """
        Apply high-pass filter with cosine rolloff.
        
        Parameters:
            data (np.ndarray): Input data
            cutoff_wavelength (float): Cutoff wavelength
            transition_width (float): Transition width (default: 0.2 * cutoff_wavelength)
            
        Returns:
            np.ndarray: High-pass filtered data
        """
        if transition_width is None:
            transition_width = 0.2 * cutoff_wavelength
            
        ny, nx = data.shape
        kx, ky = self.create_wavenumber_grids(nx, ny)
        k = np.sqrt(kx**2 + ky**2)
        
        # Calculate filter boundaries
        k_low = self.wavelength_to_wavenumber(cutoff_wavelength + transition_width/2)
        k_high = self.wavelength_to_wavenumber(cutoff_wavelength - transition_width/2)
        
        # Create filter with cosine rolloff
        filter_response = np.zeros_like(k)
        
        # Pass band (k >= k_high)
        filter_response[k >= k_high] = 1.0
        
        # Transition band (k_low < k < k_high)
        transition_mask = (k > k_low) & (k < k_high)
        filter_response[transition_mask] = 0.5 * (
            1 - np.cos(np.pi * (k[transition_mask] - k_low) / (k_high - k_low))
        )
        
        # Stop band (k <= k_low) remains 0
        
        return self._apply_filter(data, filter_response)
    
    def low_pass_filter(self, data: np.ndarray, cutoff_wavelength: float,
                       transition_width: Optional[float] = None) -> np.ndarray:
        """
        Apply low-pass filter with cosine rolloff.
        
        Parameters:
            data (np.ndarray): Input data
            cutoff_wavelength (float): Cutoff wavelength
            transition_width (float): Transition width (default: 0.2 * cutoff_wavelength)
            
        Returns:
            np.ndarray: Low-pass filtered data
        """
        if transition_width is None:
            transition_width = 0.2 * cutoff_wavelength
            
        ny, nx = data.shape
        kx, ky = self.create_wavenumber_grids(nx, ny)
        k = np.sqrt(kx**2 + ky**2)
        
        # Calculate filter boundaries
        k_inner = self.wavelength_to_wavenumber(cutoff_wavelength + transition_width/2)
        k_outer = self.wavelength_to_wavenumber(cutoff_wavelength - transition_width/2)
        
        # Create filter with cosine rolloff
        filter_response = np.ones_like(k)
        
        # Stop band (k >= k_outer)
        filter_response[k >= k_outer] = 0.0
        
        # Transition band (k_inner < k < k_outer)
        transition_mask = (k > k_inner) & (k < k_outer)
        filter_response[transition_mask] = 0.5 * (
            1 + np.cos(np.pi * (k[transition_mask] - k_inner) / (k_outer - k_inner))
        )
        
        # Pass band (k <= k_inner) remains 1
        
        return self._apply_filter(data, filter_response)
    
    def band_pass_filter(self, data: np.ndarray, low_cutoff: float, high_cutoff: float,
                        low_transition: Optional[float] = None,
                        high_transition: Optional[float] = None) -> np.ndarray:
        """
        Apply band-pass filter with cosine rolloff.
        
        Parameters:
            data (np.ndarray): Input data
            low_cutoff (float): Low cutoff wavelength (removes longer wavelengths)
            high_cutoff (float): High cutoff wavelength (removes shorter wavelengths)
            low_transition (float): Low transition width
            high_transition (float): High transition width
            
        Returns:
            np.ndarray: Band-pass filtered data
        """
        if low_transition is None:
            low_transition = 0.2 * low_cutoff
        if high_transition is None:
            high_transition = 0.2 * high_cutoff
            
        ny, nx = data.shape
        kx, ky = self.create_wavenumber_grids(nx, ny)
        k = np.sqrt(kx**2 + ky**2)
        
        # High-pass component (removes low frequencies)
        k_h_low = self.wavelength_to_wavenumber(low_cutoff + low_transition/2)
        k_h_high = self.wavelength_to_wavenumber(low_cutoff - low_transition/2)
        
        # Low-pass component (removes high frequencies)
        k_l_inner = self.wavelength_to_wavenumber(high_cutoff + high_transition/2)
        k_l_outer = self.wavelength_to_wavenumber(high_cutoff - high_transition/2)
        
        # Create high-pass component
        hp_filter = np.zeros_like(k)
        hp_filter[k >= k_h_high] = 1.0
        transition_mask = (k > k_h_low) & (k < k_h_high)
        hp_filter[transition_mask] = 0.5 * (
            1 - np.cos(np.pi * (k[transition_mask] - k_h_low) / (k_h_high - k_h_low))
        )
        
        # Create low-pass component
        lp_filter = np.ones_like(k)
        lp_filter[k >= k_l_outer] = 0.0
        transition_mask = (k > k_l_inner) & (k < k_l_outer)
        lp_filter[transition_mask] = 0.5 * (
            1 + np.cos(np.pi * (k[transition_mask] - k_l_inner) / (k_l_outer - k_l_inner))
        )
        
        # Combine filters
        filter_response = hp_filter * lp_filter
        
        return self._apply_filter(data, filter_response)
    
    def butterworth_high_pass(self, data: np.ndarray, cutoff_wavelength: float,
                             order: int = 2) -> np.ndarray:
        """
        Apply Butterworth high-pass filter.
        
        Parameters:
            data (np.ndarray): Input data
            cutoff_wavelength (float): Cutoff wavelength
            order (int): Filter order (higher = sharper transition)
            
        Returns:
            np.ndarray: Butterworth filtered data
        """
        ny, nx = data.shape
        kx, ky = self.create_wavenumber_grids(nx, ny)
        k = np.sqrt(kx**2 + ky**2)
        
        k_c = self.wavelength_to_wavenumber(cutoff_wavelength)
        
        # Butterworth filter response
        filter_response = 1.0 / (1.0 + (k_c / (k + 1e-10))**(2*order))
        
        return self._apply_filter(data, filter_response)
    
    def directional_cosine_filter(self, data: np.ndarray, center_direction: float,
                                 power: float = 2.0) -> np.ndarray:
        """
        Apply directional cosine filter.
        
        Parameters:
            data (np.ndarray): Input data
            center_direction (float): Center direction in degrees
            power (float): Power of cosine function (higher = more directional)
            
        Returns:
            np.ndarray: Directionally filtered data
        """
        ny, nx = data.shape
        kx, ky = self.create_wavenumber_grids(nx, ny)
        
        # Calculate angle of each frequency component
        theta = np.arctan2(ky, kx)
        theta_c = np.radians(center_direction)
        
        # Directional cosine filter
        filter_response = np.abs(np.cos(theta - theta_c))**power
        
        return self._apply_filter(data, filter_response)
    
    def directional_band_pass(self, data: np.ndarray, center_direction: float,
                             angular_width: float, high_pass_wavelength: float,
                             power: float = 2.0) -> np.ndarray:
        """
        Apply combined directional and high-pass filter.
        
        Parameters:
            data (np.ndarray): Input data
            center_direction (float): Center direction in degrees
            angular_width (float): Angular width in degrees
            high_pass_wavelength (float): High-pass cutoff wavelength
            power (float): Directional filter power
            
        Returns:
            np.ndarray: Directionally band-pass filtered data
        """
        # Apply high-pass filter first
        hp_filtered = self.butterworth_high_pass(data, high_pass_wavelength)
        
        # Apply directional filter
        dir_filtered = self.directional_cosine_filter(hp_filtered, center_direction, power)
        
        # Remove from original with scaling
        scaling_factor = 0.5  # Adjust to control feature suppression
        return data - scaling_factor * dir_filtered
    
    def automatic_gain_control(self, data: np.ndarray, window_size: int = 51) -> np.ndarray:
        """
        Apply Automatic Gain Control (AGC).
        
        Parameters:
            data (np.ndarray): Input data
            window_size (int): Window size for RMS calculation
            
        Returns:
            np.ndarray: AGC normalized data
        """
        from scipy.ndimage import uniform_filter
        
        # Calculate RMS in sliding window
        data_squared = data**2
        rms = np.sqrt(uniform_filter(data_squared, size=window_size))
        
        # Avoid division by zero
        rms = np.where(rms == 0, 1e-10, rms)
        
        return data / rms
    
    def _apply_filter(self, data: np.ndarray, filter_response: np.ndarray) -> np.ndarray:
        """
        Apply frequency domain filter to data.
        
        Parameters:
            data (np.ndarray): Input data
            filter_response (np.ndarray): Filter response in frequency domain
            
        Returns:
            np.ndarray: Filtered data
        """
        # Handle NaN values
        mask = ~np.isnan(data)
        if not np.any(mask):
            return data
            
        # Fill NaN values with mean
        data_filled = np.where(mask, data, np.nanmean(data))
        
        # Apply FFT
        data_fft = np.fft.fft2(data_filled)
        
        # Apply filter
        filtered_fft = data_fft * filter_response
        
        # Inverse FFT
        filtered_data = np.real(np.fft.ifft2(filtered_fft))
        
        # Restore NaN values
        filtered_data = np.where(mask, filtered_data, np.nan)
        
        return filtered_data
    
    def radial_power_spectrum(self, data: np.ndarray, 
                             num_bins: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate radially averaged power spectrum.
        
        Parameters:
            data (np.ndarray): Input data
            num_bins (int): Number of radial bins
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Wavenumbers and power spectrum
        """
        ny, nx = data.shape
        kx, ky = self.create_wavenumber_grids(nx, ny)
        k = np.sqrt(kx**2 + ky**2)
        
        # Calculate power spectrum
        data_filled = np.where(np.isnan(data), np.nanmean(data), data)
        fft_data = np.fft.fft2(data_filled)
        power = np.abs(fft_data)**2
        
        # Create radial bins
        k_max = np.max(k)
        k_bins = np.linspace(0, k_max, num_bins + 1)
        k_centers = (k_bins[:-1] + k_bins[1:]) / 2
        
        # Calculate radially averaged power
        power_radial = np.zeros(num_bins)
        for i in range(num_bins):
            mask = (k >= k_bins[i]) & (k < k_bins[i + 1])
            if np.sum(mask) > 0:
                power_radial[i] = np.mean(power[mask])
        
        return k_centers, power_radial