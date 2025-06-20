"""
Gradient-Based Filters
======================

Vectorized gradient calculations for geophysical analysis.
Includes Total Horizontal Gradient, Analytic Signal, and Tilt Angle.
"""

import numpy as np
from typing import Tuple, Optional
from scipy import ndimage

try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    NUMBA_AVAILABLE = False


class GradientFilters:
    """
    Gradient-based filtering operations for potential field data.
    
    Implements vectorized calculations of various gradient-based transformations
    commonly used in geophysical analysis.
    """
    
    def __init__(self, dx: float, dy: float):
        """
        Initialize gradient filters.
        
        Parameters:
            dx (float): Grid spacing in x-direction
            dy (float): Grid spacing in y-direction
        """
        self.dx = dx
        self.dy = dy
    
    def calculate_derivatives(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate first derivatives in x, y, and z directions.
        
        Parameters:
            data (np.ndarray): Input potential field data
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: dx, dy, dz derivatives
        """
        # Spatial derivatives using numpy gradient
        dy_data, dx_data = np.gradient(data, self.dy, self.dx)
        
        # Vertical derivative using FFT (more accurate)
        dz_data = self._vertical_derivative_fft(data)
        
        return dx_data, dy_data, dz_data
    
    def _vertical_derivative_fft(self, data: np.ndarray) -> np.ndarray:
        """
        Calculate vertical derivative using FFT method.
        
        Parameters:
            data (np.ndarray): Input data
            
        Returns:
            np.ndarray: Vertical derivative
        """
        ny, nx = data.shape
        
        # Create wavenumber grids
        kx = np.fft.fftfreq(nx, self.dx) * 2 * np.pi
        ky = np.fft.fftfreq(ny, self.dy) * 2 * np.pi
        kx_grid, ky_grid = np.meshgrid(kx, ky, indexing='ij')
        k = np.sqrt(kx_grid**2 + ky_grid**2)
        
        # Handle NaN values
        data_filled = np.where(np.isnan(data), np.nanmean(data), data)
        
        # Apply FFT
        data_fft = np.fft.fft2(data_filled)
        
        # Vertical derivative filter (multiply by k)
        dz_fft = data_fft * k
        
        # Inverse FFT
        dz_data = np.real(np.fft.ifft2(dz_fft))
        
        # Restore NaN mask
        mask = ~np.isnan(data)
        dz_data = np.where(mask, dz_data, np.nan)
        
        return dz_data
    
    def directional_derivative(self, data: np.ndarray, angle: float) -> np.ndarray:
        """
        Calculate directional derivative at specified angle.
        
        Parameters:
            data (np.ndarray): Input data
            angle (float): Angle in degrees (0 = East, 90 = North)
            
        Returns:
            np.ndarray: Directional derivative
        """
        dx_data, dy_data, _ = self.calculate_derivatives(data)
        
        # Convert angle to radians
        angle_rad = np.radians(angle)
        
        # Calculate directional derivative
        dir_deriv = dx_data * np.cos(angle_rad) + dy_data * np.sin(angle_rad)
        
        return dir_deriv
    
    def total_horizontal_gradient(self, data: np.ndarray) -> np.ndarray:
        """
        Calculate Total Horizontal Gradient (THG).
        
        THG = sqrt((∂f/∂x)² + (∂f/∂y)²)
        
        Parameters:
            data (np.ndarray): Input potential field data
            
        Returns:
            np.ndarray: Total horizontal gradient
        """
        dx_data, dy_data, _ = self.calculate_derivatives(data)
        
        # Calculate THG
        thg = np.sqrt(dx_data**2 + dy_data**2)
        
        return thg
    
    def analytic_signal(self, data: np.ndarray) -> np.ndarray:
        """
        Calculate Analytic Signal amplitude.
        
        AS = sqrt((∂f/∂x)² + (∂f/∂y)² + (∂f/∂z)²)
        
        Parameters:
            data (np.ndarray): Input potential field data
            
        Returns:
            np.ndarray: Analytic signal amplitude
        """
        dx_data, dy_data, dz_data = self.calculate_derivatives(data)
        
        # Calculate analytic signal
        analytic_sig = np.sqrt(dx_data**2 + dy_data**2 + dz_data**2)
        
        return analytic_sig
    
    def tilt_angle(self, data: np.ndarray) -> np.ndarray:
        """
        Calculate Tilt Angle.
        
        TA = arctan(∂f/∂z / sqrt((∂f/∂x)² + (∂f/∂y)²))
        
        Parameters:
            data (np.ndarray): Input potential field data
            
        Returns:
            np.ndarray: Tilt angle in radians
        """
        dx_data, dy_data, dz_data = self.calculate_derivatives(data)
        
        # Calculate horizontal gradient magnitude
        horizontal_grad = np.sqrt(dx_data**2 + dy_data**2)
        
        # Avoid division by zero
        horizontal_grad = np.where(horizontal_grad == 0, 1e-10, horizontal_grad)
        
        # Calculate tilt angle
        tilt = np.arctan(dz_data / horizontal_grad)
        
        return tilt
    
    def tilt_angle_degrees(self, data: np.ndarray) -> np.ndarray:
        """
        Calculate Tilt Angle in degrees.
        
        Parameters:
            data (np.ndarray): Input potential field data
            
        Returns:
            np.ndarray: Tilt angle in degrees
        """
        tilt_rad = self.tilt_angle(data)
        return np.degrees(tilt_rad)
    
    def horizontal_gradient_magnitude(self, data: np.ndarray) -> np.ndarray:
        """
        Calculate horizontal gradient magnitude (same as THG).
        
        Parameters:
            data (np.ndarray): Input data
            
        Returns:
            np.ndarray: Horizontal gradient magnitude
        """
        return self.total_horizontal_gradient(data)
    
    def gradient_enhanced_tilt_angle(self, data: np.ndarray) -> np.ndarray:
        """
        Calculate gradient-enhanced tilt angle for better edge detection.
        
        GETA = THG * TiltAngle
        
        Parameters:
            data (np.ndarray): Input potential field data
            
        Returns:
            np.ndarray: Gradient-enhanced tilt angle
        """
        thg = self.total_horizontal_gradient(data)
        tilt = self.tilt_angle(data)
        
        return thg * tilt
    
    def theta_map(self, data: np.ndarray) -> np.ndarray:
        """
        Calculate theta map (normalized tilt angle).
        
        Theta = arctan(THG / |∂f/∂z|)
        
        Parameters:
            data (np.ndarray): Input data
            
        Returns:
            np.ndarray: Theta map
        """
        dx_data, dy_data, dz_data = self.calculate_derivatives(data)
        
        thg = np.sqrt(dx_data**2 + dy_data**2)
        abs_dz = np.abs(dz_data)
        
        # Avoid division by zero
        abs_dz = np.where(abs_dz == 0, 1e-10, abs_dz)
        
        theta = np.arctan(thg / abs_dz)
        
        return theta
    
    def enhanced_horizontal_gradient(self, data: np.ndarray, 
                                   enhancement_factor: float = 1.0) -> np.ndarray:
        """
        Calculate enhanced horizontal gradient with optional enhancement.
        
        Parameters:
            data (np.ndarray): Input data
            enhancement_factor (float): Enhancement factor for gradient
            
        Returns:
            np.ndarray: Enhanced horizontal gradient
        """
        thg = self.total_horizontal_gradient(data)
        
        if enhancement_factor != 1.0:
            # Apply enhancement (could be logarithmic, power, etc.)
            thg = thg ** enhancement_factor
        
        return thg
    
    def local_phase(self, data: np.ndarray) -> np.ndarray:
        """
        Calculate local phase of the analytic signal.
        
        Parameters:
            data (np.ndarray): Input data
            
        Returns:
            np.ndarray: Local phase in radians
        """
        dx_data, dy_data, dz_data = self.calculate_derivatives(data)
        
        # Calculate horizontal derivatives
        horizontal_magnitude = np.sqrt(dx_data**2 + dy_data**2)
        
        # Calculate local phase
        phase = np.arctan2(horizontal_magnitude, dz_data)
        
        return phase
    
    def edge_detection_combined(self, data: np.ndarray) -> np.ndarray:
        """
        Combined edge detection using multiple gradient-based methods.
        
        Parameters:
            data (np.ndarray): Input data
            
        Returns:
            np.ndarray: Combined edge detection result
        """
        # Calculate multiple gradient measures
        thg = self.total_horizontal_gradient(data)
        analytic_sig = self.analytic_signal(data)
        tilt = np.abs(self.tilt_angle(data))
        
        # Normalize each measure
        thg_norm = (thg - np.nanmin(thg)) / (np.nanmax(thg) - np.nanmin(thg))
        as_norm = (analytic_sig - np.nanmin(analytic_sig)) / (np.nanmax(analytic_sig) - np.nanmin(analytic_sig))
        tilt_norm = (tilt - np.nanmin(tilt)) / (np.nanmax(tilt) - np.nanmin(tilt))
        
        # Combine with equal weights (can be adjusted)
        combined = (thg_norm + as_norm + tilt_norm) / 3.0
        
        return combined
    
    def curvature_analysis(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate curvature measures from the data.
        
        Parameters:
            data (np.ndarray): Input data
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: 
                Profile curvature, plan curvature, mean curvature
        """
        # Calculate first derivatives
        dx_data, dy_data, _ = self.calculate_derivatives(data)
        
        # Calculate second derivatives
        dxx = np.gradient(dx_data, self.dx, axis=1)
        dyy = np.gradient(dy_data, self.dy, axis=0)
        dxy = np.gradient(dx_data, self.dy, axis=0)
        
        # Calculate gradient magnitude
        grad_mag = np.sqrt(dx_data**2 + dy_data**2)
        grad_mag = np.where(grad_mag == 0, 1e-10, grad_mag)
        
        # Profile curvature (curvature in the direction of maximum slope)
        profile_curv = (dx_data**2 * dxx + 2*dx_data*dy_data*dxy + dy_data**2 * dyy) / (grad_mag**3)
        
        # Plan curvature (curvature perpendicular to maximum slope)
        plan_curv = (dy_data**2 * dxx - 2*dx_data*dy_data*dxy + dx_data**2 * dyy) / (grad_mag**2)
        
        # Mean curvature
        mean_curv = (profile_curv + plan_curv) / 2.0
        
        return profile_curv, plan_curv, mean_curv