"""
Vectorized Minimum Curvature Interpolation
==========================================

High-performance minimum curvature interpolation adapted from grid_interpolator.py.
Optimized for geophysical survey data with Numba JIT compilation support.
"""

import numpy as np
import logging
from typing import Tuple, Optional
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    NUMBA_AVAILABLE = False

logger = logging.getLogger(__name__)


@jit(nopython=True, parallel=True, cache=True)
def minimum_curvature_iteration_jit(grid_z, data_mask, data_values, omega, min_allowed, max_allowed):
    """
    JIT-compiled core iteration for minimum curvature interpolation.
    
    This function performs one iteration of the minimum curvature algorithm
    using a 5-point stencil and successive over-relaxation.
    
    Args:
        grid_z: Current grid values
        data_mask: Boolean mask indicating data constraint points
        data_values: Values at data constraint points
        omega: Relaxation factor
        min_allowed, max_allowed: Value clamps for stability
    
    Returns:
        max_change: Maximum change in grid values during this iteration
    """
    ny, nx = grid_z.shape
    max_change = 0.0
    
    # Create a copy of the grid for the new values
    new_grid_z = grid_z.copy()
    
    # Update interior points using 5-point stencil
    for i in range(1, ny-1):
        for j in range(1, nx-1):
            if not data_mask[i, j]:  # Don't modify data constraint points
                old_value = grid_z[i, j]
                
                # 5-point stencil for Laplacian
                neighbors = (grid_z[i+1, j] + grid_z[i-1, j] + 
                           grid_z[i, j+1] + grid_z[i, j-1]) / 4.0
                
                # Successive over-relaxation
                new_value = old_value + omega * (neighbors - old_value)
                
                # Clamp values for stability
                if new_value < min_allowed:
                    new_value = min_allowed
                elif new_value > max_allowed:
                    new_value = max_allowed
                
                new_grid_z[i, j] = new_value
                
                # Track maximum change
                change = abs(new_value - old_value)
                if change > max_change:
                    max_change = change
    
    # Apply boundary conditions
    if ny >= 3:
        # Top and bottom boundaries (zero second derivative)
        for j in range(nx):
            new_grid_z[0, j] = 2*new_grid_z[1, j] - new_grid_z[2, j]
            new_grid_z[ny-1, j] = 2*new_grid_z[ny-2, j] - new_grid_z[ny-3, j]
    
    if nx >= 3:
        # Left and right boundaries (zero second derivative)
        for i in range(ny):
            new_grid_z[i, 0] = 2*new_grid_z[i, 1] - new_grid_z[i, 2]
            new_grid_z[i, nx-1] = 2*new_grid_z[i, nx-2] - new_grid_z[i, nx-3]
    
    # Re-apply data constraints
    for i in range(ny):
        for j in range(nx):
            if data_mask[i, j]:
                new_grid_z[i, j] = data_values[i, j]
    
    # Copy results back
    for i in range(ny):
        for j in range(nx):
            grid_z[i, j] = new_grid_z[i, j]
    
    return max_change


def minimum_curvature_interpolation(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                                  grid_x: np.ndarray, grid_y: np.ndarray,
                                  max_iterations: int = 100,
                                  tolerance: float = 1e-3,
                                  omega: float = 0.5,
                                  max_points: Optional[int] = None) -> np.ndarray:
    """
    Vectorized minimum curvature interpolation for smooth surfaces.
    
    This method creates surfaces with minimal curvature while honoring data points.
    It's particularly effective for geophysical survey data and produces smooth,
    geologically reasonable interpolations.
    
    Parameters:
        x, y, z (np.ndarray): Input coordinate and value arrays
        grid_x, grid_y (np.ndarray): Output grid coordinate meshgrids
        max_iterations (int): Maximum number of iterations
        tolerance (float): Convergence tolerance
        omega (float): Relaxation factor for stability (0.5 is conservative)
        max_points (int): Maximum points to use (for performance)
        
    Returns:
        np.ndarray: Interpolated grid values
    """
    try:
        # Clean input data
        valid_mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
        if not np.any(valid_mask):
            logger.warning("No valid data points for minimum curvature")
            return np.full(grid_x.shape, np.nan)
        
        x_clean = x[valid_mask]
        y_clean = y[valid_mask]
        z_clean = z[valid_mask]
        
        # Downsample large datasets for performance (only if max_points is specified)
        if max_points is not None and len(x_clean) > max_points:
            logger.info(f"Downsampling from {len(x_clean)} to {max_points} points for minimum curvature performance")
            x_clean, y_clean, z_clean = _downsample_data(x_clean, y_clean, z_clean, max_points)
        
        logger.info(f"Minimum curvature interpolation with {len(x_clean)} data points...")
        
        # Create the output grid
        ny, nx = grid_x.shape
        grid_z = np.zeros((ny, nx))
        
        # Get grid spacing
        dx = (grid_x.max() - grid_x.min()) / (nx - 1)
        dy = (grid_y.max() - grid_y.min()) / (ny - 1)
        
        # Initialize grid with bilinear interpolation for better starting point
        grid_z_initial = griddata((x_clean, y_clean), z_clean, (grid_x, grid_y), 
                                 method='linear', fill_value=np.nan)
        
        # Handle NaN values by using nearest neighbor interpolation
        nan_mask = np.isnan(grid_z_initial)
        if np.any(nan_mask):
            grid_z_nearest = griddata((x_clean, y_clean), z_clean, (grid_x, grid_y), 
                                     method='nearest', fill_value=np.mean(z_clean))
            grid_z_initial[nan_mask] = grid_z_nearest[nan_mask]
        
        grid_z = grid_z_initial.copy()
        
        # Create data constraint mask - points that should remain fixed
        data_mask = np.zeros((ny, nx), dtype=bool)
        data_values = np.zeros((ny, nx))
        
        # Find grid points closest to data points with improved mapping
        for xi, yi, zi in zip(x_clean, y_clean, z_clean):
            # Find closest grid point with bounds checking
            i = int(np.clip(round((yi - grid_y.min()) / dy), 0, ny-1))
            j = int(np.clip(round((xi - grid_x.min()) / dx), 0, nx-1))
            
            # Use distance-weighted averaging if multiple data points map to same grid point
            if data_mask[i, j]:
                # Average with existing value
                data_values[i, j] = (data_values[i, j] + zi) / 2
            else:
                data_mask[i, j] = True
                data_values[i, j] = zi
        
        # Apply data constraints
        grid_z[data_mask] = data_values[data_mask]
        
        logger.info(f"Fixed {np.sum(data_mask)} grid points to data values")
        
        # High-performance minimum curvature iteration with adaptive algorithm selection
        # Pre-compute constants for stability
        data_range = np.ptp(z_clean)
        data_mean = np.mean(z_clean)
        min_allowed = data_mean - 3 * data_range
        max_allowed = data_mean + 3 * data_range
        
        # Choose algorithm based on grid size and available optimizations
        use_jit = NUMBA_AVAILABLE and (nx * ny > 1000)  # Use JIT for larger grids
        
        if use_jit:
            logger.info(f"Using JIT-compiled minimum curvature (Numba available, grid size: {nx}x{ny})")
        else:
            logger.info(f"Using vectorized minimum curvature (grid size: {nx}x{ny})")
        
        # Convergence tracking
        converged = False
        
        for iteration in range(max_iterations):
            if use_jit:
                # Use JIT-compiled version for better performance on large grids
                max_change = minimum_curvature_iteration_jit(
                    grid_z, data_mask, data_values, omega, min_allowed, max_allowed)
            else:
                # Use vectorized version for smaller grids or when Numba unavailable
                max_change = _minimum_curvature_iteration_vectorized(
                    grid_z, data_mask, data_values, omega, min_allowed, max_allowed)
            
            # Early detection of instability
            if max_change > data_range:
                logger.warning(f"Large change detected at iteration {iteration+1}, stopping early")
                converged = True
            
            # Check convergence
            if max_change < tolerance or converged:
                logger.info(f"Minimum curvature converged after {iteration+1} iterations (max change: {max_change:.2e})")
                break
                
            # Progress logging for iterations
            if iteration % 10 == 0 and iteration > 0:
                logger.info(f"Iteration {iteration+1}/{max_iterations}, max change: {max_change:.2e}")
                
            # Additional stability check
            if max_change > 1e10:
                logger.error(f"Numerical instability detected at iteration {iteration+1}, stopping")
                break
        else:
            logger.info(f"Minimum curvature reached maximum iterations ({max_iterations})")
        
        # Optional post-processing: very light smoothing while preserving data points
        grid_z_smooth = gaussian_filter(grid_z, sigma=0.3, mode='reflect')
        
        # Restore exact data values
        grid_z[data_mask] = data_values[data_mask]
        
        # Blend smoothed and original for non-data points (very conservative)
        blend_factor = 0.1  # Only 10% smoothing for interpolated points
        non_data_mask = ~data_mask
        grid_z[non_data_mask] = ((1 - blend_factor) * grid_z[non_data_mask] + 
                                blend_factor * grid_z_smooth[non_data_mask])
        
        logger.info("Minimum curvature interpolation completed successfully")
        return grid_z
        
    except Exception as e:
        logger.error(f"Error in minimum curvature interpolation: {e} - falling back to linear")
        return griddata((x, y), z, (grid_x, grid_y), method='linear', fill_value=np.nan)


def _minimum_curvature_iteration_vectorized(grid_z, data_mask, data_values, omega, min_allowed, max_allowed):
    """
    Vectorized implementation of minimum curvature iteration for when JIT is not available.
    """
    grid_z_old = grid_z.copy()
    
    # Vectorized computation for interior points
    interior_mask = np.zeros_like(data_mask)
    interior_mask[1:-1, 1:-1] = True
    update_mask = interior_mask & ~data_mask
    
    if np.any(update_mask):
        # Vectorized 5-point stencil computation
        neighbors = np.zeros_like(grid_z)
        neighbors[1:-1, 1:-1] = (grid_z[2:, 1:-1] +     # i+1, j
                               grid_z[:-2, 1:-1] +     # i-1, j  
                               grid_z[1:-1, 2:] +      # i, j+1
                               grid_z[1:-1, :-2]) / 4.0 # i, j-1
        
        # Vectorized relaxation update
        new_values = grid_z + omega * (neighbors - grid_z)
        new_values = np.clip(new_values, min_allowed, max_allowed)
        grid_z[update_mask] = new_values[update_mask]
    
    # Vectorized boundary conditions
    ny, nx = grid_z.shape
    if ny >= 3:
        grid_z[0, :] = 2*grid_z[1, :] - grid_z[2, :]
        grid_z[-1, :] = 2*grid_z[-2, :] - grid_z[-3, :]
    if nx >= 3:
        grid_z[:, 0] = 2*grid_z[:, 1] - grid_z[:, 2]
        grid_z[:, -1] = 2*grid_z[:, -2] - grid_z[:, -3]
    
    # Re-apply data constraints
    grid_z[data_mask] = data_values[data_mask]
    
    # Compute maximum change
    changes = np.abs(grid_z - grid_z_old)
    max_change = np.max(changes[update_mask]) if np.any(update_mask) else 0.0
    
    return max_change


def _downsample_data(x: np.ndarray, y: np.ndarray, z: np.ndarray, max_points: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Downsample data for minimum curvature performance while preserving spatial distribution.
    """
    if len(x) <= max_points:
        return x, y, z
        
    # Grid-based downsampling to preserve spatial distribution
    n_bins = int(np.sqrt(max_points))
    
    # Create bins
    x_bins = np.linspace(x.min(), x.max(), n_bins + 1)
    y_bins = np.linspace(y.min(), y.max(), n_bins + 1)
    
    # Digitize points into bins
    x_indices = np.clip(np.digitize(x, x_bins) - 1, 0, n_bins - 1)
    y_indices = np.clip(np.digitize(y, y_bins) - 1, 0, n_bins - 1)
    
    # Sample points from each bin
    sampled_indices = []
    points_per_bin = max(1, max_points // (n_bins * n_bins))
    
    for i in range(n_bins):
        for j in range(n_bins):
            bin_mask = (x_indices == i) & (y_indices == j)
            bin_indices = np.where(bin_mask)[0]
            
            if len(bin_indices) > 0:
                n_sample = min(len(bin_indices), points_per_bin)
                if len(bin_indices) <= n_sample:
                    sampled_indices.extend(bin_indices)
                else:
                    # Prefer extreme values within bin
                    bin_z = z[bin_indices]
                    z_sorted_idx = np.argsort(bin_z)
                    selected = []
                    selected.extend(bin_indices[z_sorted_idx[:n_sample//2]])  # Lowest
                    selected.extend(bin_indices[z_sorted_idx[-(n_sample-n_sample//2):]])  # Highest
                    sampled_indices.extend(selected)
    
    # Convert to array and limit to max_points
    sampled_indices = np.array(sampled_indices)
    if len(sampled_indices) > max_points:
        sampled_indices = np.random.choice(sampled_indices, max_points, replace=False)
    
    logger.info(f"Downsampled from {len(x)} to {len(sampled_indices)} points")
    
    return x[sampled_indices], y[sampled_indices], z[sampled_indices]