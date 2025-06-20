#!/usr/bin/env python3
"""
Magnetic Anomaly Detection Script
=================================

Advanced multi-scale analysis for detecting small magnetic anomalies and features
that may be missed in standard interpolation. Implements residual analysis,
frequency domain processing, and statistical anomaly detection.

This script takes the same CSV input directory as grid_interpolator.py and performs:

1. MULTI-SCALE MINIMUM CURVATURE INTERPOLATION:
   - Ultra-smooth regional field (500 points, low omega)
   - Smooth regional field (2k points, moderate omega)  
   - Standard field (15k points, normal omega)
   - High-resolution field (25k points, high omega)

2. RESIDUAL ANALYSIS:
   - Large-scale residuals (standard - ultra_smooth)
   - Medium-scale residuals (high_res - standard)
   - Small-scale residuals (high_res - smooth)
   - Regional trend analysis (smooth - ultra_smooth)

3. ANOMALY DETECTION:
   - Statistical outlier detection (2σ, 2.5σ, 3σ thresholds)
   - Gradient-based edge detection
   - Optional frequency domain analysis with high-pass filtering

4. VISUALIZATION & REPORTING:
   - Interactive Folium map with toggleable layers
   - Residual field plots with statistics
   - Comprehensive anomaly summary plots
   - Detailed CSV report with anomaly coordinates and confidence scores

Usage: python magnetic_anomaly_detector.py
Configure INPUT_DIRECTORY and other parameters in the configuration section below.
"""

import pandas as pd
import numpy as np
from scipy import fft
from scipy.interpolate import griddata
import folium
from pathlib import Path
import logging
import base64
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Try to import optional libraries - will be imported when needed
NUMBA_AVAILABLE = False
try:
    import importlib.util
    if importlib.util.find_spec('numba') is not None:
        from numba import jit
        NUMBA_AVAILABLE = True
except ImportError:
    pass

if not NUMBA_AVAILABLE:
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

SKLEARN_AVAILABLE = False
try:
    import importlib.util
    if importlib.util.find_spec('sklearn') is not None:
        SKLEARN_AVAILABLE = True
except ImportError:
    pass

# =============================================
# CONFIGURATION
# =============================================

# Input directory containing the CSV files (modify this path)
INPUT_DIRECTORY = "/Users/aleksandergarbuz/Documents/SINTEF/data/OldFlights/interpolation"

# Field selection - specify which field to analyze
TARGET_FIELD = "R1 [nT]"  # Options: "R1 [nT]", "Btotal1 [nT]", "Total [nT]", "B_total [nT]", or "auto"

# Grid settings
GRID_RESOLUTION = 300  # Number of points per axis for interpolation grid

# Multi-scale minimum curvature interpolation settings
# Each scale uses different parameters to create different levels of smoothing:
# - Lower omega = more smoothing (good for regional trends)
# - Higher omega = less smoothing (preserves details, good for anomalies)
SCALE_CONFIGURATIONS = {
    'ultra_smooth': {
        'max_points': 500,         # Heavy downsampling for ultra-smooth regional field
        'max_iterations': 20,      # Fewer iterations for speed
        'tolerance': 1e-2,         # Looser tolerance for regional trends
        'omega': 0.3,              # Low relaxation factor = heavy smoothing
        'description': 'Ultra-smooth regional field'
    },
    'smooth': {
        'max_points': 2000,        # Moderate downsampling
        'max_iterations': 30,      # Moderate iterations
        'tolerance': 1e-2,         # Moderate tolerance
        'omega': 0.4,              # Moderate smoothing
        'description': 'Smooth regional field'
    },
    'standard': {
        'max_points': 15000,       # Standard point density
        'max_iterations': 50,      # Standard iterations
        'tolerance': 1e-3,         # Standard tolerance
        'omega': 0.5,              # Balanced smoothing
        'description': 'Standard interpolation'
    },
    'high_res': {
        'max_points': 25000,       # High point density for detail preservation
        'max_iterations': 75,      # More iterations for precision
        'tolerance': 1e-3,         # Tight tolerance for accuracy
        'omega': 0.6,              # Higher relaxation = less smoothing, more detail
        'description': 'High-resolution field'
    }
}

# Anomaly detection settings
ANOMALY_DETECTION_METHODS = ['residual', 'statistical', 'gradient']
STATISTICAL_THRESHOLDS = [2.0, 2.5, 3.0]  # Sigma levels for outlier detection
GRADIENT_SENSITIVITY = 1.5  # Multiplier for gradient-based detection
MIN_ANOMALY_SIZE = 3  # Minimum number of connected pixels for anomaly

# Frequency domain settings
ENABLE_FREQUENCY_ANALYSIS = True
HIGH_PASS_CUTOFF = 0.1  # Fraction of Nyquist frequency
BAND_PASS_RANGES = [(0.1, 0.3), (0.3, 0.6)]  # Low and high band-pass ranges

# Boundary masking settings (improved for better region exclusion)
ENABLE_BOUNDARY_MASKING = True  # Set to True to enable boundary masking
BOUNDARY_METHOD = 'alpha_shape'  # Options: 'convex_hull', 'alpha_shape', 'distance', 'data_density'
BOUNDARY_BUFFER_DISTANCE = None  # Optional buffer distance in coordinate units (None for no buffer)
ALPHA_SHAPE_ALPHA = 0.1  # Alpha parameter for alpha shapes (lower = tighter boundary, higher = fewer warnings)
DATA_DENSITY_THRESHOLD = 0.1  # Minimum data density for inclusion in boundary (fraction)

# Output settings
OUTPUT_SUBFOLDER = "anomaly_analysis"  # Subfolder name within input directory
OUTPUT_HTML_FILENAME = "magnetic_anomaly_analysis.html"
GENERATE_RESIDUAL_PLOTS = True
GENERATE_FREQUENCY_PLOTS = True
GENERATE_ANOMALY_REPORT = True

# Visualization settings
MAPBOX_TOKEN = "pk.eyJ1IjoiYXRnMjE3IiwiYSI6ImNtYzBnY2kwOTAxbWwybHM3NmN0bnRlaWcifQ.B8hh4dBszYXxlj-O0KGqkg"

# =============================================
# SETUP LOGGING
# =============================================

def setup_logging():
    """Setup logging for the script"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def setup_output_directory(input_directory):
    """Create output directory as subfolder of input directory"""
    logger = logging.getLogger(__name__)
    
    input_path = Path(input_directory)
    output_path = input_path / OUTPUT_SUBFOLDER
    
    # Create output directory if it doesn't exist
    output_path.mkdir(exist_ok=True)
    
    logger.info(f"Output directory: {output_path}")
    return output_path

# =============================================
# DATA LOADING AND VALIDATION
# =============================================

def load_and_combine_csv_files(directory):
    """Load all CSV files from the specified directory and combine them into a single DataFrame"""
    logger = logging.getLogger(__name__)
    
    csv_files = list(Path(directory).glob("*.csv"))
    if not csv_files:
        logger.error(f"No CSV files found in directory: {directory}")
        return None
    
    logger.info(f"Found {len(csv_files)} CSV files in {directory}")
    
    # Combine all CSVs into one DataFrame
    df_list = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            df['Source_File'] = file.stem  # Add source filename as a column
            df_list.append(df)
            logger.info(f"Loaded {len(df)} points from {file.name}")
        except Exception as e:
            logger.warning(f"Failed to load {file.name}: {e}")
            continue
    
    if not df_list:
        logger.error("No CSV files could be loaded successfully")
        return None
    
    combined_df = pd.concat(df_list, ignore_index=True)
    logger.info(f"Combined dataset: {len(combined_df)} total points from {len(df_list)} files")
    
    return combined_df

def validate_data(df):
    """Validate that the DataFrame has required columns"""
    logger = logging.getLogger(__name__)
    
    required_columns = [
        'Latitude [Decimal Degrees]', 
        'Longitude [Decimal Degrees]'
    ]
    
    # Check for magnetic field columns
    magnetic_columns = ['R1 [nT]', 'Btotal1 [nT]', 'Total [nT]', 'B_total [nT]']
    has_magnetic_field = any(col in df.columns for col in magnetic_columns)
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return False
    
    if not has_magnetic_field:
        logger.error(f"No magnetic field columns found. Available columns: {list(df.columns)}")
        return False
    
    return True

def get_magnetic_field_column(df):
    """Automatically detect the magnetic field column"""
    if TARGET_FIELD == "auto":
        # Try to find the best magnetic field column
        possible_columns = ['R1 [nT]', 'Btotal1 [nT]', 'Total [nT]', 'B_total [nT]']
        for col in possible_columns:
            if col in df.columns:
                return col
        return None
    else:
        return TARGET_FIELD if TARGET_FIELD in df.columns else None

# =============================================
# MINIMUM CURVATURE INTERPOLATION FUNCTIONS
# =============================================

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

# =============================================
# MULTI-SCALE INTERPOLATION FUNCTIONS
# =============================================

def enhanced_downsampling(x, y, z, max_points):
    """Enhanced spatial downsampling for minimum curvature interpolation"""
    if len(x) <= max_points:
        return x, y, z
    
    # Create spatial grid for stratified sampling
    n_grid = int(np.sqrt(max_points))
    
    x_bins = np.linspace(x.min(), x.max(), n_grid + 1)
    y_bins = np.linspace(y.min(), y.max(), n_grid + 1)
    
    selected_indices = []
    
    for i in range(n_grid):
        for j in range(n_grid):
            # Find points in this grid cell
            mask = ((x >= x_bins[i]) & (x < x_bins[i+1]) & 
                   (y >= y_bins[j]) & (y < y_bins[j+1]))
            
            cell_indices = np.where(mask)[0]
            
            if len(cell_indices) > 0:
                # Take one representative point from each cell
                selected_indices.append(cell_indices[0])
    
    selected_indices = np.array(selected_indices)
    
    # If we still have too many points, randomly subsample
    if len(selected_indices) > max_points:
        selected_indices = np.random.choice(selected_indices, max_points, replace=False)
    
    return x[selected_indices], y[selected_indices], z[selected_indices]

def multi_scale_minimum_curvature_interpolation(x, y, z, grid_x, grid_y, scale_config):
    """Perform minimum curvature interpolation with specific scale configuration"""
    logger = logging.getLogger(__name__)
    
    try:
        # Clean input data
        valid_mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
        if not np.any(valid_mask):
            logger.warning(f"No valid data points for {scale_config['description']}")
            return np.full(grid_x.shape, np.nan)
        
        x_clean = x[valid_mask]
        y_clean = y[valid_mask]
        z_clean = z[valid_mask]
        
        # Apply scale-specific downsampling
        max_points = scale_config['max_points']
        
        if len(x_clean) > max_points:
            x_clean, y_clean, z_clean = enhanced_downsampling(x_clean, y_clean, z_clean, max_points)
        
        logger.info(f"{scale_config['description']}: {len(x_clean)} points, "
                   f"iterations={scale_config['max_iterations']}, omega={scale_config['omega']}")
        
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
        
        # Find grid points closest to data points
        for xi, yi, zi in zip(x_clean, y_clean, z_clean):
            # Find closest grid point with bounds checking
            i = int(np.clip(round((yi - grid_y.min()) / dy), 0, ny-1))
            j = int(np.clip(round((xi - grid_x.min()) / dx), 0, nx-1))
            
            # Use distance-weighted averaging if multiple data points map to same grid point
            if data_mask[i, j]:
                data_values[i, j] = (data_values[i, j] + zi) / 2
            else:
                data_mask[i, j] = True
                data_values[i, j] = zi
        
        # Apply data constraints
        grid_z[data_mask] = data_values[data_mask]
        
        # Minimum curvature iteration with scale-specific parameters
        data_range = np.ptp(z_clean)
        data_mean = np.mean(z_clean)
        min_allowed = data_mean - 3 * data_range
        max_allowed = data_mean + 3 * data_range
        omega = scale_config['omega']
        max_iterations = scale_config['max_iterations']
        tolerance = scale_config['tolerance']
        
        # Choose algorithm based on grid size and available optimizations
        use_jit = NUMBA_AVAILABLE and (nx * ny > 1000)
        
        # Convergence tracking
        for iteration in range(max_iterations):
            if use_jit:
                # Use JIT-compiled version for better performance
                max_change = minimum_curvature_iteration_jit(
                    grid_z, data_mask, data_values, omega, min_allowed, max_allowed)
            else:
                # Use vectorized version for smaller grids
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
            
            # Check convergence
            if max_change < tolerance:
                break
                
            # Early detection of instability
            if max_change > data_range:
                logger.warning(f"Large change detected at iteration {iteration+1}, stopping early")
                break
        
        logger.info(f"Completed {scale_config['description']} interpolation ({iteration+1} iterations)")
        return grid_z
        
    except Exception as e:
        logger.error(f"Error in {scale_config['description']} interpolation: {e}")
        # Fallback to linear interpolation
        return griddata((x, y), z, (grid_x, grid_y), method='linear', fill_value=np.nan)

# =============================================
# BOUNDARY MASKING FUNCTIONS
# =============================================

def create_boundary_mask(x, y, grid_x, grid_y, method='convex_hull', buffer_distance=None):
    """
    Create a boundary mask to prevent interpolation outside data coverage.
    
    Args:
        x, y: Data point coordinates
        grid_x, grid_y: Grid meshgrid coordinates
        method: 'convex_hull' or 'alpha_shape'
        buffer_distance: Optional buffer distance for expanding the boundary
    
    Returns:
        mask: Boolean array where True indicates points inside the data boundary
    """
    logger = logging.getLogger(__name__)
    
    try:
        from scipy.spatial import ConvexHull
        from scipy.spatial.distance import cdist
        
        # Get data points as array
        data_points = np.column_stack([x, y])
        
        # Get grid points as array
        grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        
        if method == 'convex_hull':
            # Create convex hull
            hull = ConvexHull(data_points)
            
            # Check which grid points are inside the convex hull
            from matplotlib.path import Path
            hull_path = Path(data_points[hull.vertices])
            mask_1d = hull_path.contains_points(grid_points)
            
        elif method == 'alpha_shape':
            # Alpha shape implementation (more complex boundary)
            try:
                # Try to use alphashape if available
                import alphashape
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='Singular matrix')
                    alpha_shape = alphashape.alphashape(data_points, alpha=ALPHA_SHAPE_ALPHA)
                
                if hasattr(alpha_shape, 'contains'):
                    # For shapely geometry
                    from shapely.geometry import Point
                    mask_1d = np.array([alpha_shape.contains(Point(p)) for p in grid_points])
                else:
                    # Fallback to convex hull
                    logger.warning("Alpha shape geometry not supported, falling back to convex hull")
                    return create_boundary_mask(x, y, grid_x, grid_y, 'convex_hull', buffer_distance)
                    
            except ImportError:
                logger.warning("alphashape not available, falling back to convex hull")
                return create_boundary_mask(x, y, grid_x, grid_y, 'convex_hull', buffer_distance)
        
        elif method == 'data_density':
            # Data density-based masking (excludes low-density regions)
            try:
                from scipy.spatial import cKDTree
                
                # Build KD-tree for efficient nearest neighbor queries
                tree = cKDTree(data_points)
                
                # For each grid point, find distance to nearest data points
                k_neighbors = min(10, len(data_points))  # Use up to 10 nearest neighbors
                distances, _ = tree.query(grid_points, k=k_neighbors)
                
                # Calculate average distance to k nearest neighbors
                if k_neighbors == 1:
                    avg_distances = distances
                else:
                    avg_distances = np.mean(distances, axis=1)
                
                # Calculate data density threshold based on inter-point distances
                data_distances = cdist(data_points, data_points)
                # Remove diagonal (self-distances)
                np.fill_diagonal(data_distances, np.inf)
                min_data_distances = np.min(data_distances, axis=1)
                median_data_spacing = np.median(min_data_distances)
                
                # Include points within reasonable distance of data
                density_threshold = median_data_spacing * (1.0 / DATA_DENSITY_THRESHOLD)
                mask_1d = avg_distances <= density_threshold
                
                logger.info(f"Data density masking: threshold={density_threshold:.4f}, "
                           f"median spacing={median_data_spacing:.4f}")
                
            except ImportError:
                logger.warning("scipy.spatial not available, falling back to convex hull")
                return create_boundary_mask(x, y, grid_x, grid_y, 'convex_hull', buffer_distance)
        
        else:
            # Distance-based masking as fallback
            logger.warning(f"Unknown boundary method {method}, using distance-based masking")
            
            # Find maximum distance between any two data points
            max_distance = np.max(cdist(data_points, data_points))
            threshold_distance = max_distance * 0.3  # Conservative threshold
            
            # For each grid point, find distance to nearest data point
            distances = cdist(grid_points, data_points)
            min_distances = np.min(distances, axis=1)
            mask_1d = min_distances <= threshold_distance
        
        # Apply buffer distance if specified
        if buffer_distance is not None and buffer_distance > 0:
            # Expand the boundary by buffer_distance
            distances_to_boundary = cdist(grid_points, data_points)
            min_distances_to_data = np.min(distances_to_boundary, axis=1)
            
            # Include points within buffer_distance of the original boundary
            buffer_mask = min_distances_to_data <= buffer_distance
            mask_1d = mask_1d | buffer_mask
        
        # Reshape mask to grid shape
        mask = mask_1d.reshape(grid_x.shape)
        
        inside_points = np.sum(mask)
        total_points = mask.size
        coverage_percent = (inside_points / total_points) * 100
        
        logger.info(f"Boundary mask created using {method}: {inside_points:,}/{total_points:,} "
                   f"grid points ({coverage_percent:.1f}%) inside data boundary")
        
        return mask
        
    except Exception as e:
        logger.warning(f"Failed to create boundary mask: {e}, using full grid")
        # Return mask that includes all points
        return np.ones_like(grid_x, dtype=bool)

def apply_boundary_mask(grid_field, mask):
    """
    Apply boundary mask to interpolated field, setting outside points to NaN.
    
    Args:
        grid_field: Interpolated field values
        mask: Boolean mask (True = inside boundary, False = outside)
    
    Returns:
        masked_field: Field with NaN values outside the boundary
    """
    masked_field = grid_field.copy()
    masked_field[~mask] = np.nan
    
    logger = logging.getLogger(__name__)
    total_points = grid_field.size
    masked_points = np.sum(~mask)
    valid_points = np.sum(~np.isnan(masked_field))
    
    logger.info(f"Applied boundary mask: {masked_points:,} points masked out, "
               f"{valid_points:,} valid interpolated points remaining")
    
    return masked_field

# =============================================
# RESIDUAL ANALYSIS FUNCTIONS
# =============================================

def compute_residual_fields(scale_grids):
    """Compute residual fields between different interpolation scales"""
    logger = logging.getLogger(__name__)
    
    residuals = {}
    
    # Large-scale residuals (standard - ultra_smooth)
    if 'standard' in scale_grids and 'ultra_smooth' in scale_grids:
        residuals['large_scale'] = scale_grids['standard'] - scale_grids['ultra_smooth']
        logger.info("Computed large-scale residuals (standard - ultra_smooth)")
    
    # Medium-scale residuals (high_res - standard)
    if 'high_res' in scale_grids and 'standard' in scale_grids:
        residuals['medium_scale'] = scale_grids['high_res'] - scale_grids['standard']
        logger.info("Computed medium-scale residuals (high_res - standard)")
    
    # Small-scale residuals (high_res - smooth)
    if 'high_res' in scale_grids and 'smooth' in scale_grids:
        residuals['small_scale'] = scale_grids['high_res'] - scale_grids['smooth']
        logger.info("Computed small-scale residuals (high_res - smooth)")
    
    # Regional trend (smooth - ultra_smooth)
    if 'smooth' in scale_grids and 'ultra_smooth' in scale_grids:
        residuals['regional_trend'] = scale_grids['smooth'] - scale_grids['ultra_smooth']
        logger.info("Computed regional trend residuals (smooth - ultra_smooth)")
    
    return residuals

def analyze_residual_statistics(residuals):
    """Compute comprehensive statistics for residual fields"""
    logger = logging.getLogger(__name__)
    
    stats = {}
    
    for residual_name, residual_field in residuals.items():
        valid_data = residual_field[~np.isnan(residual_field)]
        
        if len(valid_data) == 0:
            continue
        
        field_stats = {
            'mean': np.mean(valid_data),
            'std': np.std(valid_data),
            'min': np.min(valid_data),
            'max': np.max(valid_data),
            'range': np.ptp(valid_data),
            'percentiles': {
                'p5': np.percentile(valid_data, 5),
                'p25': np.percentile(valid_data, 25),
                'p50': np.percentile(valid_data, 50),
                'p75': np.percentile(valid_data, 75),
                'p95': np.percentile(valid_data, 95)
            },
            'valid_points': len(valid_data),
            'total_points': residual_field.size
        }
        
        stats[residual_name] = field_stats
        
        logger.info(f"Residual {residual_name}: mean={field_stats['mean']:.2f}, "
                   f"std={field_stats['std']:.2f}, range={field_stats['range']:.2f}")
    
    return stats

# =============================================
# ANOMALY DETECTION FUNCTIONS
# =============================================

def statistical_anomaly_detection(field, thresholds=STATISTICAL_THRESHOLDS):
    """Detect statistical anomalies using multiple sigma thresholds"""
    logger = logging.getLogger(__name__)
    
    valid_data = field[~np.isnan(field)]
    if len(valid_data) == 0:
        return {}
    
    mean_val = np.mean(valid_data)
    std_val = np.std(valid_data)
    
    anomalies = {}
    
    for threshold in thresholds:
        # Positive anomalies (above threshold)
        pos_anomaly_mask = field > (mean_val + threshold * std_val)
        # Negative anomalies (below threshold)
        neg_anomaly_mask = field < (mean_val - threshold * std_val)
        # Combined anomalies
        combined_mask = pos_anomaly_mask | neg_anomaly_mask
        
        anomalies[f'sigma_{threshold}'] = {
            'positive_mask': pos_anomaly_mask,
            'negative_mask': neg_anomaly_mask,
            'combined_mask': combined_mask,
            'positive_count': np.sum(pos_anomaly_mask),
            'negative_count': np.sum(neg_anomaly_mask),
            'total_count': np.sum(combined_mask),
            'threshold_value': threshold * std_val
        }
        
        logger.info(f"Sigma {threshold}: {np.sum(combined_mask)} anomaly points "
                   f"({np.sum(pos_anomaly_mask)} positive, {np.sum(neg_anomaly_mask)} negative)")
    
    return anomalies

def gradient_anomaly_detection(field, sensitivity=GRADIENT_SENSITIVITY):
    """Detect anomalies based on spatial gradients"""
    logger = logging.getLogger(__name__)
    
    try:
        # Compute gradients
        grad_y, grad_x = np.gradient(field)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Remove NaN values for threshold calculation
        valid_gradients = gradient_magnitude[~np.isnan(gradient_magnitude)]
        if len(valid_gradients) == 0:
            return None
        
        # Calculate threshold
        mean_grad = np.mean(valid_gradients)
        std_grad = np.std(valid_gradients)
        threshold = mean_grad + sensitivity * std_grad
        
        # Create anomaly mask
        gradient_anomalies = gradient_magnitude > threshold
        
        logger.info(f"Gradient anomalies: {np.sum(gradient_anomalies)} points above threshold "
                   f"(sensitivity={sensitivity})")
        
        return {
            'gradient_magnitude': gradient_magnitude,
            'gradient_x': grad_x,
            'gradient_y': grad_y,
            'anomaly_mask': gradient_anomalies,
            'threshold': threshold,
            'mean_gradient': mean_grad,
            'std_gradient': std_grad
        }
        
    except Exception as e:
        logger.error(f"Error in gradient anomaly detection: {e}")
        return None

# =============================================
# FREQUENCY DOMAIN ANALYSIS FUNCTIONS
# =============================================

def analyze_power_spectral_density(psd, freq_magnitude, logger):
    """
    Analyze power spectral density characteristics and provide geological interpretation
    
    Args:
        psd: Power spectral density array
        freq_magnitude: Frequency magnitude array
        logger: Logger instance
    
    Returns:
        dict: PSD analysis results with statistics and interpretation
    """
    try:
        # Remove DC component (zero frequency)
        psd_no_dc = psd.copy()
        psd_no_dc[freq_magnitude == 0] = 0
        
        # Calculate radial PSD profile
        ny, nx = psd.shape
        center_y, center_x = ny // 2, nx // 2
        
        # Create radial frequency bins
        max_radius = min(center_y, center_x)
        radial_freqs = np.linspace(0, 0.5, max_radius)
        radial_psd = np.zeros(len(radial_freqs) - 1)
        
        for i in range(len(radial_freqs) - 1):
            freq_min, freq_max = radial_freqs[i], radial_freqs[i + 1]
            mask = (freq_magnitude >= freq_min) & (freq_magnitude < freq_max)
            if np.any(mask):
                radial_psd[i] = np.mean(psd_no_dc[mask])
        
        # Calculate frequency domain statistics
        total_power = np.sum(psd_no_dc)
        
        # Define frequency bands for geological interpretation
        freq_bands = {
            'regional': (0.0, 0.1),      # Large-scale geological trends
            'intermediate': (0.1, 0.25), # Medium-scale structures
            'local': (0.25, 0.4),       # Local anomalies
            'noise': (0.4, 0.5)         # High-frequency noise
        }
        
        band_power = {}
        band_percentage = {}
        
        for band_name, (f_min, f_max) in freq_bands.items():
            mask = (freq_magnitude >= f_min) & (freq_magnitude < f_max)
            power = np.sum(psd_no_dc[mask])
            band_power[band_name] = power
            band_percentage[band_name] = (power / total_power) * 100 if total_power > 0 else 0
        
        # Find dominant frequencies
        psd_1d = psd_no_dc.flatten()
        freq_1d = freq_magnitude.flatten()
        
        # Get top 5 frequencies by power
        top_indices = np.argsort(psd_1d)[-5:][::-1]
        dominant_freqs = freq_1d[top_indices]
        dominant_powers = psd_1d[top_indices]
        
        # Calculate spectral slope (power law exponent)
        # Fit power law to radial PSD: PSD ~ f^(-beta)
        valid_mask = (radial_psd > 0) & (radial_freqs[1:] > 0.05)  # Avoid low frequencies
        if np.sum(valid_mask) > 3:
            log_freq = np.log10(radial_freqs[1:][valid_mask])
            log_psd = np.log10(radial_psd[valid_mask])
            
            # Linear fit in log space
            coeffs = np.polyfit(log_freq, log_psd, 1)
            spectral_slope = -coeffs[0]  # Negative because PSD typically decreases with frequency
        else:
            spectral_slope = np.nan
        
        # Geological interpretation
        interpretation = interpret_psd_characteristics(band_percentage, spectral_slope, logger)
        
        logger.info("Power Spectral Density Analysis:")
        logger.info(f"  Total power: {total_power:.2e}")
        logger.info(f"  Spectral slope (β): {spectral_slope:.2f}")
        logger.info("  Power distribution by frequency band:")
        for band_name, percentage in band_percentage.items():
            logger.info(f"    {band_name.capitalize()}: {percentage:.1f}%")
        
        return {
            'total_power': total_power,
            'band_power': band_power,
            'band_percentage': band_percentage,
            'dominant_frequencies': dominant_freqs,
            'dominant_powers': dominant_powers,
            'radial_frequencies': radial_freqs[1:],
            'radial_psd': radial_psd,
            'spectral_slope': spectral_slope,
            'interpretation': interpretation
        }
        
    except Exception as e:
        logger.error(f"Error in PSD analysis: {e}")
        return {}

def interpret_psd_characteristics(band_percentage, spectral_slope, logger):
    """
    Provide geological interpretation of PSD characteristics
    
    Args:
        band_percentage: Power percentage in each frequency band
        spectral_slope: Spectral slope (power law exponent)
        logger: Logger instance
    
    Returns:
        str: Geological interpretation text
    """
    interpretation = []
    
    # Interpret frequency band distribution
    regional_power = band_percentage.get('regional', 0)
    intermediate_power = band_percentage.get('intermediate', 0)
    local_power = band_percentage.get('local', 0)
    noise_power = band_percentage.get('noise', 0)
    
    if regional_power > 50:
        interpretation.append("• Dominated by regional geological trends (deep structures, basement topography)")
    elif local_power > 30:
        interpretation.append("• Rich in local anomalies (potential ore bodies, intrusions, shallow structures)")
    elif intermediate_power > 40:
        interpretation.append("• Dominated by intermediate-scale features (geological units, major contacts)")
    
    if noise_power > 20:
        interpretation.append("• High noise content - consider additional filtering or data quality review")
    
    # Interpret spectral slope
    if not np.isnan(spectral_slope):
        if spectral_slope < 1.5:
            interpretation.append("• Shallow spectral slope suggests complex, multi-scale geological structures")
        elif spectral_slope > 3.0:
            interpretation.append("• Steep spectral slope indicates dominant large-scale features with limited small-scale detail")
        else:
            interpretation.append("• Moderate spectral slope indicates balanced geological complexity across scales")
    
    # Signal quality assessment
    signal_to_noise = (regional_power + intermediate_power + local_power) / max(noise_power, 1)
    if signal_to_noise > 4:
        interpretation.append("• Excellent signal-to-noise ratio - high quality magnetic data")
    elif signal_to_noise > 2:
        interpretation.append("• Good signal-to-noise ratio - reliable for anomaly detection")
    else:
        interpretation.append("• Poor signal-to-noise ratio - consider data filtering or quality control")
    
    logger.info("Geological Interpretation:")
    for line in interpretation:
        logger.info(f"  {line}")
    
    return "\n".join(interpretation)

def frequency_domain_analysis(field):
    """Perform 2D FFT analysis and filtering of magnetic field"""
    logger = logging.getLogger(__name__)
    
    try:
        # Handle NaN values by interpolating
        valid_mask = ~np.isnan(field)
        if np.sum(valid_mask) < field.size * 0.5:
            logger.warning("Too many NaN values for reliable frequency analysis")
            return None
        
        # Interpolate NaN values for FFT
        field_clean = field.copy()
        if np.any(~valid_mask):
            ny, nx = field.shape
            y_coords, x_coords = np.mgrid[0:ny, 0:nx]
            valid_points = np.column_stack([y_coords[valid_mask], x_coords[valid_mask]])
            valid_values = field[valid_mask]
            
            nan_points = np.column_stack([y_coords[~valid_mask], x_coords[~valid_mask]])
            if len(nan_points) > 0:
                interp_values = griddata(valid_points, valid_values, nan_points, method='linear')
                # Handle any remaining NaN values from interpolation
                nan_mask = np.isnan(interp_values)
                if np.any(nan_mask):
                    # Use nearest neighbor for remaining NaN values
                    interp_values[nan_mask] = griddata(valid_points, valid_values, 
                                                     nan_points[nan_mask], method='nearest')
                field_clean[~valid_mask] = interp_values
        
        # Final check for any remaining NaN values
        if np.any(np.isnan(field_clean)):
            # Replace any remaining NaN with field mean
            field_mean = np.nanmean(field_clean)
            field_clean[np.isnan(field_clean)] = field_mean
        
        # Perform 2D FFT
        fft_field = fft.fft2(field_clean)
        fft_magnitude = np.abs(fft_field)
        fft_phase = np.angle(fft_field)
        
        # Create frequency grids
        ny, nx = field.shape
        freq_y = fft.fftfreq(ny)
        freq_x = fft.fftfreq(nx)
        freq_magnitude = np.sqrt(freq_x[np.newaxis, :]**2 + freq_y[:, np.newaxis]**2)
        
        # High-pass filtering for small-scale features
        high_pass_filter = freq_magnitude > HIGH_PASS_CUTOFF
        fft_high_pass = fft_field * high_pass_filter
        high_pass_field = np.real(fft.ifft2(fft_high_pass))
        
        # Band-pass filtering
        band_pass_fields = {}
        for i, (low_freq, high_freq) in enumerate(BAND_PASS_RANGES):
            band_filter = (freq_magnitude >= low_freq) & (freq_magnitude <= high_freq)
            fft_band_pass = fft_field * band_filter
            band_pass_field = np.real(fft.ifft2(fft_band_pass))
            band_pass_fields[f'band_{i+1}'] = band_pass_field
        
        # Power spectral density
        psd = np.abs(fft_field)**2
        
        logger.info("Completed frequency domain analysis")
        
        # Analyze PSD characteristics
        psd_analysis = analyze_power_spectral_density(psd, freq_magnitude, logger)
        
        return {
            'fft_magnitude': fft_magnitude,
            'fft_phase': fft_phase,
            'freq_magnitude': freq_magnitude,
            'high_pass_field': high_pass_field,
            'band_pass_fields': band_pass_fields,
            'power_spectral_density': psd,
            'frequency_grids': (freq_x, freq_y),
            'psd_analysis': psd_analysis
        }
        
    except Exception as e:
        logger.error(f"Error in frequency domain analysis: {e}")
        return None

# =============================================
# VISUALIZATION FUNCTIONS
# =============================================

def generate_psd_analysis_plot(frequency_results, output_dir, logger):
    """Generate dedicated PSD analysis plot with detailed statistics"""
    try:
        psd_analysis = frequency_results['psd_analysis']
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Radial PSD profile
        ax = axes[0, 0]
        radial_freqs = psd_analysis['radial_frequencies']
        radial_psd = psd_analysis['radial_psd']
        
        ax.loglog(radial_freqs, radial_psd, 'b-', linewidth=2, label='Radial PSD')
        
        # Add spectral slope line if available
        if not np.isnan(psd_analysis['spectral_slope']):
            slope = psd_analysis['spectral_slope']
            # Fit line for visualization
            freq_fit = radial_freqs[radial_freqs > 0.05]
            if len(freq_fit) > 0:
                psd_fit = radial_psd[0] * (freq_fit / freq_fit[0]) ** (-slope)
                ax.loglog(freq_fit, psd_fit, 'r--', linewidth=2, 
                         label=f'Power law fit (β={slope:.2f})')
        
        ax.set_xlabel('Spatial Frequency')
        ax.set_ylabel('Power Spectral Density')
        ax.set_title('Radial Power Spectral Density Profile')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 2. Frequency band power distribution
        ax = axes[0, 1]
        band_names = list(psd_analysis['band_percentage'].keys())
        band_values = list(psd_analysis['band_percentage'].values())
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        bars = ax.bar(band_names, band_values, color=colors, alpha=0.7)
        ax.set_ylabel('Power Percentage (%)')
        ax.set_title('Power Distribution by Frequency Band')
        ax.set_ylim(0, max(band_values) * 1.1)
        
        # Add value labels on bars
        for bar, value in zip(bars, band_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{value:.1f}%', ha='center', va='bottom')
        
        # 3. 2D PSD visualization
        ax = axes[1, 0]
        psd = frequency_results['power_spectral_density']
        psd_log = np.log10(psd + 1e-10)
        psd_shifted = np.fft.fftshift(psd_log)
        
        im = ax.imshow(psd_shifted, cmap='viridis', origin='lower',
                      vmin=np.percentile(psd_shifted, 1), 
                      vmax=np.percentile(psd_shifted, 99))
        ax.set_title('2D Power Spectral Density (Log Scale)')
        ax.set_xlabel('Frequency X')
        ax.set_ylabel('Frequency Y')
        plt.colorbar(im, ax=ax, label='Log(PSD)')
        
        # 4. Interpretation text
        ax = axes[1, 1]
        ax.axis('off')
        
        # Create interpretation text
        interpretation_text = "POWER SPECTRAL DENSITY ANALYSIS\n\n"
        interpretation_text += f"Total Power: {psd_analysis['total_power']:.2e}\n"
        interpretation_text += f"Spectral Slope (β): {psd_analysis['spectral_slope']:.2f}\n\n"
        
        interpretation_text += "FREQUENCY BAND DISTRIBUTION:\n"
        for band_name, percentage in psd_analysis['band_percentage'].items():
            interpretation_text += f"• {band_name.capitalize()}: {percentage:.1f}%\n"
        
        interpretation_text += "\nGEOLOGICAL INTERPRETATION:\n"
        interpretation_text += psd_analysis['interpretation']
        
        ax.text(0.05, 0.95, interpretation_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        output_file = output_dir / 'power_spectral_density_analysis.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Generated PSD analysis plot: {output_file}")
        
        # Also save PSD analysis to CSV
        save_psd_analysis_csv(psd_analysis, output_dir, logger)
        
    except Exception as e:
        logger.error(f"Error generating PSD analysis plot: {e}")

def save_psd_analysis_csv(psd_analysis, output_dir, logger):
    """Save PSD analysis results to CSV file"""
    try:
        import pandas as pd
        
        # Create summary data
        summary_data = {
            'Metric': ['Total Power', 'Spectral Slope'],
            'Value': [psd_analysis['total_power'], psd_analysis['spectral_slope']],
            'Unit': ['Power Units', 'Dimensionless']
        }
        
        # Add frequency band data
        for band_name, percentage in psd_analysis['band_percentage'].items():
            summary_data['Metric'].append(f'{band_name.capitalize()} Band Power')
            summary_data['Value'].append(percentage)
            summary_data['Unit'].append('%')
        
        summary_df = pd.DataFrame(summary_data)
        
        # Create radial PSD data
        radial_data = {
            'Frequency': psd_analysis['radial_frequencies'],
            'PSD': psd_analysis['radial_psd']
        }
        radial_df = pd.DataFrame(radial_data)
        
        # Save to CSV files
        summary_file = output_dir / 'psd_analysis_summary.csv'
        radial_file = output_dir / 'psd_radial_profile.csv'
        
        summary_df.to_csv(summary_file, index=False)
        radial_df.to_csv(radial_file, index=False)
        
        logger.info(f"Generated PSD analysis CSV files: {summary_file}, {radial_file}")
        
    except Exception as e:
        logger.error(f"Error saving PSD analysis CSV: {e}")

def create_residual_plots(residuals, residual_stats, grid_lat, grid_lon, output_dir):
    """Create visualization plots for residual fields"""
    logger = logging.getLogger(__name__)
    
    if not GENERATE_RESIDUAL_PLOTS:
        return
    
    try:
        n_residuals = len(residuals)
        if n_residuals == 0:
            return
        
        # Create subplots
        cols = min(2, n_residuals)
        rows = (n_residuals + cols - 1) // cols
        
        _, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        if n_residuals == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, (residual_name, residual_field) in enumerate(residuals.items()):
            ax = axes[i]
            
            # Plot residual field
            valid_data = residual_field[~np.isnan(residual_field)]
            if len(valid_data) > 0:
                vmin, vmax = np.percentile(valid_data, [5, 95])
                
                im = ax.contourf(grid_lon, grid_lat, residual_field, 
                               levels=50, cmap='viridis', vmin=vmin, vmax=vmax)
                
                # Add contour lines
                ax.contour(grid_lon, grid_lat, residual_field, 
                          levels=10, colors='black', alpha=0.3, linewidths=0.5)
                
                # Colorbar
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('Residual [nT]')
                
                # Statistics text
                stats = residual_stats.get(residual_name, {})
                stats_text = f"Mean: {stats.get('mean', 0):.2f}\n"
                stats_text += f"Std: {stats.get('std', 0):.2f}\n"
                stats_text += f"Range: {stats.get('range', 0):.2f}"
                
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_xlabel('Longitude [degrees]')
            ax.set_ylabel('Latitude [degrees]')
            ax.set_title(f'Residual Field: {residual_name.replace("_", " ").title()}')
            ax.set_aspect('equal')
        
        # Hide unused subplots
        for i in range(n_residuals, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        output_file = output_dir / 'anomaly_residual_fields.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Generated residual field plots: {output_file}")
        
    except Exception as e:
        logger.error(f"Error creating residual plots: {e}")

def create_anomaly_summary_plot(anomaly_results, scale_grids, grid_lat, grid_lon, output_dir):
    """Create a comprehensive anomaly summary visualization"""
    logger = logging.getLogger(__name__)
    
    try:
        _, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: High-resolution field
        if 'high_res' in scale_grids:
            field = scale_grids['high_res']
            valid_data = field[~np.isnan(field)]
            if len(valid_data) > 0:
                vmin, vmax = np.percentile(valid_data, [2, 98])
                im1 = axes[0, 0].contourf(grid_lon, grid_lat, field, levels=50, cmap='viridis', vmin=vmin, vmax=vmax)
                plt.colorbar(im1, ax=axes[0, 0])
                axes[0, 0].set_title('High-Resolution Field')
        
        # Plot 2: Statistical anomalies (strongest residual)
        strongest_residual = None
        max_anomaly_count = 0
        
        for key, anomalies in anomaly_results.items():
            if 'statistical' in key and isinstance(anomalies, dict):
                for threshold, data in anomalies.items():
                    if data['total_count'] > max_anomaly_count:
                        max_anomaly_count = data['total_count']
                        strongest_residual = (key, threshold, data)
        
        if strongest_residual:
            key, threshold, data = strongest_residual
            anomaly_field = np.zeros_like(grid_lat)
            anomaly_field[data['positive_mask']] = 1
            anomaly_field[data['negative_mask']] = -1
            
            im2 = axes[0, 1].imshow(anomaly_field, extent=[grid_lon.min(), grid_lon.max(), 
                                                         grid_lat.min(), grid_lat.max()],
                                  cmap='viridis', vmin=-1, vmax=1, origin='lower')
            plt.colorbar(im2, ax=axes[0, 1])
            axes[0, 1].set_title(f'Statistical Anomalies\n{key} - {threshold}')
        
        # Plot 3: Gradient anomalies
        if 'gradient_anomalies' in anomaly_results and anomaly_results['gradient_anomalies']:
            grad_data = anomaly_results['gradient_anomalies']
            gradient_field = grad_data['gradient_magnitude']
            
            valid_grad = gradient_field[~np.isnan(gradient_field)]
            if len(valid_grad) > 0:
                vmax = np.percentile(valid_grad, 95)
                im3 = axes[0, 2].contourf(grid_lon, grid_lat, gradient_field, 
                                        levels=50, cmap='viridis', vmin=0, vmax=vmax)
                
                # Overlay anomaly points
                anomaly_mask = grad_data['anomaly_mask']
                y_anom, x_anom = np.where(anomaly_mask)
                if len(x_anom) > 0:
                    axes[0, 2].scatter(grid_lon.flat[x_anom + y_anom * grid_lon.shape[1]], 
                                     grid_lat.flat[x_anom + y_anom * grid_lat.shape[0]], 
                                     c='cyan', s=1, alpha=0.8)
                
                plt.colorbar(im3, ax=axes[0, 2])
                axes[0, 2].set_title('Gradient Magnitude & Anomalies')
        
        # Plot 4-6: Anomaly statistics histograms
        plot_idx = 3
        for key, anomalies in anomaly_results.items():
            if plot_idx >= 6:
                break
            
            if 'statistical' in key and isinstance(anomalies, dict):
                # Get the field name from key
                field_name = key.replace('_statistical', '').replace('_', ' ').title()
                
                # Collect all anomaly counts
                thresholds = []
                pos_counts = []
                neg_counts = []
                
                for threshold, data in anomalies.items():
                    if 'sigma_' in threshold:
                        sigma_val = float(threshold.replace('sigma_', ''))
                        thresholds.append(sigma_val)
                        pos_counts.append(data['positive_count'])
                        neg_counts.append(data['negative_count'])
                
                if thresholds:
                    ax = axes[plot_idx // 3, plot_idx % 3]
                    x_pos = np.arange(len(thresholds))
                    
                    ax.bar(x_pos - 0.2, pos_counts, 0.4, label='Positive', color='red', alpha=0.7)
                    ax.bar(x_pos + 0.2, neg_counts, 0.4, label='Negative', color='blue', alpha=0.7)
                    
                    ax.set_xlabel('Sigma Threshold')
                    ax.set_ylabel('Anomaly Count')
                    ax.set_title(f'Anomaly Count - {field_name}')
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels([f'{t:.1f}σ' for t in thresholds])
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    plot_idx += 1
        
        # Hide unused subplots
        for i in range(plot_idx, 6):
            axes[i // 3, i % 3].set_visible(False)
        
        plt.tight_layout()
        output_file = output_dir / 'anomaly_summary_analysis.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Generated anomaly summary plot: {output_file}")
        
    except Exception as e:
        logger.error(f"Error creating anomaly summary plot: {e}")

def create_interactive_anomaly_map(scale_grids, residuals, anomaly_results, grid_lat, grid_lon, latitudes, longitudes, output_dir):
    """Create an interactive Folium map with all anomaly layers"""
    logger = logging.getLogger(__name__)
    
    try:
        # Calculate map center
        center_lat = np.mean(latitudes)
        center_lon = np.mean(longitudes)
        
        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=10,
            tiles=None
        )
        
        # Greyscale topographic base layer
        folium.WmsTileLayer(
            url='https://wms.geonorge.no/skwms1/wms.topo4.graatone',
            name='Kartverket Topo – greyscale',
            layers='topo4graatone_WMS',   # main layer id
            fmt='image/png',
            transparent=False,
            version='1.3.0',
            attr='©️ Kartverket',
            overlay=False,                # this becomes the base map
            control=True
        ).add_to(m)

        # Add base tiles
        # folium.TileLayer('OpenStreetMap').add_to(m)
        # if MAPBOX_TOKEN:
        #     folium.TileLayer(
        #         tiles=f'https://api.mapbox.com/styles/v1/mapbox/satellite-v9/tiles/{{z}}/{{x}}/{{y}}?access_token={MAPBOX_TOKEN}',
        #         attr='Mapbox',
        #         name='Satellite',
        #         overlay=False,
        #         control=True
        #     ).add_to(m)
        
        # Function to create overlay from grid data
        def create_overlay(data, name, opacity=0.7):
            if data is None or np.all(np.isnan(data)):
                return
            
            valid_data = data[~np.isnan(data)]
            if len(valid_data) == 0:
                return
            
            # Normalize data for visualization
            vmin, vmax = np.percentile(valid_data, [5, 95])
            
            # Always use viridis colormap
            cmap = plt.cm.viridis
            
            # Normalize and apply colormap
            norm_data = np.clip((data - vmin) / (vmax - vmin), 0, 1)
            colored_data = cmap(norm_data)
            
            # Convert to PIL Image - FIX ORIENTATION: flip vertically
            img_array = (colored_data * 255).astype(np.uint8)
            img = Image.fromarray(np.flipud(img_array))  # Flip to fix upside-down issue
            
            # Encode as base64
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            
            # Add to map
            bounds = [[grid_lat.min(), grid_lon.min()], [grid_lat.max(), grid_lon.max()]]
            folium.raster_layers.ImageOverlay(
                image=f'data:image/png;base64,{img_str}',
                bounds=bounds,
                opacity=opacity,
                name=name
            ).add_to(m)
        
        # Add scale grids
        for scale_name, scale_data in scale_grids.items():
            create_overlay(scale_data, f'{scale_name.replace("_", " ").title()} Field', 'viridis', 0.6)
        
        # Add residual fields
        for residual_name, residual_data in residuals.items():
            create_overlay(residual_data, f'{residual_name.replace("_", " ").title()} Residual', 'viridis', 0.7)
        
        # Add gradient field if available
        if 'gradient_anomalies' in anomaly_results and anomaly_results['gradient_anomalies']:
            grad_magnitude = anomaly_results['gradient_anomalies']['gradient_magnitude']
            create_overlay(grad_magnitude, 'Gradient Magnitude', 'viridis', 0.6)
        
        # Add toggleable anomaly point layers
        for key, anomalies in anomaly_results.items():
            if 'statistical' in key and isinstance(anomalies, dict):
                # Create a readable field name
                field_name = key.replace('_statistical', '').replace('_', ' ').title()
                
                for threshold, data in anomalies.items():
                    if 'sigma_' in threshold:
                        sigma_value = threshold.replace('sigma_', '')
                        
                        # Create feature groups for positive and negative anomalies
                        pos_group_name = f'{field_name} Positive ({sigma_value}σ)'
                        neg_group_name = f'{field_name} Negative ({sigma_value}σ)'
                        
                        pos_group = folium.FeatureGroup(name=pos_group_name)
                        neg_group = folium.FeatureGroup(name=neg_group_name)
                        
                        # Positive anomalies
                        pos_mask = data['positive_mask']
                        if np.any(pos_mask):
                            y_pos, x_pos = np.where(pos_mask)
                            # Limit to reasonable number of points for performance
                            step = max(1, len(y_pos) // 200)
                            for i in range(0, len(y_pos), step):
                                lat_pos = grid_lat[y_pos[i], x_pos[i]]
                                lon_pos = grid_lon[y_pos[i], x_pos[i]]
                                
                                # Get the actual field value at this point
                                value = scale_grids.get('high_res', np.full_like(grid_lat, np.nan))[y_pos[i], x_pos[i]]
                                
                                folium.CircleMarker(
                                    location=[lat_pos, lon_pos],
                                    radius=4,
                                    popup=f'<b>Positive Anomaly</b><br>'
                                          f'Field: {field_name}<br>'
                                          f'Threshold: {sigma_value}σ<br>'
                                          f'Value: {value:.2f} nT<br>'
                                          f'Location: {lat_pos:.6f}, {lon_pos:.6f}',
                                    color='red',
                                    fillColor='red',
                                    fillOpacity=0.8,
                                    weight=2
                                ).add_to(pos_group)
                        
                        # Negative anomalies
                        neg_mask = data['negative_mask']
                        if np.any(neg_mask):
                            y_neg, x_neg = np.where(neg_mask)
                            # Limit to reasonable number of points for performance
                            step = max(1, len(y_neg) // 200)
                            for i in range(0, len(y_neg), step):
                                lat_neg = grid_lat[y_neg[i], x_neg[i]]
                                lon_neg = grid_lon[y_neg[i], x_neg[i]]
                                
                                # Get the actual field value at this point
                                value = scale_grids.get('high_res', np.full_like(grid_lat, np.nan))[y_neg[i], x_neg[i]]
                                
                                folium.CircleMarker(
                                    location=[lat_neg, lon_neg],
                                    radius=4,
                                    popup=f'<b>Negative Anomaly</b><br>'
                                          f'Field: {field_name}<br>'
                                          f'Threshold: {sigma_value}σ<br>'
                                          f'Value: {value:.2f} nT<br>'
                                          f'Location: {lat_neg:.6f}, {lon_neg:.6f}',
                                    color='blue',
                                    fillColor='blue',
                                    fillOpacity=0.8,
                                    weight=2
                                ).add_to(neg_group)
                        
                        # Add groups to map
                        pos_group.add_to(m)
                        neg_group.add_to(m)
        
        # Add gradient anomalies as toggleable layer
        if 'gradient_anomalies' in anomaly_results and anomaly_results['gradient_anomalies']:
            grad_data = anomaly_results['gradient_anomalies']
            anomaly_mask = grad_data['anomaly_mask']
            
            if np.any(anomaly_mask):
                grad_group = folium.FeatureGroup(name='Gradient Anomalies')
                
                y_grad, x_grad = np.where(anomaly_mask)
                # Limit to reasonable number of points
                step = max(1, len(y_grad) // 100)
                
                for i in range(0, len(y_grad), step):
                    lat_grad = grid_lat[y_grad[i], x_grad[i]]
                    lon_grad = grid_lon[y_grad[i], x_grad[i]]
                    
                    gradient_value = grad_data['gradient_magnitude'][y_grad[i], x_grad[i]]
                    
                    folium.CircleMarker(
                        location=[lat_grad, lon_grad],
                        radius=3,
                        popup=f'<b>Gradient Anomaly</b><br>'
                              f'Gradient: {gradient_value:.3f}<br>'
                              f'Threshold: {grad_data["threshold"]:.3f}<br>'
                              f'Location: {lat_grad:.6f}, {lon_grad:.6f}',
                        color='orange',
                        fillColor='orange',
                        fillOpacity=0.7,
                        weight=2
                    ).add_to(grad_group)
                
                grad_group.add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Add "Uncheck All" button with JavaScript functionality
        uncheck_all_html = '''
        <div style="position: fixed; 
                    top: 10px; left: 10px; width: 120px; height: auto; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:12px; padding: 8px; border-radius: 5px;
                    ">
        <button onclick="uncheckAllLayers()" 
                style="width: 100%; padding: 6px; background-color: #ff6b6b; color: white; 
                       border: none; border-radius: 3px; cursor: pointer; font-size: 11px;">
            📋 Uncheck All Layers
        </button>
        <button onclick="checkAllLayers()" 
                style="width: 100%; padding: 6px; background-color: #51cf66; color: white; 
                       border: none; border-radius: 3px; cursor: pointer; font-size: 11px; margin-top: 3px;">
            ✓ Check All Layers
        </button>
        </div>
        
        <script>
        function uncheckAllLayers() {
            // Wait a bit for the map to be fully loaded
            setTimeout(function() {
                // Find all checkboxes in the layer control
                var checkboxes = document.querySelectorAll('.leaflet-control-layers-overlays input[type="checkbox"]');
                checkboxes.forEach(function(checkbox) {
                    if (checkbox.checked) {
                        checkbox.click(); // Trigger the click event to properly hide layers
                    }
                });
            }, 100);
        }
        
        function checkAllLayers() {
            // Wait a bit for the map to be fully loaded
            setTimeout(function() {
                // Find all checkboxes in the layer control
                var checkboxes = document.querySelectorAll('.leaflet-control-layers-overlays input[type="checkbox"]');
                checkboxes.forEach(function(checkbox) {
                    if (!checkbox.checked) {
                        checkbox.click(); // Trigger the click event to properly show layers
                    }
                });
            }, 100);
        }
        
        // Auto-uncheck all layers when page loads (optional)
        document.addEventListener('DOMContentLoaded', function() {
            setTimeout(function() {
                // Uncomment the line below if you want all layers to start unchecked
                // uncheckAllLayers();
            }, 1000);
        });
        </script>
        '''
        
        # Add the JavaScript to the map
        m.get_root().html.add_child(folium.Element(uncheck_all_html))
        
        # Save map
        output_file = output_dir / OUTPUT_HTML_FILENAME
        m.save(str(output_file))
        logger.info(f"Generated interactive anomaly map: {output_file}")
        
    except Exception as e:
        logger.error(f"Error creating interactive map: {e}")

def generate_anomaly_report(anomaly_results, residual_stats, scale_grids, output_dir):
    """Generate comprehensive anomaly detection report"""
    logger = logging.getLogger(__name__)
    
    if not GENERATE_ANOMALY_REPORT:
        return
    
    try:
        report_data = []
        
        # Summary statistics for each scale
        for scale_name, scale_data in scale_grids.items():
            valid_data = scale_data[~np.isnan(scale_data)]
            if len(valid_data) > 0:
                report_data.append({
                    'Analysis_Type': 'Scale_Statistics',
                    'Scale_Residual': scale_name,
                    'Method': 'interpolation',
                    'Threshold': 'N/A',
                    'Count': len(valid_data),
                    'Mean': np.mean(valid_data),
                    'Std': np.std(valid_data),
                    'Min': np.min(valid_data),
                    'Max': np.max(valid_data),
                    'Range': np.ptp(valid_data)
                })
        
        # Residual statistics
        for residual_name, stats in residual_stats.items():
            report_data.append({
                'Analysis_Type': 'Residual_Statistics',
                'Scale_Residual': residual_name,
                'Method': 'residual_analysis',
                'Threshold': 'N/A',
                'Count': stats['valid_points'],
                'Mean': stats['mean'],
                'Std': stats['std'],
                'Min': stats['min'],
                'Max': stats['max'],
                'Range': stats['range']
            })
        
        # Anomaly detection results
        for key, anomalies in anomaly_results.items():
            if 'statistical' in key and isinstance(anomalies, dict):
                for threshold, data in anomalies.items():
                    report_data.append({
                        'Analysis_Type': 'Statistical_Anomalies',
                        'Scale_Residual': key.replace('_statistical', ''),
                        'Method': 'statistical',
                        'Threshold': threshold,
                        'Count': data['total_count'],
                        'Positive_Count': data['positive_count'],
                        'Negative_Count': data['negative_count'],
                        'Threshold_Value': data['threshold_value']
                    })
        
        # Gradient anomalies
        if 'gradient_anomalies' in anomaly_results and anomaly_results['gradient_anomalies']:
            grad_data = anomaly_results['gradient_anomalies']
            report_data.append({
                'Analysis_Type': 'Gradient_Anomalies',
                'Scale_Residual': 'high_res',
                'Method': 'gradient',
                'Threshold': f'sensitivity_{GRADIENT_SENSITIVITY}',
                'Count': np.sum(grad_data['anomaly_mask']),
                'Threshold_Value': grad_data['threshold'],
                'Mean_Gradient': grad_data['mean_gradient'],
                'Std_Gradient': grad_data['std_gradient']
            })
        
        # Create DataFrame and save
        report_df = pd.DataFrame(report_data)
        output_file = output_dir / 'magnetic_anomaly_report.csv'
        report_df.to_csv(output_file, index=False)
        
        logger.info(f"Generated anomaly detection report: {output_file}")
        logger.info(f"Report contains {len(report_data)} analysis entries")
        
    except Exception as e:
        logger.error(f"Error generating anomaly report: {e}")

if __name__ == "__main__":
    logger = setup_logging()
    logger.info("Starting Magnetic Anomaly Detection Analysis")
    logger.info(f"NumPy version: {np.__version__}")
    logger.info(f"Numba available: {NUMBA_AVAILABLE}")
    logger.info(f"Scikit-learn available: {SKLEARN_AVAILABLE}")
    
    # Setup output directory
    output_dir = setup_output_directory(INPUT_DIRECTORY)
    
    # Load and validate data
    logger.info(f"Loading data from: {INPUT_DIRECTORY}")
    df = load_and_combine_csv_files(INPUT_DIRECTORY)
    
    if df is None:
        logger.error("Failed to load data")
        exit(1)
    
    if not validate_data(df):
        logger.error("Data validation failed")
        exit(1)
    
    # Get magnetic field column
    mag_field_col = get_magnetic_field_column(df)
    if mag_field_col is None:
        logger.error("No suitable magnetic field column found")
        exit(1)
    
    logger.info(f"Using magnetic field column: {mag_field_col}")
    logger.info(f"Dataset contains {len(df)} data points")
    
    # Extract coordinates and field values
    latitudes = df['Latitude [Decimal Degrees]'].values
    longitudes = df['Longitude [Decimal Degrees]'].values
    field_values = df[mag_field_col].values
    source_files = df['Source_File'].values if 'Source_File' in df.columns else None
    
    # Create interpolation grid
    grid_lat = np.linspace(latitudes.min(), latitudes.max(), GRID_RESOLUTION)
    grid_lon = np.linspace(longitudes.min(), longitudes.max(), GRID_RESOLUTION)
    grid_lon_mesh, grid_lat_mesh = np.meshgrid(grid_lon, grid_lat)
    
    logger.info(f"Created interpolation grid: {GRID_RESOLUTION}x{GRID_RESOLUTION}")
    
    # Perform multi-scale interpolation
    logger.info("Starting multi-scale interpolation...")
    scale_grids = {}
    
    for scale_name, scale_config in SCALE_CONFIGURATIONS.items():
        logger.info(f"Processing {scale_config['description']}...")
        scale_grids[scale_name] = multi_scale_minimum_curvature_interpolation(
            longitudes, latitudes, field_values,
            grid_lon_mesh, grid_lat_mesh, scale_config
        )
    
    # Apply boundary masking to all scales if enabled
    if ENABLE_BOUNDARY_MASKING:
        logger.info(f"Applying boundary masking using {BOUNDARY_METHOD} method")
        boundary_mask = create_boundary_mask(
            longitudes, latitudes, grid_lon_mesh, grid_lat_mesh, 
            method=BOUNDARY_METHOD, buffer_distance=BOUNDARY_BUFFER_DISTANCE
        )
        
        # Apply mask to all scale grids
        for scale_name in scale_grids.keys():
            scale_grids[scale_name] = apply_boundary_mask(scale_grids[scale_name], boundary_mask)
    
    # Compute residual fields
    logger.info("Computing residual fields...")
    residuals = compute_residual_fields(scale_grids)
    
    # Analyze residual statistics
    logger.info("Analyzing residual statistics...")
    residual_stats = analyze_residual_statistics(residuals)
    
    # Detect anomalies
    logger.info("Detecting anomalies...")
    anomaly_results = {}
    
    # Statistical anomaly detection on each residual field
    for residual_name, residual_field in residuals.items():
        logger.info(f"Statistical analysis of {residual_name} residuals...")
        anomaly_results[f"{residual_name}_statistical"] = statistical_anomaly_detection(residual_field)
    
    # Gradient anomaly detection on high-resolution field
    if 'high_res' in scale_grids:
        logger.info("Gradient anomaly detection on high-resolution field...")
        anomaly_results['gradient_anomalies'] = gradient_anomaly_detection(scale_grids['high_res'])
    
    # Frequency domain analysis (optional)
    frequency_results = None
    if ENABLE_FREQUENCY_ANALYSIS and 'high_res' in scale_grids:
        logger.info("Performing frequency domain analysis...")
        frequency_results = frequency_domain_analysis(scale_grids['high_res'])
        if frequency_results:
            # Add high-pass filtered field to analysis
            high_pass_field = frequency_results['high_pass_field']
            logger.info("Analyzing high-pass filtered field for anomalies...")
            anomaly_results['high_pass_statistical'] = statistical_anomaly_detection(high_pass_field)
    
    # Generate visualizations
    logger.info("Generating visualizations...")
    
    # Create residual field plots
    if residuals:
        create_residual_plots(residuals, residual_stats, grid_lat_mesh, grid_lon_mesh, output_dir)
    
    # Create comprehensive anomaly summary
    create_anomaly_summary_plot(anomaly_results, scale_grids, grid_lat_mesh, grid_lon_mesh, output_dir)
    
    # Create interactive map
    create_interactive_anomaly_map(
        scale_grids, residuals, anomaly_results, 
        grid_lat_mesh, grid_lon_mesh, latitudes, longitudes, output_dir
    )
    
    # Generate comprehensive report
    generate_anomaly_report(anomaly_results, residual_stats, scale_grids, output_dir)
    
    # Frequency plots (if frequency analysis was performed)
    if GENERATE_FREQUENCY_PLOTS and frequency_results:
        try:
            logger.info("Generating frequency domain plots...")
            
            # Generate PSD analysis plot if available
            if 'psd_analysis' in frequency_results and frequency_results['psd_analysis']:
                generate_psd_analysis_plot(frequency_results, output_dir, logger)
            
            _, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Power spectral density
            psd = frequency_results['power_spectral_density']
            psd_log = np.log10(psd + 1e-10)  # Add small value to avoid log(0)
            psd_shifted = np.fft.fftshift(psd_log)
            
            # Check if PSD has valid range
            psd_min, psd_max = np.min(psd_shifted), np.max(psd_shifted)
            logger.info(f"PSD log range: {psd_min:.2f} to {psd_max:.2f}")
            
            im1 = axes[0, 0].imshow(psd_shifted, cmap='viridis', origin='lower',
                                  vmin=np.percentile(psd_shifted, 1), 
                                  vmax=np.percentile(psd_shifted, 99))
            axes[0, 0].set_title('Log Power Spectral Density')
            axes[0, 0].set_xlabel('Frequency X')
            axes[0, 0].set_ylabel('Frequency Y')
            plt.colorbar(im1, ax=axes[0, 0])
            
            # High-pass filtered field
            high_pass_field = frequency_results['high_pass_field']
            valid_hp = high_pass_field[~np.isnan(high_pass_field)]
            logger.info(f"High-pass field valid points: {len(valid_hp)}/{high_pass_field.size}")
            
            if len(valid_hp) > 0:
                hp_range = np.ptp(valid_hp)
                logger.info(f"High-pass field range: {hp_range:.2f}")
                
                if hp_range > 1e-10:  # Only plot if there's meaningful variation
                    vmin, vmax = np.percentile(valid_hp, [2, 98])
                    im2 = axes[0, 1].contourf(grid_lon_mesh, grid_lat_mesh, high_pass_field, 
                                            levels=50, cmap='viridis', vmin=vmin, vmax=vmax)
                    axes[0, 1].set_title(f'High-Pass Filtered Field (range: {hp_range:.2f})')
                    axes[0, 1].set_xlabel('Longitude')
                    axes[0, 1].set_ylabel('Latitude')
                    plt.colorbar(im2, ax=axes[0, 1])
                else:
                    axes[0, 1].text(0.5, 0.5, 'High-pass field\nhas no variation', 
                                  ha='center', va='center', transform=axes[0, 1].transAxes)
                    axes[0, 1].set_title('High-Pass Filtered Field (no variation)')
            else:
                axes[0, 1].text(0.5, 0.5, 'No valid data\nin high-pass field', 
                              ha='center', va='center', transform=axes[0, 1].transAxes)
                axes[0, 1].set_title('High-Pass Filtered Field (no data)')
            
            # Band-pass fields
            band_pass_fields = frequency_results['band_pass_fields']
            for i, (band_name, band_field) in enumerate(band_pass_fields.items()):
                if i >= 2:  # Only plot first 2 band-pass fields
                    break
                
                ax = axes[1, i]
                valid_bp = band_field[~np.isnan(band_field)]
                logger.info(f"Band-pass {band_name} valid points: {len(valid_bp)}/{band_field.size}")
                
                if len(valid_bp) > 0:
                    bp_range = np.ptp(valid_bp)
                    logger.info(f"Band-pass {band_name} range: {bp_range:.2f}")
                    
                    if bp_range > 1e-10:  # Only plot if there's meaningful variation
                        vmin, vmax = np.percentile(valid_bp, [2, 98])
                        im = ax.contourf(grid_lon_mesh, grid_lat_mesh, band_field, 
                                       levels=50, cmap='viridis', vmin=vmin, vmax=vmax)
                        ax.set_title(f'Band-Pass {band_name} (range: {bp_range:.2f})')
                        ax.set_xlabel('Longitude')
                        ax.set_ylabel('Latitude')
                        plt.colorbar(im, ax=ax)
                    else:
                        ax.text(0.5, 0.5, f'Band-pass {band_name}\nhas no variation', 
                               ha='center', va='center', transform=ax.transAxes)
                        ax.set_title(f'Band-Pass {band_name} (no variation)')
                else:
                    ax.text(0.5, 0.5, f'No valid data\nin band-pass {band_name}', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'Band-Pass {band_name} (no data)')
            
            # Handle empty subplots
            for i in range(len(band_pass_fields), 2):
                axes[1, i].axis('off')
            
            plt.tight_layout()
            output_file = output_dir / 'frequency_domain_analysis.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Generated frequency domain plots: {output_file}")
            
        except Exception as e:
            logger.error(f"Error generating frequency plots: {e}")
    
    # Final summary
    logger.info("="*60)
    logger.info("MAGNETIC ANOMALY DETECTION ANALYSIS COMPLETED")
    logger.info("="*60)
    logger.info(f"Dataset: {len(df)} points from {INPUT_DIRECTORY}")
    logger.info(f"Field analyzed: {mag_field_col}")
    logger.info(f"Grid resolution: {GRID_RESOLUTION}x{GRID_RESOLUTION}")
    logger.info("")
    logger.info("Generated outputs:")
    logger.info(f"  • Interactive map: {output_dir / OUTPUT_HTML_FILENAME}")
    logger.info(f"  • Residual plots: {output_dir / 'anomaly_residual_fields.png'}")
    logger.info(f"  • Summary analysis: {output_dir / 'anomaly_summary_analysis.png'}")
    logger.info(f"  • Anomaly report: {output_dir / 'magnetic_anomaly_report.csv'}")
    if frequency_results:
        logger.info(f"  • Frequency analysis: {output_dir / 'frequency_domain_analysis.png'}")
    logger.info("")
    logger.info("Minimum curvature interpolation scales generated:")
    for scale_name, config in SCALE_CONFIGURATIONS.items():
        logger.info(f"  • {scale_name}: {config['description']} ({config['max_points']} points, omega={config['omega']})")
    logger.info("")
    logger.info("Residual fields computed:")
    for residual_name in residuals.keys():
        logger.info(f"  • {residual_name}")
    logger.info("")
    logger.info("Anomaly detection methods applied:")
    total_anomalies = 0
    for key, anomalies in anomaly_results.items():
        if 'statistical' in key and isinstance(anomalies, dict):
            for threshold, data in anomalies.items():
                logger.info(f"  • {key} - {threshold}: {data['total_count']} anomalies")
                total_anomalies += data['total_count']
        elif key == 'gradient_anomalies' and anomalies:
            count = np.sum(anomalies['anomaly_mask'])
            logger.info(f"  • Gradient anomalies: {count} anomalies")
            total_anomalies += count
    
    logger.info("")
    logger.info(f"Total anomalies detected: {total_anomalies}")
    logger.info("Analysis complete! Check the generated files for detailed results.")