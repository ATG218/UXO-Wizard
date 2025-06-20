#!/usr/bin/env python3
"""
Grid Interpolation and Interactive Mapping Script
===============================================

Combines CSV files from a directory, performs 2D grid interpolation using various methods
(focusing on kriging and minimum curvature), and creates interactive maps with Folium.

Features:
- Automatic CSV file combination from specified directory
- Multiple interpolation methods: kriging and minimum curvature
- Interactive Folium maps with toggleable layers
- High-quality magnetic field visualization
- Comprehensive statistical analysis
"""

import pandas as pd
import numpy as np
from branca.colormap import linear
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from pykrige import OrdinaryKriging, UniversalKriging
import folium
from pathlib import Path
import logging
import base64
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.contour import ContourSet
import warnings
warnings.filterwarnings('ignore')

# Try to import Numba for JIT compilation
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    # Define a no-op decorator if Numba is not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    NUMBA_AVAILABLE = False

# =============================================
# CONFIGURATION
# =============================================

# Input directory containing the CSV files (modify this path)
INPUT_DIRECTORY = "/Users/aleksandergarbuz/Documents/SINTEF/data/OldFlights/interpolation"

# Field selection - specify which field to interpolate
TARGET_FIELD = "R1 [nT]"  # Options: "R1 [nT]", "Btotal1 [nT]", "Total [nT]", "B_total [nT]", or "auto"

# Interpolation settings
GRID_RESOLUTION = 300  # Number of points per axis for interpolation grid
MAX_KRIGING_POINTS = 15000  # Maximum points for kriging (downsampled for performance & memory)
# Alternative kriging point limits for different approaches:
MAX_KRIGING_POINTS_FAST = 5000      # Fast kriging with heavy downsampling
MAX_KRIGING_POINTS_MEDIUM = 15000   # Medium quality kriging (good balance)
MAX_KRIGING_POINTS_HIGH = 25000     # High quality kriging (slower but better)
INTERPOLATION_METHODS = [
    #'ordinary_kriging_fast',         # Fast ordinary kriging (5k points, gaussian)
    #'ordinary_kriging_medium',       # Medium ordinary kriging (15k points, gaussian)
    #'ordinary_kriging_exponential_fast',    # Fast exponential kriging (5k points)
    #'ordinary_kriging_exponential_medium',  # Medium exponential kriging (15k points)
    #'ordinary_kriging_exponential_high',    # High exponential kriging (25k points)
    #'ordinary_kriging_high',         # High quality ordinary kriging (25k points, gaussian)
    'ordinary_kriging_spherical_fast',    # Fast spherical kriging (5k points)
    #'ordinary_kriging_linear_medium',       # Medium linear kriging (15k points)
    'minimum_curvature'             # Minimum curvature interpolation for smooth surfaces
]

# Processing settings
SKIP_INTERPOLATION = True  # Set to True to skip interpolation and use existing CSV grids
MAX_FLIGHT_PATH_SEGMENTS = 5000  # Limit flight path segments to reduce HTML file size
SPLIT_HTML_BY_METHOD = False  # Set to True to create separate HTML file for each method
GENERATE_DIAGNOSTIC_PLOTS = True  # Set to True to generate diagnostic PNG plots
GENERATE_FIELD_PLOTS = True  # Set to True to generate field visualization PNGs with contours
GENERATE_NOISE_ANALYSIS = True  # Set to True to generate noise analysis and filtering preview plots

# Boundary masking settings
ENABLE_BOUNDARY_MASKING = True  # Set to True to enable boundary masking
BOUNDARY_METHOD = 'convex_hull'  # Options: 'convex_hull', 'alpha_shape', 'distance'
BOUNDARY_BUFFER_DISTANCE = None  # Optional buffer distance in coordinate units (None for no buffer)

# Visualization settings
MAPBOX_TOKEN = "pk.eyJ1IjoiYXRnMjE3IiwiYSI6ImNtYzBnY2kwOTAxbWwybHM3NmN0bnRlaWcifQ.B8hh4dBszYXxlj-O0KGqkg"  # Replace with your Mapbox token

# Output settings
OUTPUT_HTML_FILENAME = "combined_magnetic_field_map.html"

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

# =============================================
# DATA LOADING AND COMBINATION
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
        logger.error(f"No magnetic field column found. Looking for one of: {magnetic_columns}")
        return False
    
    # Determine which magnetic field column to use
    for col in magnetic_columns:
        if col in df.columns:
            logger.info(f"Using magnetic field column: {col}")
            break
    
    return True

def get_magnetic_field_column(df):
    """Get the name of the magnetic field column in the DataFrame"""
    logger = logging.getLogger(__name__)
    
    # If specific field is requested and exists, use it
    if TARGET_FIELD != "auto" and TARGET_FIELD in df.columns:
        logger.info(f"Using specified target field: {TARGET_FIELD}")
        return TARGET_FIELD
    
    # Otherwise, search for available magnetic field columns
    magnetic_columns = ['R1 [nT]', 'Btotal1 [nT]', 'Total [nT]', 'B_total [nT]']
    for col in magnetic_columns:
        if col in df.columns:
            logger.info(f"Auto-detected magnetic field column: {col}")
            return col
    
    logger.error(f"Target field '{TARGET_FIELD}' not found and no standard magnetic field columns available")
    return None

# =============================================
# INTERPOLATION METHODS
# =============================================

def select_best_variogram_model(x, y, z, models=['gaussian', 'spherical', 'exponential']):
    """
    Select the best variogram model by comparing log-likelihood or cross-validation.
    
    Args:
        x, y, z: Input coordinates and values
        models: List of variogram models to test
    
    Returns:
        best_model: Name of the best variogram model
        model_scores: Dictionary of model scores
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Clean input data
        valid_mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
        if not np.any(valid_mask):
            logger.warning("No valid data for variogram model selection")
            return 'gaussian', {}
        
        x_clean = x[valid_mask]
        y_clean = y[valid_mask]
        z_clean = z[valid_mask]
        
        # Downsample if needed for faster model comparison
        if len(x_clean) > 1000:  # Use smaller subset for model selection
            x_clean, y_clean, z_clean = downsample_for_kriging(x_clean, y_clean, z_clean, 1000)
        
        model_scores = {}
        best_model = 'gaussian'  # Default fallback
        best_score = float('inf')
        
        for model in models:
            try:
                # Create kriging object
                ok = OrdinaryKriging(
                    x_clean, y_clean, z_clean,
                    variogram_model=model,
                    weight=True,
                    exact_values=True,
                    verbose=False,
                    enable_plotting=False,
                    coordinates_type='geographic'
                )
                
                # Execute kriging on a small test grid to fit the variogram
                # Create a minimal test grid for evaluation
                test_x = np.linspace(x_clean.min(), x_clean.max(), 3)
                test_y = np.linspace(y_clean.min(), y_clean.max(), 3)
                
                try:
                    # Execute to fit the variogram
                    _, _ = ok.execute('grid', test_x, test_y)
                    
                    # Now try to access fitted parameters
                    if hasattr(ok, 'lnlike') and ok.lnlike is not None:
                        score = -ok.lnlike  # Use negative log-likelihood (lower is better)
                    else:
                        # Use variogram parameters if available
                        nugget = getattr(ok, 'nugget', 0)
                        sill = getattr(ok, 'sill', np.var(z_clean))
                        range_param = getattr(ok, 'range', 1)
                        
                        # Simple score based on how well variogram parameters make sense
                        data_variance = np.var(z_clean)
                        score = abs(nugget) + abs(sill - data_variance) * 0.1 + abs(range_param) * 0.001
                        
                except Exception as e:
                    # If execution fails, use cross-validation fallback
                    logger.warning(f"Model {model} execution failed, using cross-validation: {e}")
                    
                    # Simple cross-validation for small datasets
                    if len(x_clean) <= 50:
                        residuals = []
                        for i in range(0, len(x_clean), max(1, len(x_clean)//10)):  # Sample every nth point
                            try:
                                # Leave one out
                                x_loo = np.concatenate([x_clean[:i], x_clean[i+1:]])
                                y_loo = np.concatenate([y_clean[:i], y_clean[i+1:]])
                                z_loo = np.concatenate([z_clean[:i], z_clean[i+1:]])
                                
                                if len(x_loo) < 3:  # Need minimum points for kriging
                                    continue
                                
                                # Create kriging with reduced data
                                ok_loo = OrdinaryKriging(
                                    x_loo, y_loo, z_loo,
                                    variogram_model=model,
                                    weight=True,
                                    exact_values=True,
                                    verbose=False,
                                    enable_plotting=False,
                                    coordinates_type='geographic'
                                )
                                
                                # Predict at left-out point
                                pred, _ = ok_loo.execute('points', [x_clean[i]], [y_clean[i]])
                                residuals.append((pred[0] - z_clean[i])**2)
                                
                            except Exception:
                                continue
                        
                        score = np.mean(residuals) if residuals else float('inf')
                    else:
                        # For larger datasets, just use a default penalty
                        score = float('inf')
                
                model_scores[model] = score
                
                if score < best_score:
                    best_score = score
                    best_model = model
                
                logger.info(f"Variogram model {model}: score = {score:.3f}")
                
            except Exception as e:
                logger.warning(f"Failed to evaluate variogram model {model}: {e}")
                model_scores[model] = float('inf')
                continue
        
        logger.info(f"Best variogram model selected: {best_model} (score: {best_score:.3f})")
        return best_model, model_scores
        
    except Exception as e:
        logger.error(f"Error in variogram model selection: {e}")
        return 'gaussian', {}

def downsample_for_kriging(x, y, z, max_points):
    """
    Intelligently downsample data for kriging while preserving spatial distribution
    and semivariogram structure.
    """
    logger = logging.getLogger(__name__)
    
    if len(x) <= max_points:
        logger.info(f"No downsampling needed: {len(x)} points <= {max_points}")
        return x, y, z
    
    logger.info(f"Downsampling from {len(x)} to {max_points} points for kriging")
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    try:
        # Strategy: Stratified sampling with k-means clustering to preserve spatial structure
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        # Prepare coordinate data for clustering
        coords = np.column_stack([x, y])
        
        # Standardize coordinates for clustering
        scaler = StandardScaler()
        coords_scaled = scaler.fit_transform(coords)
        
        # Use k-means to create spatial clusters
        n_clusters = min(max_points // 2, len(x) // 10, 100)  # Reasonable number of clusters
        if n_clusters < 2:
            n_clusters = min(2, len(x))
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(coords_scaled)
        
        # Sample from each cluster proportionally
        sampled_indices = []
        points_per_cluster = max_points / n_clusters
        
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) == 0:
                continue
            
            # Number of points to sample from this cluster
            n_sample = max(1, int(np.round(points_per_cluster)))
            n_sample = min(n_sample, len(cluster_indices))
            
            # Within each cluster, prefer points with extreme values or good spatial distribution
            if len(cluster_indices) <= n_sample:
                sampled_indices.extend(cluster_indices)
            else:
                # Combine value-based and spatial sampling
                cluster_z = z[cluster_indices]
                
                # Sample some extreme values (high/low)
                extreme_indices = []
                if len(cluster_indices) >= 4:
                    z_sorted_idx = np.argsort(cluster_z)
                    # Take some highest and lowest values
                    n_extreme = min(n_sample // 3, len(cluster_indices) // 4)
                    if n_extreme > 0:
                        extreme_indices.extend(cluster_indices[z_sorted_idx[:n_extreme]])  # Lowest
                        extreme_indices.extend(cluster_indices[z_sorted_idx[-n_extreme:]])  # Highest
                
                # Fill remaining with random sampling from cluster
                remaining_needed = n_sample - len(extreme_indices)
                if remaining_needed > 0:
                    available = list(set(cluster_indices) - set(extreme_indices))
                    if len(available) > 0:
                        additional = np.random.choice(available, 
                                                     min(remaining_needed, len(available)), 
                                                     replace=False)
                        extreme_indices.extend(additional)
                
                sampled_indices.extend(extreme_indices[:n_sample])
        
        # Convert to array and ensure we don't exceed max_points
        sampled_indices = np.array(sampled_indices)
        if len(sampled_indices) > max_points:
            sampled_indices = np.random.choice(sampled_indices, max_points, replace=False)
        
        logger.info(f"K-means based sampling selected {len(sampled_indices)} points from {n_clusters} clusters")
        
        return x[sampled_indices], y[sampled_indices], z[sampled_indices]
        
    except ImportError:
        logger.warning("sklearn not available, using grid-based sampling")
        # Fallback to original grid-based method if sklearn not available
        
    except Exception as e:
        logger.warning(f"K-means sampling failed: {e}, falling back to grid-based sampling")
    
    # Fallback: Grid-based downsampling to preserve spatial distribution
    try:
        # Create a coarse grid for spatial binning
        n_bins = int(np.sqrt(max_points))  # Roughly square grid
        
        # Create bins
        x_bins = np.linspace(x.min(), x.max(), n_bins + 1)
        y_bins = np.linspace(y.min(), y.max(), n_bins + 1)
        
        # Digitize points into bins
        x_indices = np.digitize(x, x_bins) - 1
        y_indices = np.digitize(y, y_bins) - 1
        
        # Clip to valid range
        x_indices = np.clip(x_indices, 0, n_bins - 1)
        y_indices = np.clip(y_indices, 0, n_bins - 1)
        
        # Sample points from each bin, preferring extreme values
        sampled_indices = []
        points_per_bin = max(1, max_points // (n_bins * n_bins))
        
        for i in range(n_bins):
            for j in range(n_bins):
                # Find points in this bin
                bin_mask = (x_indices == i) & (y_indices == j)
                bin_indices = np.where(bin_mask)[0]
                
                if len(bin_indices) > 0:
                    # Sample up to points_per_bin from this bin
                    n_sample = min(len(bin_indices), points_per_bin)
                    
                    if len(bin_indices) <= n_sample:
                        sampled_indices.extend(bin_indices)
                    else:
                        # Prefer extreme values within bin
                        bin_z = z[bin_indices]
                        z_sorted_idx = np.argsort(bin_z)
                        
                        # Take mix of extreme and random values
                        n_extreme = n_sample // 2
                        selected = []
                        if n_extreme > 0:
                            selected.extend(bin_indices[z_sorted_idx[:n_extreme//2]])  # Lowest
                            selected.extend(bin_indices[z_sorted_idx[-(n_extreme-n_extreme//2):]])  # Highest
                        
                        # Fill remaining with random
                        remaining = n_sample - len(selected)
                        if remaining > 0:
                            available = list(set(bin_indices) - set(selected))
                            if available:
                                additional = np.random.choice(available, 
                                                             min(remaining, len(available)), 
                                                             replace=False)
                                selected.extend(additional)
                        
                        sampled_indices.extend(selected)
        
        # Convert to array and limit to max_points
        sampled_indices = np.array(sampled_indices)
        if len(sampled_indices) > max_points:
            sampled_indices = np.random.choice(sampled_indices, max_points, replace=False)
        
        logger.info(f"Grid-based sampling selected {len(sampled_indices)} points")
        
        return x[sampled_indices], y[sampled_indices], z[sampled_indices]
        
    except Exception as e:
        logger.warning(f"Grid-based sampling failed: {e}, falling back to random sampling")
        
        # Final fallback: Random sampling
        indices = np.random.choice(len(x), max_points, replace=False)
        logger.info(f"Random sampling selected {len(indices)} points")
        
        return x[indices], y[indices], z[indices]

def enhanced_kriging_downsample(x, y, z, max_points):
    """
    Enhanced downsampling specifically optimized for kriging performance.
    
    Uses multiple strategies to preserve spatial correlation structure:
    1. Stratified spatial sampling to maintain coverage
    2. Preserve extreme values and gradients
    3. Ensure minimum distance between points to avoid clustering
    4. Adaptive density based on local variation
    """
    logger = logging.getLogger(__name__)
    
    if len(x) <= max_points:
        logger.info(f"No downsampling needed: {len(x)} points <= {max_points}")
        return x, y, z
    
    logger.info(f"Enhanced kriging downsampling from {len(x)} to {max_points} points")
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    try:
        # Strategy 1: Multi-stage adaptive sampling
        
        # Stage 1: Remove obvious redundancy (very close points)
        min_distance = np.sqrt((x.max() - x.min()) * (y.max() - y.min())) / np.sqrt(max_points * 2)
        
        # Build spatial index for efficient neighbor finding
        coords = np.column_stack([x, y])
        
        # Greedy selection maintaining minimum distance
        selected_indices = []
        remaining_indices = list(range(len(x)))
        
        # Start with random point
        first_idx = np.random.choice(remaining_indices)
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)
        
        # Iteratively add points that are far enough from existing points
        for _ in range(min(max_points - 1, len(remaining_indices))):
            if not remaining_indices:
                break
                
            # Calculate distances to all selected points
            best_idx = None
            best_min_dist = 0
            
            for candidate_idx in remaining_indices:
                candidate_pos = coords[candidate_idx]
                
                # Find minimum distance to any selected point
                min_dist_to_selected = float('inf')
                for selected_idx in selected_indices:
                    selected_pos = coords[selected_idx]
                    dist = np.sqrt(np.sum((candidate_pos - selected_pos)**2))
                    min_dist_to_selected = min(min_dist_to_selected, dist)
                
                # Select point that maximizes minimum distance (farthest from existing)
                if min_dist_to_selected > best_min_dist:
                    best_min_dist = min_dist_to_selected
                    best_idx = candidate_idx
            
            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
            else:
                break
        
        # If we still need more points, add remaining with preference for extreme values
        if len(selected_indices) < max_points and remaining_indices:
            remaining_z = z[remaining_indices]
            
            # Sort by z-value and take extreme values
            z_order = np.argsort(remaining_z)
            remaining_sorted = np.array(remaining_indices)[z_order]
            
            # Take alternating from high and low values
            additional_needed = max_points - len(selected_indices)
            additional_indices = []
            
            for i in range(min(additional_needed, len(remaining_sorted))):
                if i % 2 == 0:
                    # Take from low end
                    additional_indices.append(remaining_sorted[i // 2])
                else:
                    # Take from high end
                    additional_indices.append(remaining_sorted[-(i // 2 + 1)])
            
            selected_indices.extend(additional_indices[:additional_needed])
        
        selected_indices = np.array(selected_indices[:max_points])
        
        logger.info(f"Enhanced sampling selected {len(selected_indices)} points with improved spatial distribution")
        
        return x[selected_indices], y[selected_indices], z[selected_indices]
        
    except Exception as e:
        logger.warning(f"Enhanced sampling failed: {e}, falling back to k-means sampling")
        # Fallback to original k-means method
        return downsample_for_kriging(x, y, z, max_points)

def detect_flight_lines(x, y, z, source_files=None):
    """
    Detect flight lines in 250 Hz magnetic survey data.
    
    Args:
        x, y, z: Coordinate and value arrays
        source_files: Array of source file names for each point
    
    Returns:
        flight_lines: List of dictionaries with flight line information
    """
    logger = logging.getLogger(__name__)
    
    try:
        flight_lines = []
        
        if source_files is not None:
            # Use source file information to separate flight lines
            unique_files = np.unique(source_files)
            
            for file_name in unique_files:
                file_mask = source_files == file_name
                file_x = x[file_mask]
                file_y = y[file_mask]
                file_z = z[file_mask]
                file_indices = np.where(file_mask)[0]
                
                if len(file_x) < 10:  # Skip files with too few points
                    continue
                
                # For each file, detect individual flight line segments
                line_segments = detect_segments_in_file(file_x, file_y, file_z, file_indices)
                
                for segment in line_segments:
                    segment['source_file'] = file_name
                    flight_lines.append(segment)
        
        else:
            # Use spatial clustering to detect flight lines
            flight_lines = detect_lines_by_clustering(x, y, z)
        
        logger.info(f"Detected {len(flight_lines)} flight line segments")
        return flight_lines
        
    except Exception as e:
        logger.warning(f"Flight line detection failed: {e}, using simple approach")
        # Fallback: treat each source file as one flight line
        return [{'indices': np.arange(len(x)), 'source_file': 'combined', 
                'start_idx': 0, 'end_idx': len(x)-1}]

def detect_segments_in_file(x, y, z, original_indices):
    """
    Detect individual flight line segments within a single source file.
    """
    try:
        segments = []
        
        if len(x) < 10:
            return segments
        
        # Calculate movement vectors between consecutive points
        dx = np.diff(x)
        dy = np.diff(y)
        distances = np.sqrt(dx**2 + dy**2)
        
        # Find large gaps that indicate flight line breaks
        gap_threshold = np.percentile(distances[distances > 0], 95) * 3  # 3x the 95th percentile
        gap_indices = np.where(distances > gap_threshold)[0]
        
        # Create segments between gaps
        start_idx = 0
        
        for gap_idx in gap_indices:
            if gap_idx - start_idx > 50:  # Minimum segment length
                segment_indices = original_indices[start_idx:gap_idx+1]
                segments.append({
                    'indices': segment_indices,
                    'start_idx': start_idx,
                    'end_idx': gap_idx,
                    'length': len(segment_indices),
                    'x_range': [x[start_idx], x[gap_idx]],
                    'y_range': [y[start_idx], y[gap_idx]]
                })
            start_idx = gap_idx + 1
        
        # Add final segment
        if len(x) - start_idx > 50:
            segment_indices = original_indices[start_idx:]
            segments.append({
                'indices': segment_indices,
                'start_idx': start_idx,
                'end_idx': len(x)-1,
                'length': len(segment_indices),
                'x_range': [x[start_idx], x[-1]],
                'y_range': [y[start_idx], y[-1]]
            })
        
        return segments
        
    except Exception as e:
        # Fallback: treat entire file as one segment
        return [{'indices': original_indices, 'start_idx': 0, 'end_idx': len(x)-1, 
                'length': len(x)}]

def detect_lines_by_clustering(x, y, z):
    """
    Detect flight lines using spatial clustering when source files aren't available.
    """
    try:
        from sklearn.cluster import DBSCAN
        from sklearn.preprocessing import StandardScaler
        
        # Prepare data for clustering
        coords = np.column_stack([x, y])
        scaler = StandardScaler()
        coords_scaled = scaler.fit_transform(coords)
        
        # Use DBSCAN to find linear clusters
        clustering = DBSCAN(eps=0.1, min_samples=50).fit(coords_scaled)
        labels = clustering.labels_
        
        flight_lines = []
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            if label == -1:  # Skip noise points
                continue
                
            cluster_mask = labels == label
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) > 100:  # Minimum flight line length
                flight_lines.append({
                    'indices': cluster_indices,
                    'start_idx': cluster_indices[0],
                    'end_idx': cluster_indices[-1],
                    'length': len(cluster_indices),
                    'source_file': f'cluster_{label}'
                })
        
        return flight_lines
        
    except ImportError:
        # Fallback if sklearn not available
        return [{'indices': np.arange(len(x)), 'source_file': 'combined', 
                'start_idx': 0, 'end_idx': len(x)-1}]

def downsample_flight_line(x, y, z, indices, target_points):
    """
    Smart downsampling of a single flight line preserving spatial structure.
    
    Args:
        x, y, z: Full coordinate arrays
        indices: Indices belonging to this flight line
        target_points: Target number of points for this line
    
    Returns:
        selected_indices: Downsampled indices maintaining spatial distribution
    """
    if len(indices) <= target_points:
        return indices
    
    line_x = x[indices]
    line_y = y[indices]
    line_z = z[indices]
    
    # Calculate cumulative distance along flight line
    distances = np.sqrt(np.diff(line_x)**2 + np.diff(line_y)**2)
    cumulative_dist = np.concatenate([[0], np.cumsum(distances)])
    
    # Create evenly spaced points along the flight path
    target_distances = np.linspace(0, cumulative_dist[-1], target_points)
    
    # Find closest actual points to target distances
    selected_local_indices = []
    for target_dist in target_distances:
        closest_idx = np.argmin(np.abs(cumulative_dist - target_dist))
        selected_local_indices.append(closest_idx)
    
    # Remove duplicates while preserving order
    selected_local_indices = sorted(list(set(selected_local_indices)))
    
    # Convert back to global indices
    selected_indices = indices[selected_local_indices]
    
    return selected_indices

def flight_line_aware_downsample(x, y, z, source_files, target_total_points):
    """
    Perform flight-line-aware downsampling that preserves spatial structure.
    
    Args:
        x, y, z: Coordinate and value arrays
        source_files: Source file information for flight line detection
        target_total_points: Total target number of points after downsampling
    
    Returns:
        selected_indices: Indices of selected points
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Detect flight lines
        flight_lines = detect_flight_lines(x, y, z, source_files)
        
        if not flight_lines:
            logger.warning("No flight lines detected, using random downsampling")
            if len(x) <= target_total_points:
                return np.arange(len(x))
            return np.random.choice(len(x), target_total_points, replace=False)
        
        # Calculate points per line based on line length
        total_length = sum(line['length'] for line in flight_lines)
        selected_indices = []
        
        for line in flight_lines:
            # Proportional allocation based on line length
            line_proportion = line['length'] / total_length
            line_target_points = max(1, int(target_total_points * line_proportion))
            
            # Ensure we don't exceed available points
            line_target_points = min(line_target_points, line['length'])
            
            # Downsample this flight line
            line_selected = downsample_flight_line(x, y, z, line['indices'], line_target_points)
            selected_indices.extend(line_selected)
        
        selected_indices = np.array(selected_indices)
        
        # If we have too many points due to rounding, randomly remove some
        if len(selected_indices) > target_total_points:
            keep_indices = np.random.choice(len(selected_indices), target_total_points, replace=False)
            selected_indices = selected_indices[keep_indices]
        
        logger.info(f"Flight-line-aware downsampling: {len(x)} -> {len(selected_indices)} points "
                   f"across {len(flight_lines)} flight lines")
        
        return selected_indices
        
    except Exception as e:
        logger.error(f"Flight-line-aware downsampling failed: {e}")
        # Fallback to random downsampling
        if len(x) <= target_total_points:
            return np.arange(len(x))
        return np.random.choice(len(x), target_total_points, replace=False)

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
                alpha_shape = alphashape.alphashape(data_points, alpha=0.1)
                
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
    
    # Log masking statistics
    total_points = grid_field.size
    masked_points = np.sum(~mask)
    valid_points = np.sum(~np.isnan(masked_field))
    
    logger = logging.getLogger(__name__)
    logger.info(f"Applied boundary mask: {masked_points:,} points masked out, "
               f"{valid_points:,} valid interpolated points remaining")
    
    return masked_field

def ordinary_kriging_interpolation(x, y, z, grid_x, grid_y, variogram_model='gaussian', max_points=None, source_files=None):
    """
    Ordinary kriging interpolation using pykrige.
    
    Automatically downsamples large datasets to MAX_KRIGING_POINTS for performance.
    Uses spatial grid-based sampling to preserve geographic distribution.
    
    Args:
        x, y, z: Input coordinates and values
        grid_x, grid_y: Meshgrid for interpolation
        variogram_model: Variogram model ('gaussian', 'spherical', 'exponential', 'linear')
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Clean input data
        valid_mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
        if not np.any(valid_mask):
            logger.warning(f"No valid data points for ordinary kriging ({variogram_model})")
            return np.full(grid_x.shape, np.nan)
        
        x_clean = x[valid_mask]
        y_clean = y[valid_mask] 
        z_clean = z[valid_mask]
        
        # Downsample for kriging performance using flight-line-aware method when possible
        point_limit = max_points if max_points is not None else MAX_KRIGING_POINTS
        
        if source_files is not None:
            # Use flight-line-aware downsampling
            source_files_clean = source_files[valid_mask]
            selected_indices = flight_line_aware_downsample(x_clean, y_clean, z_clean, source_files_clean, point_limit)
            x_clean = x_clean[selected_indices]
            y_clean = y_clean[selected_indices]
            z_clean = z_clean[selected_indices]
        else:
            # Fallback to enhanced downsampling
            x_clean, y_clean, z_clean = enhanced_kriging_downsample(x_clean, y_clean, z_clean, point_limit)
        
        logger.info(f"Ordinary kriging ({variogram_model}) with {len(x_clean)} data points...")
        
        # Create 1D coordinate arrays for kriging grid - FIXED coordinate handling
        # Get unique sorted coordinates
        grid_lon_1d = np.unique(grid_x.ravel())
        grid_lat_1d = np.unique(grid_y.ravel())
        
        # Sort to ensure proper ordering
        grid_lon_1d = np.sort(grid_lon_1d)
        grid_lat_1d = np.sort(grid_lat_1d)
        
        logger.info(f"Grid dimensions: {len(grid_lon_1d)} x {len(grid_lat_1d)}")
        
        # Create ordinary kriging object
        ok = OrdinaryKriging(
            x_clean, y_clean, z_clean,
            variogram_model=variogram_model,
            weight=True,
            exact_values=True,
            verbose=False,
            enable_plotting=False,
            coordinates_type='geographic'  # Specify we're using lat/lon
        )
        
        # Execute kriging
        z_pred, _ = ok.execute('grid', grid_lon_1d, grid_lat_1d)
        
        # Log variogram parameters for debugging (safely handle missing attributes)
        try:
            # Try different possible attribute names in pykrige
            model_name = getattr(ok, 'variogram_model', variogram_model)
            
            # Try various ways to access parameters (safely handle different parameter counts)
            params = getattr(ok, 'variogram_model_parameters', []) if hasattr(ok, 'variogram_model_parameters') else []
            nugget = getattr(ok, 'nugget', params[0] if len(params) > 0 else 'N/A')
            sill = getattr(ok, 'sill', params[1] if len(params) > 1 else 'N/A')
            range_param = getattr(ok, 'range', params[2] if len(params) > 2 else 'N/A')
            
            # Try accessing through other possible attributes
            if nugget == 'N/A' and hasattr(ok, 'variogram_function'):
                try:
                    params = getattr(ok.variogram_function, 'variogram_model_parameters', None)
                    if params and len(params) >= 3:
                        nugget, sill, range_param = params[:3]
                except Exception:
                    pass
            
            # Format parameters nicely
            if isinstance(nugget, (int, float)) and nugget != 'N/A':
                nugget = f"{nugget:.3f}"
            if isinstance(sill, (int, float)) and sill != 'N/A':
                sill = f"{sill:.3f}"
            if isinstance(range_param, (int, float)) and range_param != 'N/A':
                range_param = f"{range_param:.6f}"
            
            logger.info(f"Variogram parameters - Model: {model_name}, "
                       f"Nugget: {nugget}, Sill: {sill}, Range: {range_param}")
                       
            # Also log what attributes are actually available for debugging
            available_attrs = [attr for attr in dir(ok) if not attr.startswith('_') and 'variogram' in attr.lower()]
            if available_attrs:
                logger.debug(f"Available variogram-related attributes: {available_attrs}")
                
        except Exception as e:
            logger.warning(f"Could not log variogram parameters: {e}")
        
        logger.info(f"Ordinary kriging ({variogram_model}) completed successfully")
        return z_pred
        
    except Exception as e:
        logger.error(f"Error in ordinary kriging ({variogram_model}): {e} - falling back to linear")
        return griddata((x, y), z, (grid_x, grid_y), method='linear', fill_value=np.nan)

def universal_kriging_interpolation(x, y, z, grid_x, grid_y, variogram_model='gaussian', source_files=None):
    """
    Universal kriging interpolation using pykrige.
    
    Automatically downsamples large datasets to MAX_KRIGING_POINTS for performance.
    Uses spatial grid-based sampling to preserve geographic distribution.
    Includes regional linear drift terms for trend modeling.
    
    Args:
        x, y, z: Input coordinates and values
        grid_x, grid_y: Meshgrid for interpolation
        variogram_model: Variogram model ('gaussian', 'spherical', 'exponential', 'linear')
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Clean input data
        valid_mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
        if not np.any(valid_mask):
            logger.warning(f"No valid data points for universal kriging ({variogram_model})")
            return np.full(grid_x.shape, np.nan)
        
        x_clean = x[valid_mask]
        y_clean = y[valid_mask] 
        z_clean = z[valid_mask]
        
        # Downsample for kriging performance using flight-line-aware method when possible
        if source_files is not None:
            # Use flight-line-aware downsampling
            source_files_clean = source_files[valid_mask]
            selected_indices = flight_line_aware_downsample(x_clean, y_clean, z_clean, source_files_clean, MAX_KRIGING_POINTS)
            x_clean = x_clean[selected_indices]
            y_clean = y_clean[selected_indices]
            z_clean = z_clean[selected_indices]
        else:
            # Fallback to standard downsampling
            x_clean, y_clean, z_clean = downsample_for_kriging(x_clean, y_clean, z_clean, MAX_KRIGING_POINTS)
        
        logger.info(f"Universal kriging ({variogram_model}) with {len(x_clean)} data points...")
        
        # Create 1D coordinate arrays for kriging grid - FIXED coordinate handling
        # Get unique sorted coordinates
        grid_lon_1d = np.unique(grid_x.ravel())
        grid_lat_1d = np.unique(grid_y.ravel())
        
        # Sort to ensure proper ordering
        grid_lon_1d = np.sort(grid_lon_1d)
        grid_lat_1d = np.sort(grid_lat_1d)
        
        logger.info(f"Grid dimensions: {len(grid_lon_1d)} x {len(grid_lat_1d)}")
        
        # Create universal kriging object with drift terms
        uk = UniversalKriging(
            x_clean, y_clean, z_clean,
            variogram_model=variogram_model,
            drift_terms=['regional_linear'],  # Accounts for regional trends
            weight=True,
            exact_values=True,
            verbose=False,
            enable_plotting=False
        )
        
        # Execute kriging
        z_pred, _ = uk.execute('grid', grid_lon_1d, grid_lat_1d)
        
        # Log variogram parameters for debugging (safely handle missing attributes)
        try:
            # Try different possible attribute names in pykrige
            model_name = getattr(uk, 'variogram_model', variogram_model)
            
            # Try various ways to access parameters (safely handle different parameter counts)
            params = getattr(uk, 'variogram_model_parameters', []) if hasattr(uk, 'variogram_model_parameters') else []
            nugget = getattr(uk, 'nugget', params[0] if len(params) > 0 else 'N/A')
            sill = getattr(uk, 'sill', params[1] if len(params) > 1 else 'N/A')
            range_param = getattr(uk, 'range', params[2] if len(params) > 2 else 'N/A')
            
            # Try accessing through other possible attributes
            if nugget == 'N/A' and hasattr(uk, 'variogram_function'):
                try:
                    params = getattr(uk.variogram_function, 'variogram_model_parameters', None)
                    if params and len(params) >= 3:
                        nugget, sill, range_param = params[:3]
                except Exception:
                    pass
            
            # Format parameters nicely
            if isinstance(nugget, (int, float)) and nugget != 'N/A':
                nugget = f"{nugget:.3f}"
            if isinstance(sill, (int, float)) and sill != 'N/A':
                sill = f"{sill:.3f}"
            if isinstance(range_param, (int, float)) and range_param != 'N/A':
                range_param = f"{range_param:.6f}"
            
            logger.info(f"Variogram parameters - Model: {model_name}, "
                       f"Nugget: {nugget}, Sill: {sill}, Range: {range_param}")
                       
        except Exception as e:
            logger.warning(f"Could not log variogram parameters: {e}")
        
        logger.info(f"Universal kriging ({variogram_model}) completed successfully")
        return z_pred
        
    except Exception as e:
        logger.error(f"Error in universal kriging ({variogram_model}): {e} - falling back to linear")
        return griddata((x, y), z, (grid_x, grid_y), method='linear', fill_value=np.nan)


def minimum_curvature_interpolation(x, y, z, grid_x, grid_y, max_iterations=100, tolerance=1e-3):
    """
    Minimum curvature interpolation for smooth surfaces.
    
    This method creates surfaces with minimal curvature while honoring data points.
    It's particularly effective for geophysical survey data and produces smooth,
    geologically reasonable interpolations.
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Clean input data
        valid_mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
        if not np.any(valid_mask):
            logger.warning("No valid data points for minimum curvature")
            return np.full(grid_x.shape, np.nan)
        
        x_clean = x[valid_mask]
        y_clean = y[valid_mask]
        z_clean = z[valid_mask]
        
        # Downsample large datasets for performance (minimum curvature is very slow)
        if len(x_clean) > 100000000:
            logger.info(f"Downsampling from {len(x_clean)} to 100000000 points for minimum curvature performance")
            x_clean, y_clean, z_clean = downsample_for_kriging(x_clean, y_clean, z_clean, 100000000)
        
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
        omega = 0.5  # Relaxation factor for stability
        
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
            
            # Early detection of instability (same for both algorithms)
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

def load_existing_grid(output_dir, method, mag_field_col):
    """
    Load existing interpolated grid from CSV file.
    
    Args:
        output_dir: Directory containing the grid CSV files
        method: Interpolation method name
        mag_field_col: Magnetic field column name
    
    Returns:
        grid_lat_mesh, grid_lon_mesh, grid_field, mag_field_col or None if not found
    """
    logger = logging.getLogger(__name__)
    
    # Construct expected filename
    clean_col_name = mag_field_col.replace(' ', '_').replace('[', '').replace(']', '')
    expected_filename = f"grid_{method}_{clean_col_name}.csv"
    grid_file_path = output_dir / expected_filename
    
    if not grid_file_path.exists():
        logger.warning(f"Grid file not found: {grid_file_path}")
        return None
    
    try:
        # Load the grid data
        grid_df = pd.read_csv(grid_file_path)
        
        # Extract coordinates and values
        lats = grid_df['Latitude'].values
        lons = grid_df['Longitude'].values
        field_col = f'{clean_col_name}_interpolated'
        
        if field_col not in grid_df.columns:
            # Try alternative column names
            field_cols = [col for col in grid_df.columns if 'interpolated' in col]
            if field_cols:
                field_col = field_cols[0]
            else:
                logger.error(f"No interpolated field column found in {grid_file_path}")
                return None
        
        field_values = grid_df[field_col].values
        
        # Determine grid dimensions by finding unique coordinates
        unique_lats = np.unique(lats)
        unique_lons = np.unique(lons)
        
        # Create meshgrid
        grid_lon_mesh, grid_lat_mesh = np.meshgrid(unique_lons, unique_lats)
        
        # Reshape field values to match grid
        grid_field = np.full(grid_lat_mesh.shape, np.nan)
        
        # Fill in the field values
        for i, (lat, lon, val) in enumerate(zip(lats, lons, field_values)):
            if np.isfinite(val):
                lat_idx = np.argmin(np.abs(unique_lats - lat))
                lon_idx = np.argmin(np.abs(unique_lons - lon))
                grid_field[lat_idx, lon_idx] = val
        
        logger.info(f"Loaded existing grid from {grid_file_path} ({len(grid_df)} points)")
        return grid_lat_mesh, grid_lon_mesh, grid_field, mag_field_col
        
    except Exception as e:
        logger.error(f"Error loading grid from {grid_file_path}: {e}")
        return None

def interpolate_magnetic_field(df, method='ordinary_kriging'):
    """Main interpolation function that handles the specified method"""
    logger = logging.getLogger(__name__)
    
    # Get data coordinates and values
    latitudes = df['Latitude [Decimal Degrees]'].values
    longitudes = df['Longitude [Decimal Degrees]'].values
    
    # Get magnetic field values
    mag_field_col = get_magnetic_field_column(df)
    if mag_field_col is None:
        raise ValueError("No magnetic field column found in data")
    
    field_values = df[mag_field_col].values
    
    # Get source file information for flight-line-aware processing
    source_files = df['Source_File'].values if 'Source_File' in df.columns else None
    
    # Create interpolation grid
    grid_lat = np.linspace(latitudes.min(), latitudes.max(), GRID_RESOLUTION)
    grid_lon = np.linspace(longitudes.min(), longitudes.max(), GRID_RESOLUTION)
    grid_lon_mesh, grid_lat_mesh = np.meshgrid(grid_lon, grid_lat)
    
    # Choose interpolation method
    if method.startswith('ordinary_kriging_'):
        method_parts = method.split('_')
        
        # Determine point limit and variogram model
        # First check for variogram model in method name
        if 'exponential' in method:
            variogram_model = 'exponential'
        elif 'spherical' in method:
            variogram_model = 'spherical'
        elif 'linear' in method:
            variogram_model = 'linear'
        else:
            variogram_model = 'gaussian'  # Default
        
        # Then determine point limit
        if 'fast' in method:
            point_limit = MAX_KRIGING_POINTS_FAST
        elif 'medium' in method:
            point_limit = MAX_KRIGING_POINTS_MEDIUM
        elif 'high' in method:
            point_limit = MAX_KRIGING_POINTS_HIGH
        else:
            # Standard or auto methods
            point_limit = MAX_KRIGING_POINTS
            # Override variogram model for auto methods
            if 'auto' in method:
                variogram_model = 'auto'
        
        # Handle automatic variogram selection
        if variogram_model == 'auto':
            variogram_model, _ = select_best_variogram_model(longitudes, latitudes, field_values)
            logger.info(f"Auto-selected variogram model: {variogram_model}")
        
        logger.info(f"Using {point_limit:,} points for {method}")
        grid_field = ordinary_kriging_interpolation(longitudes, latitudes, field_values, 
                                                  grid_lon_mesh, grid_lat_mesh, variogram_model, point_limit, source_files)
    elif method.startswith('universal_kriging_'):
        variogram_model = method.split('_')[-1]  # Extract variogram model from method name
        
        # Handle automatic variogram selection
        if variogram_model == 'auto':
            variogram_model, _ = select_best_variogram_model(longitudes, latitudes, field_values)
            logger.info(f"Auto-selected variogram model: {variogram_model}")
        
        grid_field = universal_kriging_interpolation(longitudes, latitudes, field_values, 
                                                   grid_lon_mesh, grid_lat_mesh, variogram_model, source_files)
    elif method == 'minimum_curvature':
        grid_field = minimum_curvature_interpolation(longitudes, latitudes, field_values, 
                                                   grid_lon_mesh, grid_lat_mesh)
    else:
        logger.warning(f"Unknown method {method}, falling back to linear interpolation")
        grid_field = griddata(
            (longitudes, latitudes), 
            field_values, 
            (grid_lon_mesh, grid_lat_mesh), 
            method='linear', 
            fill_value=np.nan
        )
    
    # Apply boundary masking if enabled
    if ENABLE_BOUNDARY_MASKING:
        logger.info(f"Applying boundary masking using {BOUNDARY_METHOD} method")
        boundary_mask = create_boundary_mask(
            longitudes, latitudes, grid_lon_mesh, grid_lat_mesh, 
            method=BOUNDARY_METHOD, buffer_distance=BOUNDARY_BUFFER_DISTANCE
        )
        grid_field = apply_boundary_mask(grid_field, boundary_mask)
    
    return grid_lat_mesh, grid_lon_mesh, grid_field, mag_field_col

# =============================================
# VISUALIZATION FUNCTIONS
# =============================================

def create_contour_lines(grid_lat, grid_lon, grid_field, num_contours=10):
    """
    Create contour lines from interpolated grid data.
    
    Args:
        grid_lat, grid_lon: Coordinate meshgrids
        grid_field: Interpolated field values
        num_contours: Number of contour levels
    
    Returns:
        contour_features: List of GeoJSON features for contour lines
    """
    
    try:
        # Remove NaN values for contouring
        valid_mask = ~np.isnan(grid_field)
        if not np.any(valid_mask):
            return []
        
        # Get data range for contour levels
        field_min = np.nanmin(grid_field)
        field_max = np.nanmax(grid_field)
        
        if field_min >= field_max:
            return []
        
        # Create contour levels
        contour_levels = np.linspace(field_min, field_max, num_contours + 1)
        
        # Create matplotlib figure for contour calculation (not displayed)
        fig, ax = plt.subplots(figsize=(1, 1))
        ax.set_aspect('equal')
        
        # Generate contours
        cs = ax.contour(grid_lon, grid_lat, grid_field, levels=contour_levels)
        
        contour_features = []
        
        # Extract contour paths and convert to GeoJSON
        # Handle different matplotlib versions
        if hasattr(cs, 'collections'):
            collections = cs.collections
        else:
            # For newer matplotlib versions (QuadContourSet)
            collections = cs.allsegs if hasattr(cs, 'allsegs') else []
        
        if hasattr(cs, 'collections'):
            # Old matplotlib API
            for i, collection in enumerate(collections):
                level = contour_levels[i]
                
                # Determine color based on positive/negative
                if level >= 0:
                    color = '#FF0000'  # Red for positive
                    opacity = 0.8
                else:
                    color = '#0000FF'  # Blue for negative
                    opacity = 0.8
                
                # Extract paths from collection
                for path in collection.get_paths():
                    vertices = path.vertices
                    if len(vertices) > 2:  # Need at least 3 points for a line
                        # Convert to lat/lon coordinates (swap x,y to lon,lat)
                        coordinates = [[float(v[0]), float(v[1])] for v in vertices]
                        
                        # Create GeoJSON feature
                        feature = {
                            'type': 'Feature',
                            'properties': {
                                'level': float(level),
                                'color': color,
                                'opacity': opacity,
                                'weight': 2 if abs(level) < (field_max - field_min) * 0.1 else 1.5
                            },
                            'geometry': {
                                'type': 'LineString',
                                'coordinates': coordinates
                            }
                        }
                        contour_features.append(feature)
        else:
            # New matplotlib API - use allsegs
            for i, level_segs in enumerate(collections):
                if i >= len(contour_levels):
                    break
                level = contour_levels[i]
                
                # Determine color based on positive/negative
                if level >= 0:
                    color = '#FF0000'  # Red for positive
                    opacity = 0.8
                else:
                    color = '#0000FF'  # Blue for negative
                    opacity = 0.8
                
                # Extract segments for this level
                for seg in level_segs:
                    if len(seg) > 2:  # Need at least 3 points for a line
                        # Convert to lat/lon coordinates (swap x,y to lon,lat)
                        coordinates = [[float(v[0]), float(v[1])] for v in seg]
                        
                        # Create GeoJSON feature
                        feature = {
                            'type': 'Feature',
                            'properties': {
                                'level': float(level),
                                'color': color,
                                'opacity': opacity,
                                'weight': 2 if abs(level) < (field_max - field_min) * 0.1 else 1.5
                            },
                            'geometry': {
                                'type': 'LineString',
                                'coordinates': coordinates
                            }
                        }
                        contour_features.append(feature)
        
        plt.close(fig)  # Clean up matplotlib figure
        
        return contour_features
        
    except Exception as e:
        print(f"Error creating contour lines: {e}")
        return []

def create_raster_overlay(grid_field, vmin, vmax, alpha=0.7):
    """Create a raster overlay from interpolated data using viridis colormap"""
    
    # Normalize the data to 0-1 range for colormap
    normalized_data = (grid_field - vmin) / (vmax - vmin)
    
    # Handle NaN values
    mask = ~np.isnan(normalized_data)
    
    # Create RGBA image using viridis colormap
    import matplotlib.cm as cm
    viridis = cm.get_cmap('viridis')
    
    # Create RGBA array
    rgba_array = np.zeros((grid_field.shape[0], grid_field.shape[1], 4))
    
    # Apply colormap where data is valid
    rgba_array[mask] = viridis(normalized_data[mask])
    
    # Set alpha channel
    rgba_array[:, :, 3] = alpha * mask.astype(float)
    
    # Convert to uint8
    rgba_array = (rgba_array * 255).astype(np.uint8)
    
    # Flip the array vertically to match geographic coordinates
    rgba_array = np.flipud(rgba_array)
    
    # Create PIL Image
    img = Image.fromarray(rgba_array, 'RGBA')
    
    # Convert to base64 for embedding
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return img_str

def create_flight_path_features(df, mag_field_col, colormap_range):
    """
    Create colored flight path features grouped by source file.
    
    Args:
        df: DataFrame with flight data
        mag_field_col: Name of magnetic field column
        colormap_range: (vmin, vmax) for color scaling
    
    Returns:
        flight_path_features: List of GeoJSON features for flight paths
    """
    import matplotlib.cm as cm
    
    try:
        vmin, vmax = colormap_range
        
        # Create colormap
        cmap = cm.get_cmap('viridis')
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        
        flight_path_features = []
        total_segments_added = 0
        
        # Group by source file to create separate flight paths
        for source_file, group_df in df.groupby('Source_File'):
            # Sort by index to maintain flight order (assuming data is in flight order)
            group_df = group_df.sort_index()
            
            # Calculate step size to limit segments per flight
            max_segments_per_flight = MAX_FLIGHT_PATH_SEGMENTS // len(df.groupby('Source_File'))
            step_size = max(1, len(group_df) // max_segments_per_flight)
            
            # Create segments between consecutive points (with step size to reduce count)
            for i in range(0, len(group_df) - step_size, step_size):
                if total_segments_added >= MAX_FLIGHT_PATH_SEGMENTS:
                    break
                row1 = group_df.iloc[i]
                row2 = group_df.iloc[i + step_size]
                
                lat1, lon1 = row1['Latitude [Decimal Degrees]'], row1['Longitude [Decimal Degrees]']
                lat2, lon2 = row2['Latitude [Decimal Degrees]'], row2['Longitude [Decimal Degrees]']
                value1, value2 = row1[mag_field_col], row2[mag_field_col]
                
                # Skip if any coordinates or values are invalid
                if not all(np.isfinite([lat1, lon1, lat2, lon2, value1, value2])):
                    continue
                
                # Use average value for segment color
                avg_value = (value1 + value2) / 2
                
                # Get color from colormap
                color_rgba = cmap(norm(avg_value))
                color_hex = mcolors.rgb2hex(color_rgba[:3])
                
                # Create line segment
                feature = {
                    'type': 'Feature',
                    'properties': {
                        'source_file': source_file,
                        'avg_value': float(avg_value),
                        'color': color_hex,
                        'opacity': 0.8,
                        'weight': 3
                    },
                    'geometry': {
                        'type': 'LineString',
                        'coordinates': [[lon1, lat1], [lon2, lat2]]
                    }
                }
                flight_path_features.append(feature)
                total_segments_added += 1
            
            if total_segments_added >= MAX_FLIGHT_PATH_SEGMENTS:
                break
        
        return flight_path_features
        
    except Exception as e:
        print(f"Error creating flight path features: {e}")
        return []

def create_diagnostic_plots(df, interpolation_results, output_dir):
    """
    Create diagnostic plots for interpolation analysis.
    
    Args:
        df: Original data DataFrame
        interpolation_results: Dictionary of interpolation results
        output_dir: Output directory for plots
    """
    logger = logging.getLogger(__name__)
    
    try:
        import matplotlib.pyplot as plt
        
        # Try to import seaborn, but make it optional
        try:
            import seaborn as sns
            sns.set_palette("husl")
            has_seaborn = True
        except ImportError:
            logger.warning("Seaborn not available, using matplotlib defaults")
            has_seaborn = False
        
        # Set style for better plots
        plt.style.use('default')
        
        mag_field_col = get_magnetic_field_column(df)
        original_values = df[mag_field_col].values
        original_lats = df['Latitude [Decimal Degrees]'].values
        original_lons = df['Longitude [Decimal Degrees]'].values
        
        # Create plots directory
        plots_dir = output_dir / "diagnostic_plots"
        plots_dir.mkdir(exist_ok=True)
        
        # 1. Data distribution histogram
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.hist(original_values[~np.isnan(original_values)], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title(f'Original Data Distribution\n{mag_field_col}')
        plt.xlabel('Magnetic Field (nT)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # 2. Spatial distribution of data points
        plt.subplot(2, 2, 2)
        scatter = plt.scatter(original_lons, original_lats, c=original_values, 
                            cmap='viridis', s=1, alpha=0.6)
        plt.colorbar(scatter, label='Magnetic Field (nT)')
        plt.title('Spatial Distribution of Data Points')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.axis('equal')
        
        # 3. Data statistics by source file
        plt.subplot(2, 2, 3)
        source_stats = []
        for source_file in df['Source_File'].unique():
            file_data = df[df['Source_File'] == source_file][mag_field_col]
            source_stats.append({
                'File': source_file[:15],  # Truncate long names
                'Mean': np.nanmean(file_data),
                'Std': np.nanstd(file_data),
                'Count': len(file_data)
            })
        
        source_files = [s['File'] for s in source_stats]
        means = [s['Mean'] for s in source_stats]
        stds = [s['Std'] for s in source_stats]
        
        x = range(len(source_files))
        plt.errorbar(x, means, yerr=stds, fmt='o-', capsize=5, capthick=2)
        plt.xticks(x, source_files, rotation=45)
        plt.title('Statistics by Source File')
        plt.ylabel('Magnetic Field (nT)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 4. Interpolation method comparison
        plt.subplot(2, 2, 4)
        method_rmse = []
        method_names = []
        
        for method, (grid_lat, grid_lon, grid_field, _) in interpolation_results.items():
            # Calculate RMSE at data points
            interp_at_orig = griddata(
                (grid_lon.ravel(), grid_lat.ravel()), 
                grid_field.ravel(),
                (original_lons, original_lats), 
                method='linear',
                fill_value=np.nan
            )
            
            residuals = original_values - interp_at_orig
            valid_residuals = residuals[~np.isnan(residuals)]
            
            if len(valid_residuals) > 0:
                rmse = np.sqrt(np.mean(valid_residuals**2))
                method_rmse.append(rmse)
                method_names.append(method.replace('_', ' ').title())
        
        if method_rmse:
            plt.bar(range(len(method_names)), method_rmse, color='coral', alpha=0.7)
            plt.xticks(range(len(method_names)), method_names, rotation=45)
            plt.title('RMSE Comparison by Method')
            plt.ylabel('RMSE (nT)')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'data_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Residual analysis for each method
        for method, (grid_lat, grid_lon, grid_field, _) in interpolation_results.items():
            plt.figure(figsize=(15, 10))
            
            # Calculate residuals
            interp_at_orig = griddata(
                (grid_lon.ravel(), grid_lat.ravel()), 
                grid_field.ravel(),
                (original_lons, original_lats), 
                method='linear',
                fill_value=np.nan
            )
            
            residuals = original_values - interp_at_orig
            valid_mask = ~np.isnan(residuals)
            valid_residuals = residuals[valid_mask]
            valid_lons = original_lons[valid_mask]
            valid_lats = original_lats[valid_mask]
            
            if len(valid_residuals) > 0:
                # Residual histogram
                plt.subplot(2, 3, 1)
                plt.hist(valid_residuals, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
                plt.title(f'Residual Distribution\n{method.replace("_", " ").title()}')
                plt.xlabel('Residual (nT)')
                plt.ylabel('Frequency')
                plt.grid(True, alpha=0.3)
                
                # Q-Q plot
                plt.subplot(2, 3, 2)
                try:
                    from scipy import stats
                    stats.probplot(valid_residuals, dist="norm", plot=plt)
                    plt.title('Q-Q Plot (Normal Distribution)')
                except ImportError:
                    # Fallback: simple histogram if scipy not available
                    plt.hist(valid_residuals, bins=20, alpha=0.7, density=True, color='lightcoral', edgecolor='black')
                    plt.title('Residual Distribution (Normalized)')
                    plt.xlabel('Residual (nT)')
                    plt.ylabel('Density')
                plt.grid(True, alpha=0.3)
                
                # Spatial residuals
                plt.subplot(2, 3, 3)
                scatter = plt.scatter(valid_lons, valid_lats, c=valid_residuals, 
                                    cmap='RdBu_r', s=2, alpha=0.7)
                plt.colorbar(scatter, label='Residual (nT)')
                plt.title('Spatial Distribution of Residuals')
                plt.xlabel('Longitude')
                plt.ylabel('Latitude')
                plt.axis('equal')
                
                # Predicted vs Observed
                plt.subplot(2, 3, 4)
                predicted = interp_at_orig[valid_mask]
                observed = original_values[valid_mask]
                plt.scatter(predicted, observed, alpha=0.5, s=1)
                
                # Perfect prediction line
                min_val = min(np.min(predicted), np.min(observed))
                max_val = max(np.max(predicted), np.max(observed))
                plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 Line')
                
                plt.xlabel('Predicted (nT)')
                plt.ylabel('Observed (nT)')
                plt.title('Predicted vs Observed')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Residuals vs Predicted
                plt.subplot(2, 3, 5)
                plt.scatter(predicted, valid_residuals, alpha=0.5, s=1)
                plt.axhline(y=0, color='r', linestyle='--')
                plt.xlabel('Predicted (nT)')
                plt.ylabel('Residuals (nT)')
                plt.title('Residuals vs Predicted')
                plt.grid(True, alpha=0.3)
                
                # Statistics text
                plt.subplot(2, 3, 6)
                plt.axis('off')
                rmse = np.sqrt(np.mean(valid_residuals**2))
                mae = np.mean(np.abs(valid_residuals))
                r_squared = np.corrcoef(predicted, observed)[0, 1]**2
                
                stats_text = f"""
                Statistics for {method.replace('_', ' ').title()}:
                
                RMSE: {rmse:.2f} nT
                MAE: {mae:.2f} nT
                R²: {r_squared:.3f}
                
                Mean Residual: {np.mean(valid_residuals):.2f} nT
                Std Residual: {np.std(valid_residuals):.2f} nT
                
                Valid Points: {len(valid_residuals):,}
                Total Points: {len(original_values):,}
                """
                
                plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes, 
                        fontsize=12, verticalalignment='top', 
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(plots_dir / f'residual_analysis_{method}.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Diagnostic plots saved to {plots_dir}")
        
    except Exception as e:
        logger.error(f"Error creating diagnostic plots: {e}")

def create_field_visualization_plots(df, interpolation_results, output_dir):
    """
    Create field visualization PNGs with contour overlays.
    
    Args:
        df: Original data DataFrame
        interpolation_results: Dictionary of interpolation results
        output_dir: Output directory for plots
    """
    logger = logging.getLogger(__name__)
    
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        # Create field plots directory
        field_plots_dir = output_dir / "field_visualizations"
        field_plots_dir.mkdir(exist_ok=True)
        
        mag_field_col = get_magnetic_field_column(df)
        original_lats = df['Latitude [Decimal Degrees]'].values
        original_lons = df['Longitude [Decimal Degrees]'].values
        original_values = df[mag_field_col].values
        
        # Find global min/max for consistent color scaling
        all_values = original_values[~np.isnan(original_values)]
        for _, _, grid_field, _ in interpolation_results.values():
            valid_interp = grid_field[~np.isnan(grid_field) & np.isfinite(grid_field)]
            if len(valid_interp) > 0:
                all_values = np.concatenate([all_values, valid_interp.flatten()])
        
        global_min = np.min(all_values)
        global_max = np.max(all_values)
        
        # Create individual field plots for each method
        for method, (grid_lat, grid_lon, grid_field, _) in interpolation_results.items():
            fig, axes = plt.subplots(1, 3, figsize=(20, 6))
            
            # Plot 1: Interpolated field with contours
            ax1 = axes[0]
            
            # Create filled contour plot
            levels = np.linspace(global_min, global_max, 20)
            im1 = ax1.contourf(grid_lon, grid_lat, grid_field, levels=levels, 
                              cmap='viridis', extend='both')
            
            # Add contour lines
            contour_levels = np.linspace(global_min, global_max, 10)
            positive_levels = contour_levels[contour_levels >= 0]
            negative_levels = contour_levels[contour_levels < 0]
            
            if len(positive_levels) > 0:
                cs_pos = ax1.contour(grid_lon, grid_lat, grid_field, 
                                   levels=positive_levels, colors='red', linewidths=1.5, alpha=0.7)
                ax1.clabel(cs_pos, inline=True, fontsize=8, fmt='%d')
            
            if len(negative_levels) > 0:
                cs_neg = ax1.contour(grid_lon, grid_lat, grid_field, 
                                   levels=negative_levels, colors='blue', linewidths=1.5, alpha=0.7)
                ax1.clabel(cs_neg, inline=True, fontsize=8, fmt='%d')
            
            ax1.set_title(f'{method.replace("_", " ").title()}\nInterpolated Field with Contours')
            ax1.set_xlabel('Longitude')
            ax1.set_ylabel('Latitude')
            ax1.set_aspect('equal')
            
            # Add colorbar
            cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
            cbar1.set_label('Magnetic Field (nT)')
            
            # Plot 2: Original data points overlay
            ax2 = axes[1]
            
            # Background field
            ax2.contourf(grid_lon, grid_lat, grid_field, levels=levels, 
                        cmap='viridis', extend='both', alpha=0.7)
            
            # Original data points
            scatter = ax2.scatter(original_lons, original_lats, c=original_values, 
                                cmap='viridis', s=1, alpha=0.8, vmin=global_min, vmax=global_max)
            
            ax2.set_title(f'{method.replace("_", " ").title()}\nField + Original Data Points')
            ax2.set_xlabel('Longitude')
            ax2.set_ylabel('Latitude')
            ax2.set_aspect('equal')
            
            # Add colorbar
            cbar2 = plt.colorbar(scatter, ax=ax2, shrink=0.8)
            cbar2.set_label('Magnetic Field (nT)')
            
            # Plot 3: Contours only (clean view)
            ax3 = axes[2]
            
            # White background
            ax3.set_facecolor('white')
            
            # Just contour lines on white background
            if len(positive_levels) > 0:
                cs_pos_clean = ax3.contour(grid_lon, grid_lat, grid_field, 
                                         levels=positive_levels, colors='red', linewidths=2)
                ax3.clabel(cs_pos_clean, inline=True, fontsize=10, fmt='%d nT')
            
            if len(negative_levels) > 0:
                cs_neg_clean = ax3.contour(grid_lon, grid_lat, grid_field, 
                                         levels=negative_levels, colors='blue', linewidths=2)
                ax3.clabel(cs_neg_clean, inline=True, fontsize=10, fmt='%d nT')
            
            # Add zero contour if it exists
            zero_level = [0] if global_min <= 0 <= global_max else []
            if zero_level:
                cs_zero = ax3.contour(grid_lon, grid_lat, grid_field, 
                                    levels=zero_level, colors='black', linewidths=3)
                ax3.clabel(cs_zero, inline=True, fontsize=12, fmt='%d nT')
            
            ax3.set_title(f'{method.replace("_", " ").title()}\nContour Lines Only')
            ax3.set_xlabel('Longitude')
            ax3.set_ylabel('Latitude')
            ax3.set_aspect('equal')
            ax3.grid(True, alpha=0.3)
            
            # Add legend for contour colors
            red_patch = patches.Patch(color='red', label='Positive Field')
            blue_patch = patches.Patch(color='blue', label='Negative Field')
            black_patch = patches.Patch(color='black', label='Zero Level')
            ax3.legend(handles=[red_patch, blue_patch, black_patch], loc='upper right')
            
            plt.tight_layout()
            plt.savefig(field_plots_dir / f'field_visualization_{method}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        # Create a combined comparison plot
        n_methods = len(interpolation_results)
        if n_methods > 1:
            fig, axes = plt.subplots(2, min(n_methods, 3), figsize=(5*min(n_methods, 3), 10))
            if n_methods == 1:
                axes = [axes]
            elif n_methods <= 3:
                axes = axes.reshape(2, -1)
            
            for idx, (method, (grid_lat, grid_lon, grid_field, _)) in enumerate(interpolation_results.items()):
                if idx >= 6:  # Limit to 6 methods max
                    break
                    
                row = idx // 3
                col = idx % 3
                
                if n_methods <= 3:
                    ax_field = axes[0, col] if n_methods > 1 else axes[0]
                    ax_contour = axes[1, col] if n_methods > 1 else axes[1]
                else:
                    ax_field = axes[row, col] if row < 2 else None
                    ax_contour = None
                
                if ax_field is not None:
                    # Field plot
                    levels = np.linspace(global_min, global_max, 15)
                    im = ax_field.contourf(grid_lon, grid_lat, grid_field, levels=levels, 
                                         cmap='viridis', extend='both')
                    ax_field.set_title(f'{method.replace("_", " ").title()}')
                    ax_field.set_aspect('equal')
                    
                if ax_contour is not None:
                    # Contour plot
                    contour_levels = np.linspace(global_min, global_max, 8)
                    positive_levels = contour_levels[contour_levels >= 0]
                    negative_levels = contour_levels[contour_levels < 0]
                    
                    if len(positive_levels) > 0:
                        ax_contour.contour(grid_lon, grid_lat, grid_field, 
                                         levels=positive_levels, colors='red', linewidths=1.5)
                    if len(negative_levels) > 0:
                        ax_contour.contour(grid_lon, grid_lat, grid_field, 
                                         levels=negative_levels, colors='blue', linewidths=1.5)
                    
                    ax_contour.set_title(f'{method.replace("_", " ").title()} - Contours')
                    ax_contour.set_aspect('equal')
                    ax_contour.grid(True, alpha=0.3)
            
            # Remove empty subplots
            for idx in range(len(interpolation_results), 6):
                row = idx // 3
                col = idx % 3
                if row < 2 and col < 3:
                    if n_methods <= 3:
                        axes[0, col].remove() if n_methods > 1 else None
                        axes[1, col].remove() if n_methods > 1 else None
                    else:
                        if row < axes.shape[0] and col < axes.shape[1]:
                            axes[row, col].remove()
            
            plt.tight_layout()
            plt.savefig(field_plots_dir / 'method_comparison.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Field visualization plots saved to {field_plots_dir}")
        
    except Exception as e:
        logger.error(f"Error creating field visualization plots: {e}")

def create_noise_analysis_plots(df, output_dir):
    """
    Create comprehensive noise analysis plots for the magnetic field data.
    
    Args:
        df: Original data DataFrame
        output_dir: Output directory for plots
    """
    logger = logging.getLogger(__name__)
    
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy import signal
        from scipy.fft import fft, fftfreq
        
        # Create noise analysis directory
        noise_dir = output_dir / "noise_analysis"
        noise_dir.mkdir(exist_ok=True)
        
        mag_field_col = get_magnetic_field_column(df)
        
        logger.info("Analyzing noise characteristics for each flight...")
        
        # Analyze each flight separately
        for source_file in df['Source_File'].unique():
            flight_data = df[df['Source_File'] == source_file].copy()
            flight_data = flight_data.sort_index()  # Ensure chronological order
            
            # Get magnetic field data
            mag_values = flight_data[mag_field_col].values
            valid_mask = ~np.isnan(mag_values)
            
            if np.sum(valid_mask) < 100:  # Need sufficient data points
                logger.warning(f"Insufficient data points for {source_file}, skipping noise analysis")
                continue
            
            mag_clean = mag_values[valid_mask]
            n_points = len(mag_clean)
            
            # Estimate sampling rate (approximate, based on data spacing)
            if len(flight_data) > 1:
                # Use time differences if available, otherwise spatial spacing
                lats = flight_data['Latitude [Decimal Degrees]'].values[valid_mask]
                lons = flight_data['Longitude [Decimal Degrees]'].values[valid_mask]
                
                # Calculate approximate distances between consecutive points
                distances = np.sqrt(np.diff(lats)**2 + np.diff(lons)**2)
                mean_spacing = np.mean(distances[distances > 0]) * 111000  # Convert to meters (approx)
                
                # Assume typical aircraft speed for magnetic surveys (50-100 m/s)
                assumed_speed = 75  # m/s
                sampling_rate = assumed_speed / mean_spacing if mean_spacing > 0 else 1.0
            else:
                sampling_rate = 1.0  # Hz (fallback)
            
            # Create comprehensive noise analysis plot
            fig, axes = plt.subplots(3, 3, figsize=(18, 15))
            fig.suptitle(f'Noise Analysis: {source_file}', fontsize=16)
            
            # 1. Raw signal time series
            ax1 = axes[0, 0]
            time_axis = np.arange(n_points) / sampling_rate
            ax1.plot(time_axis, mag_clean, 'b-', linewidth=0.5)
            ax1.set_title('Raw Magnetic Field Signal')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Magnetic Field (nT)')
            ax1.grid(True, alpha=0.3)
            
            # 2. Detrended signal
            ax2 = axes[0, 1]
            # Remove linear trend
            detrended = signal.detrend(mag_clean, type='linear')
            ax2.plot(time_axis, detrended, 'g-', linewidth=0.5)
            ax2.set_title('Detrended Signal')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Detrended Field (nT)')
            ax2.grid(True, alpha=0.3)
            
            # 3. High-frequency components (approximate)
            ax3 = axes[0, 2]
            # Simple high-pass approximation using difference
            high_freq = np.diff(mag_clean)
            time_diff = time_axis[:-1]
            ax3.plot(time_diff, high_freq, 'r-', linewidth=0.5)
            ax3.set_title('High-Frequency Components')
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Field Difference (nT)')
            ax3.grid(True, alpha=0.3)
            
            # 4. Power Spectral Density
            ax4 = axes[1, 0]
            try:
                freqs, psd = signal.welch(detrended, fs=sampling_rate, nperseg=min(1024, len(detrended)//4))
                ax4.loglog(freqs[1:], psd[1:])  # Skip DC component
                ax4.set_title('Power Spectral Density')
                ax4.set_xlabel('Frequency (Hz)')
                ax4.set_ylabel('PSD (nT²/Hz)')
                ax4.grid(True, alpha=0.3)
                
                # Mark potential noise regions
                if len(freqs) > 10:
                    high_freq_threshold = sampling_rate * 0.1  # 10% of Nyquist
                    ax4.axvline(x=high_freq_threshold, color='r', linestyle='--', 
                               label=f'Potential noise > {high_freq_threshold:.2f} Hz')
                    ax4.legend()
            except Exception as e:
                ax4.text(0.5, 0.5, f'PSD calculation failed: {str(e)[:50]}', 
                        transform=ax4.transAxes, ha='center', va='center')
            
            # 5. FFT Magnitude
            ax5 = axes[1, 1]
            try:
                fft_vals = fft(detrended)
                fft_freqs = fftfreq(len(detrended), d=1/sampling_rate)
                fft_mag = np.abs(fft_vals)
                
                # Only plot positive frequencies
                pos_mask = fft_freqs > 0
                ax5.loglog(fft_freqs[pos_mask], fft_mag[pos_mask])
                ax5.set_title('FFT Magnitude Spectrum')
                ax5.set_xlabel('Frequency (Hz)')
                ax5.set_ylabel('Magnitude')
                ax5.grid(True, alpha=0.3)
            except Exception as e:
                ax5.text(0.5, 0.5, f'FFT calculation failed: {str(e)[:50]}', 
                        transform=ax5.transAxes, ha='center', va='center')
            
            # 6. Noise histogram
            ax6 = axes[1, 2]
            noise_estimate = detrended - signal.medfilt(detrended, kernel_size=min(51, len(detrended)//10))
            ax6.hist(noise_estimate, bins=50, alpha=0.7, color='orange', edgecolor='black')
            ax6.set_title(f'Noise Distribution\nStd: {np.std(noise_estimate):.2f} nT')
            ax6.set_xlabel('Noise Amplitude (nT)')
            ax6.set_ylabel('Frequency')
            ax6.grid(True, alpha=0.3)
            
            # 7. Autocorrelation
            ax7 = axes[2, 0]
            try:
                autocorr = np.correlate(detrended, detrended, mode='full')
                autocorr = autocorr[autocorr.size // 2:]
                autocorr = autocorr / autocorr[0]  # Normalize
                
                lags = np.arange(len(autocorr)) / sampling_rate
                max_lag = min(len(autocorr), int(sampling_rate * 10))  # Show up to 10 seconds
                
                ax7.plot(lags[:max_lag], autocorr[:max_lag])
                ax7.set_title('Autocorrelation Function')
                ax7.set_xlabel('Lag (s)')
                ax7.set_ylabel('Correlation')
                ax7.grid(True, alpha=0.3)
                ax7.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            except Exception as e:
                ax7.text(0.5, 0.5, f'Autocorr failed: {str(e)[:30]}', 
                        transform=ax7.transAxes, ha='center', va='center')
            
            # 8. Simulated Low-Pass Filter Effect
            ax8 = axes[2, 1]
            try:
                # Simulate different low-pass filter cutoffs
                cutoff_freqs = [0.1, 0.5, 1.0]  # Hz
                colors = ['red', 'blue', 'green']
                
                for cutoff, color in zip(cutoff_freqs, colors):
                    if cutoff < sampling_rate / 2:
                        sos = signal.butter(4, cutoff, btype='low', fs=sampling_rate, output='sos')
                        filtered_sim = signal.sosfilt(sos, detrended)
                        ax8.plot(time_axis[:len(filtered_sim)], filtered_sim + np.mean(mag_clean), 
                                color=color, linewidth=1, alpha=0.7, 
                                label=f'Low-pass {cutoff} Hz')
                
                ax8.plot(time_axis, mag_clean, 'k-', linewidth=0.5, alpha=0.5, label='Original')
                ax8.set_title('Simulated Low-Pass Filtering')
                ax8.set_xlabel('Time (s)')
                ax8.set_ylabel('Magnetic Field (nT)')
                ax8.legend()
                ax8.grid(True, alpha=0.3)
            except Exception as e:
                ax8.text(0.5, 0.5, f'Filter sim failed: {str(e)[:30]}', 
                        transform=ax8.transAxes, ha='center', va='center')
            
            # 9. Simulated High-Pass Filter Effect
            ax9 = axes[2, 2]
            try:
                # Simulate different high-pass filter cutoffs
                cutoff_freqs = [0.01, 0.05, 0.1]  # Hz
                
                for cutoff, color in zip(cutoff_freqs, colors):
                    if cutoff < sampling_rate / 2:
                        sos = signal.butter(4, cutoff, btype='high', fs=sampling_rate, output='sos')
                        filtered_sim = signal.sosfilt(sos, detrended)
                        ax9.plot(time_axis[:len(filtered_sim)], filtered_sim, 
                                color=color, linewidth=1, alpha=0.7, 
                                label=f'High-pass {cutoff} Hz')
                
                ax9.plot(time_axis, detrended, 'k-', linewidth=0.5, alpha=0.5, label='Detrended')
                ax9.set_title('Simulated High-Pass Filtering')
                ax9.set_xlabel('Time (s)')
                ax9.set_ylabel('Filtered Signal (nT)')
                ax9.legend()
                ax9.grid(True, alpha=0.3)
            except Exception as e:
                ax9.text(0.5, 0.5, f'Filter sim failed: {str(e)[:30]}', 
                        transform=ax9.transAxes, ha='center', va='center')
            
            plt.tight_layout()
            plt.savefig(noise_dir / f'noise_analysis_{source_file}.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Create combined noise statistics plot
        logger.info("Creating combined noise statistics...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Combined Noise Analysis - All Flights', fontsize=16)
        
        flight_stats = []
        all_noise_data = []
        
        for source_file in df['Source_File'].unique():
            flight_data = df[df['Source_File'] == source_file].copy()
            mag_values = flight_data[mag_field_col].values
            valid_mask = ~np.isnan(mag_values)
            
            if np.sum(valid_mask) < 100:
                continue
            
            mag_clean = mag_values[valid_mask]
            detrended = signal.detrend(mag_clean, type='linear')
            
            # Estimate noise
            noise_estimate = detrended - signal.medfilt(detrended, kernel_size=min(51, len(detrended)//10))
            
            flight_stats.append({
                'flight': source_file,
                'std_noise': np.std(noise_estimate),
                'rms_noise': np.sqrt(np.mean(noise_estimate**2)),
                'peak_to_peak': np.ptp(noise_estimate),
                'snr_estimate': np.std(mag_clean) / np.std(noise_estimate)
            })
            
            all_noise_data.extend(noise_estimate)
        
        # Plot 1: Noise statistics by flight
        ax1 = axes[0, 0]
        flights = [s['flight'] for s in flight_stats]
        noise_stds = [s['std_noise'] for s in flight_stats]
        
        bars = ax1.bar(range(len(flights)), noise_stds, color='skyblue', alpha=0.7)
        ax1.set_title('Noise Standard Deviation by Flight')
        ax1.set_xlabel('Flight')
        ax1.set_ylabel('Noise Std (nT)')
        ax1.set_xticks(range(len(flights)))
        ax1.set_xticklabels([f[:10] for f in flights], rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, std in zip(bars, noise_stds):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(noise_stds),
                    f'{std:.2f}', ha='center', va='bottom', fontsize=8)
        
        # Plot 2: Signal-to-Noise Ratio
        ax2 = axes[0, 1]
        snrs = [s['snr_estimate'] for s in flight_stats]
        bars2 = ax2.bar(range(len(flights)), snrs, color='lightcoral', alpha=0.7)
        ax2.set_title('Estimated Signal-to-Noise Ratio')
        ax2.set_xlabel('Flight')
        ax2.set_ylabel('SNR')
        ax2.set_xticks(range(len(flights)))
        ax2.set_xticklabels([f[:10] for f in flights], rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, snr in zip(bars2, snrs):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(snrs),
                    f'{snr:.1f}', ha='center', va='bottom', fontsize=8)
        
        # Plot 3: Combined noise distribution
        ax3 = axes[1, 0]
        if all_noise_data:
            ax3.hist(all_noise_data, bins=100, alpha=0.7, color='orange', edgecolor='black', density=True)
            ax3.set_title(f'Combined Noise Distribution\nStd: {np.std(all_noise_data):.2f} nT')
            ax3.set_xlabel('Noise Amplitude (nT)')
            ax3.set_ylabel('Density')
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Summary statistics table
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        if flight_stats:
            summary_text = "Noise Analysis Summary:\n\n"
            summary_text += f"Total Flights: {len(flight_stats)}\n"
            summary_text += f"Total Data Points: {len(df):,}\n\n"
            summary_text += f"Noise Statistics:\n"
            summary_text += f"Mean Noise Std: {np.mean(noise_stds):.2f} nT\n"
            summary_text += f"Max Noise Std: {np.max(noise_stds):.2f} nT\n"
            summary_text += f"Min Noise Std: {np.min(noise_stds):.2f} nT\n\n"
            summary_text += f"SNR Statistics:\n"
            summary_text += f"Mean SNR: {np.mean(snrs):.1f}\n"
            summary_text += f"Max SNR: {np.max(snrs):.1f}\n"
            summary_text += f"Min SNR: {np.min(snrs):.1f}\n\n"
            
            # Filtering recommendations
            summary_text += "Filtering Recommendations:\n"
            if np.mean(noise_stds) > 5:
                summary_text += "• High noise detected\n"
                summary_text += "• Consider low-pass filter < 1 Hz\n"
            else:
                summary_text += "• Moderate noise levels\n"
                summary_text += "• Consider low-pass filter < 2 Hz\n"
            
            if np.min(snrs) < 5:
                summary_text += "• Some flights have low SNR\n"
                summary_text += "• Consider aggressive filtering\n"
            
            ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, 
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(noise_dir / 'combined_noise_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Noise analysis plots saved to {noise_dir}")
        
    except ImportError as e:
        logger.error(f"Missing required module for noise analysis: {e}")
        logger.info("Install scipy for full noise analysis: pip install scipy")
    except Exception as e:
        logger.error(f"Error creating noise analysis plots: {e}")

def calculate_interpolation_statistics(df, interpolation_results):
    """Calculate statistics for each interpolation method"""
    logger = logging.getLogger(__name__)
    
    mag_field_col = get_magnetic_field_column(df)
    original_values = df[mag_field_col].values
    original_lats = df['Latitude [Decimal Degrees]'].values
    original_lons = df['Longitude [Decimal Degrees]'].values
    
    statistics = {}
    
    for method, (grid_lat, grid_lon, grid_field, _) in interpolation_results.items():
        if not np.all(np.isnan(grid_field)):
            # Calculate residuals at data points
            interp_at_orig = griddata(
                (grid_lon.ravel(), grid_lat.ravel()), 
                grid_field.ravel(),
                (original_lons, original_lats), 
                method='linear',
                fill_value=np.nan
            )
            
            residuals = original_values - interp_at_orig
            valid_residuals = residuals[~np.isnan(residuals)]
            
            if len(valid_residuals) > 0:
                rmse = np.sqrt(np.mean(valid_residuals**2))
                mean_residual = np.mean(valid_residuals)
                std_residual = np.std(valid_residuals)
                
                statistics[method] = {
                    'rmse': rmse,
                    'mean_residual': mean_residual,
                    'std_residual': std_residual,
                    'valid_points': len(valid_residuals),
                    'total_points': len(original_values)
                }
                
                logger.info(f"{method}: RMSE={rmse:.2f} nT, Mean residual={mean_residual:.2f} nT")
            else:
                statistics[method] = {'error': 'No valid residuals calculated'}
        else:
            statistics[method] = {'error': 'Interpolation failed'}
    
    return statistics

def create_interactive_map(df, interpolation_results, output_dir):
    """Create an interactive Folium map with interpolation layers"""
    logger = logging.getLogger(__name__)
    
    # Get map center from data
    center_lat = df['Latitude [Decimal Degrees]'].mean()
    center_lon = df['Longitude [Decimal Degrees]'].mean()
    
    # Initialize map with satellite base layer
    m = folium.Map(location=[center_lat, center_lon], zoom_start=14)
    
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

    # Add Mapbox satellite layer
    folium.TileLayer(
        tiles=f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/tiles/{{z}}/{{x}}/{{y}}?access_token={MAPBOX_TOKEN}",
        attr="Mapbox Attribution: &copy; <a href='https://www.mapbox.com/about/maps/'>Mapbox</a>",
        name="Satellite"
    ).add_to(m)
    
    # Find global min/max for consistent colormap
    mag_field_col = get_magnetic_field_column(df)
    all_values = []
    
    # Add original data values
    original_values = df[mag_field_col].values
    valid_original = original_values[~np.isnan(original_values)]
    if len(valid_original) > 0:
        all_values.extend(valid_original)
    
    # Add interpolated values
    for _, _, grid_field, _ in interpolation_results.values():
        valid_interp = grid_field[~np.isnan(grid_field) & np.isfinite(grid_field)]
        if len(valid_interp) > 0:
            all_values.extend(valid_interp.flatten())
    
    if len(all_values) == 0:
        logger.error("No valid values found for colormap")
        return None
    
    global_min = np.min(all_values)
    global_max = np.max(all_values)
    
    # Ensure min < max to avoid colormap issues
    if global_min >= global_max:
        global_max = global_min + 1.0
        logger.warning(f"Adjusted colormap range: {global_min:.2f} to {global_max:.2f}")
    
    logger.info(f"Colormap range: {global_min:.2f} to {global_max:.2f} nT")
    
    # Create colormap
    cm_field = linear.viridis.scale(global_min, global_max)
    cm_field.caption = f"Magnetic Field ({mag_field_col})"
    cm_field.add_to(m)
    
    # Method descriptions
    method_descriptions = {
        'ordinary_kriging_auto': 'Ordinary Kriging (Auto) - Automatic variogram model selection',
        'ordinary_kriging_fast': 'Ordinary Kriging (Fast) - Gaussian variogram, 5k points',
        'ordinary_kriging_medium': 'Ordinary Kriging (Medium) - Gaussian variogram, 15k points',
        'ordinary_kriging_high': 'Ordinary Kriging (High) - Gaussian variogram, 25k points',
        'ordinary_kriging_exponential_fast': 'Ordinary Kriging (Exponential, Fast) - 5k points',
        'ordinary_kriging_exponential_medium': 'Ordinary Kriging (Exponential, Medium) - 15k points',
        'ordinary_kriging_exponential_high': 'Ordinary Kriging (Exponential, High) - 25k points',
        'ordinary_kriging_spherical_fast': 'Ordinary Kriging (Spherical, Fast) - 5k points',
        'ordinary_kriging_spherical_medium': 'Ordinary Kriging (Spherical, Medium) - 15k points',
        'ordinary_kriging_linear_medium': 'Ordinary Kriging (Linear, Medium) - 15k points',
        'universal_kriging_auto': 'Universal Kriging (Auto) - Automatic variogram model selection',
        'universal_kriging_gaussian': 'Universal Kriging (Gaussian) - Kriging with trend modeling',
        'universal_kriging_spherical': 'Universal Kriging (Spherical) - Spherical variogram with trend',
        'minimum_curvature': 'Minimum Curvature - Smooth surface with minimal curvature'
    }
    
    # Add interpolation layers (both raster and contour)
    for method, (grid_lat, grid_lon, grid_field, _) in interpolation_results.items():
        logger.info(f"Adding {method} layers to map")
        
        try:
            # Create raster overlay
            img_str = create_raster_overlay(grid_field, global_min, global_max, alpha=0.8)
            
            # Define bounds
            bounds = [
                [grid_lat.min(), grid_lon.min()],  # Southwest corner
                [grid_lat.max(), grid_lon.max()]   # Northeast corner
            ]
            
            description = method_descriptions.get(method, f"{method.replace('_', ' ').title()} Interpolation")
            
            # Add raster overlay
            folium.raster_layers.ImageOverlay(
                image=f"data:image/png;base64,{img_str}",
                bounds=bounds,
                opacity=0.85,
                interactive=True,
                cross_origin=False,
                name=description
            ).add_to(m)
            
            # Create and add contour lines
            contour_features = create_contour_lines(grid_lat, grid_lon, grid_field, num_contours=15)
            
            if contour_features:
                # Create feature group for contours
                contour_group = folium.FeatureGroup(name=f"{description} - Contours")
                
                # Add each contour line
                for feature in contour_features:
                    geom = feature['geometry']
                    props = feature['properties']
                    
                    folium.PolyLine(
                        locations=[[coord[1], coord[0]] for coord in geom['coordinates']],  # Swap lon,lat to lat,lon
                        color=props['color'],
                        weight=props['weight'],
                        opacity=props['opacity'],
                        popup=f"Contour: {props['level']:.2f} nT",
                        tooltip=f"{props['level']:.2f} nT"
                    ).add_to(contour_group)
                
                contour_group.add_to(m)
                logger.info(f"Added {len(contour_features)} contour lines for {method}")
            
            logger.info(f"Successfully added {method} layers")
            
        except Exception as e:
            logger.error(f"Failed to create layers for {method}: {e}")
            continue
    
    # Add colored flight paths
    logger.info("Adding colored flight paths to map")
    try:
        flight_path_features = create_flight_path_features(df, mag_field_col, (global_min, global_max))
        
        if flight_path_features:
            # Group flight paths by source file
            flight_groups = {}
            for feature in flight_path_features:
                source_file = feature['properties']['source_file']
                if source_file not in flight_groups:
                    flight_groups[source_file] = folium.FeatureGroup(name=f"Flight Path: {source_file}")
                
                geom = feature['geometry']
                props = feature['properties']
                
                folium.PolyLine(
                    locations=[[coord[1], coord[0]] for coord in geom['coordinates']],  # Swap lon,lat to lat,lon
                    color=props['color'],
                    weight=props['weight'],
                    opacity=props['opacity'],
                    popup=f"Flight: {source_file}<br>Magnetic Field: {props['avg_value']:.2f} nT",
                    tooltip=f"{props['avg_value']:.2f} nT"
                ).add_to(flight_groups[source_file])
            
            # Add all flight path groups to map
            for group in flight_groups.values():
                group.add_to(m)
            
            logger.info(f"Added {len(flight_path_features)} flight path segments from {len(flight_groups)} flights")
        
    except Exception as e:
        logger.error(f"Failed to create flight path visualization: {e}")
    
    # Add original data points
    logger.info("Adding original data points to map")
    data_points = folium.FeatureGroup(name="Original Data Points")
    
    # Subsample for performance if too many points
    df_display = df
    if len(df) > 1000:
        subsample_rate = len(df) // 1000
        df_display = df.iloc[::subsample_rate]
        logger.info(f"Subsampled data points for display: {len(df_display)} of {len(df)}")
    
    for _, row in df_display.iterrows():
        lat = row['Latitude [Decimal Degrees]']
        lon = row['Longitude [Decimal Degrees]']
        value = row[mag_field_col]
        
        # Skip invalid values
        if not (np.isfinite(lat) and np.isfinite(lon) and np.isfinite(value)):
            continue
            
        # Ensure value is within colormap range
        if value < global_min or value > global_max:
            continue
        
        try:
            # Color based on field value
            color = cm_field(value)
            
            folium.CircleMarker(
                location=[lat, lon],
                radius=3,
                color='white',
                weight=1,
                fill=True,
                fill_color=color,
                fill_opacity=0.8,
                popup=f'{mag_field_col}: {value:.2f} nT<br>Lat: {lat:.6f}<br>Lon: {lon:.6f}<br>Source: {row["Source_File"]}',
                tooltip=f'{value:.2f} nT'
            ).add_to(data_points)
        except Exception as e:
            logger.warning(f"Skipping invalid data point: value={value}, error={e}")
            continue
    
    data_points.add_to(m)
    
    # Add custom legend for contour colors
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 150px; height: 90px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <p><b>Contour Legend</b></p>
    <p><i class="fa fa-minus" style="color:red"></i> Positive Field</p>
    <p><i class="fa fa-minus" style="color:blue"></i> Negative Field</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add fullscreen button
    from folium.plugins import Fullscreen
    Fullscreen().add_to(m)
    
    # Save map
    output_path = output_dir / OUTPUT_HTML_FILENAME
    m.save(output_path)
    
    logger.info(f"Interactive map saved to {output_path}")
    return output_path

# =============================================
# MAIN PROCESSING FUNCTION
# =============================================

def main():
    """Main execution function"""
    logger = setup_logging()
    logger.info("Starting Grid Interpolation and Interactive Mapping")
    
    # Validate input directory
    input_path = Path(INPUT_DIRECTORY)
    if not input_path.exists():
        logger.error(f"Input directory not found: {INPUT_DIRECTORY}")
        logger.error("Please update the INPUT_DIRECTORY variable in the script configuration")
        return
    
    # Load and combine data
    df = load_and_combine_csv_files(INPUT_DIRECTORY)
    if df is None:
        return
    
    # Validate data
    if not validate_data(df):
        return
    
    # Clean data - remove any rows with NaN in critical columns
    mag_field_col = get_magnetic_field_column(df)
    initial_count = len(df)
    df = df.dropna(subset=['Latitude [Decimal Degrees]', 'Longitude [Decimal Degrees]', mag_field_col])
    final_count = len(df)
    
    if final_count < initial_count:
        logger.info(f"Removed {initial_count - final_count} rows with missing data. {final_count} rows remaining.")
    
    if final_count == 0:
        logger.error("No valid data points remaining after cleaning")
        return
    
    # Create output directory
    output_dir = input_path / "interpolation_output"
    output_dir.mkdir(exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Store interpolation results
    interpolation_results = {}
    
    # Get magnetic field column name
    mag_field_col = get_magnetic_field_column(df)
    
    # Process each interpolation method
    for method in INTERPOLATION_METHODS:
        if SKIP_INTERPOLATION:
            # Try to load existing grid from CSV
            logger.info(f"Loading existing {method} grid from CSV...")
            result = load_existing_grid(output_dir, method, mag_field_col)
            
            if result is not None:
                grid_lat, grid_lon, grid_field, mag_col = result
                interpolation_results[method] = (grid_lat, grid_lon, grid_field, mag_col)
                logger.info(f"Successfully loaded {method} grid from existing CSV")
            else:
                logger.warning(f"Could not load existing {method} grid, skipping...")
                continue
        else:
            # Perform interpolation
            logger.info(f"Performing {method} interpolation...")
            
            try:
                grid_lat, grid_lon, grid_field, mag_col = interpolate_magnetic_field(df, method)
                
                if grid_field is not None and not np.all(np.isnan(grid_field)):
                    interpolation_results[method] = (grid_lat, grid_lon, grid_field, mag_col)
                    
                    # Save grid as CSV for further analysis
                    grid_df = pd.DataFrame({
                        'Latitude': grid_lat.ravel(),
                        'Longitude': grid_lon.ravel(),
                        f'{mag_col}_interpolated': grid_field.ravel()
                    })
                    grid_df = grid_df.dropna()  # Remove NaN points
                    
                    grid_output_path = output_dir / f"grid_{method}_{mag_col.replace(' ', '_').replace('[', '').replace(']', '')}.csv"
                    grid_df.to_csv(grid_output_path, index=False)
                    logger.info(f"Saved interpolated grid to {grid_output_path}")
                    
                    logger.info(f"Successfully completed {method} interpolation")
                else:
                    logger.warning(f"{method} interpolation failed - all values are NaN")
                    
            except Exception as e:
                logger.error(f"Error in {method} interpolation: {e}")
                continue
    
    # Create visualizations if we have successful interpolations
    if interpolation_results:
        logger.info("Calculating interpolation statistics...")
        statistics = calculate_interpolation_statistics(df, interpolation_results)
        
        # Generate diagnostic plots if enabled
        if GENERATE_DIAGNOSTIC_PLOTS:
            logger.info("Creating diagnostic plots...")
            create_diagnostic_plots(df, interpolation_results, output_dir)
        
        # Generate field visualization plots if enabled
        if GENERATE_FIELD_PLOTS:
            logger.info("Creating field visualization plots...")
            create_field_visualization_plots(df, interpolation_results, output_dir)
        
        # Generate noise analysis plots if enabled
        if GENERATE_NOISE_ANALYSIS:
            logger.info("Creating noise analysis plots...")
            create_noise_analysis_plots(df, output_dir)
        
        logger.info("Creating interactive map...")
        if SPLIT_HTML_BY_METHOD:
            # Create separate HTML files for each method
            map_paths = []
            for method, result in interpolation_results.items():
                single_method_results = {method: result}
                method_filename = f"{method}_magnetic_field_map.html"
                method_output_dir = output_dir
                
                # Temporarily change output filename
                original_filename = OUTPUT_HTML_FILENAME
                globals()['OUTPUT_HTML_FILENAME'] = method_filename
                
                method_path = create_interactive_map(df, single_method_results, method_output_dir)
                map_paths.append(method_path)
                
                # Restore original filename
                globals()['OUTPUT_HTML_FILENAME'] = original_filename
                
            map_path = map_paths[0] if map_paths else None  # Return first one for logging
            logger.info(f"Created {len(map_paths)} separate HTML files")
        else:
            # Create combined HTML file
            map_path = create_interactive_map(df, interpolation_results, output_dir)
        
        # Create summary report
        summary_path = output_dir / "interpolation_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("Magnetic Field Grid Interpolation Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Input directory: {INPUT_DIRECTORY}\n")
            f.write(f"Combined data points: {len(df)}\n")
            f.write(f"Magnetic field column: {mag_field_col}\n")
            f.write(f"Grid resolution: {GRID_RESOLUTION} x {GRID_RESOLUTION}\n")
            f.write(f"Max kriging points (downsampled): {MAX_KRIGING_POINTS}\n\n")
            
            f.write("Source files:\n")
            for source_file in df['Source_File'].unique():
                count = len(df[df['Source_File'] == source_file])
                f.write(f"  - {source_file}: {count} points\n")
            
            f.write(f"\nInterpolation methods completed: {len(interpolation_results)}\n")
            for method in interpolation_results.keys():
                f.write(f"  - {method}\n")
            
            # Add statistics section
            f.write("\nInterpolation Statistics:\n")
            for method, stats in statistics.items():
                if 'error' in stats:
                    f.write(f"  {method}: {stats['error']}\n")
                else:
                    f.write(f"  {method}:\n")
                    f.write(f"    - RMSE: {stats['rmse']:.2f} nT\n")
                    f.write(f"    - Mean residual: {stats['mean_residual']:.2f} nT\n")
                    f.write(f"    - Std residual: {stats['std_residual']:.2f} nT\n")
                    f.write(f"    - Valid points: {stats['valid_points']}/{stats['total_points']}\n")
            
            f.write("\nOutputs created:\n")
            f.write(f"  - Interactive map: {map_path.name}\n")
            for method in interpolation_results.keys():
                f.write(f"  - Grid data ({method}): grid_{method}_{mag_field_col.replace(' ', '_').replace('[', '').replace(']', '')}.csv\n")
        
        logger.info(f"Summary report saved to {summary_path}")
        
        logger.info("Grid interpolation and mapping completed successfully!")
        logger.info(f"Interactive map: {map_path}")
        logger.info(f"All outputs saved to: {output_dir}")
        
    else:
        logger.error("No successful interpolations - no outputs created")

if __name__ == "__main__":
    main()