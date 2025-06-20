import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from branca.colormap import linear
from scipy.interpolate import griddata, RBFInterpolator
from scipy.spatial import cKDTree, ConvexHull
from scipy.ndimage import gaussian_filter
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import folium
import branca
from pathlib import Path
import logging
import base64
from io import BytesIO
from PIL import Image
import matplotlib.path
import warnings
warnings.filterwarnings('ignore')

# =============================================
# CONFIGURATION
# =============================================

# Input directory containing the CSV files
INPUT_DIRECTORY = "/Users/aleksandergarbuz/Documents/SINTEF/data/20250611_081139_MWALK_#0122_processed_20250616_094044_segmented/"  # Update this with your directory path

# Interpolation settings
GRID_RESOLUTION = 300  # Increased resolution for denser field (number of points per axis)
INTERPOLATION_METHODS = [
    # 'linear', 'nearest', 'cubic',  # Standard methods
    # 'parallel_lines_rbf',  # RBF optimized for parallel lines
    'directional_kriging',  # Gaussian process with directional kernel
    'edge_preserving',     # Edge-aware interpolation with masking
    # 'multiscale_rbf',      # Multi-scale RBF for better edge handling
    # 'anisotropic_kernel',  # Anisotropic kernel for flight line geometry
    'adaptive_grid'       # Adaptive grid resolution based on flight line direction
    # 'high_quality_rbf'     # High-quality RBF with optimal parameters for magnetic data
]  

SHOW_PLOTS = True  # Whether to display plots
MAPBOX_TOKEN = "pk.eyJ1IjoiYXRnMjE3IiwiYSI6ImNtYzBnY2kwOTAxbWwybHM3NmN0bnRlaWcifQ.B8hh4dBszYXxlj-O0KGqkg"  # Replace with your Mapbox token

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
# LOAD DATA FROM CSV
# =============================================

def load_data_from_csv(directory):
    """Load all CSV files from the specified directory into a single DataFrame"""
    logger = logging.getLogger(__name__)
    
    csv_files = list(Path(directory).glob("*.csv"))
    if not csv_files:
        logger.error("No CSV files found in the directory.")
        return None
    
    logger.info(f"Loading {len(csv_files)} CSV files from {directory}")
    
    # Combine all CSVs into one DataFrame
    df_list = []
    for file in csv_files:
        df = pd.read_csv(file)
        df['Filename'] = file.stem  # Add filename as a column
        df_list.append(df)
    
    return pd.concat(df_list, ignore_index=True)

# =============================================
# INTERPOLATION METHODS
# =============================================

def interpolate_full_field(df, method='linear'):
    """Interpolate the full field by connecting parallel lines using various advanced interpolation methods"""
    
    # Extract the relevant columns
    latitudes = df['Latitude [Decimal Degrees]'].values
    longitudes = df['Longitude [Decimal Degrees]'].values
    field_values = df['R1 [nT]'].values if 'R1 [nT]' in df.columns else df['Btotal1 [nT]'].values

    # Create a grid for interpolation
    grid_lat = np.linspace(latitudes.min(), latitudes.max(), GRID_RESOLUTION)
    grid_lon = np.linspace(longitudes.min(), longitudes.max(), GRID_RESOLUTION)
    grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)
    
    # Choose interpolation method
    if method in ['linear', 'nearest', 'cubic']:
        # Standard scipy griddata methods
        grid_field = griddata(
            (longitudes, latitudes), 
            field_values, 
            (grid_lon, grid_lat), 
            method=method, 
            fill_value=np.nan
        )
    elif method == 'parallel_lines_rbf':
        grid_field = parallel_lines_rbf_interpolation(longitudes, latitudes, field_values, grid_lon, grid_lat)
    elif method == 'directional_kriging':
        grid_field = directional_kriging_interpolation(longitudes, latitudes, field_values, grid_lon, grid_lat)
    elif method == 'edge_preserving':
        grid_field = edge_preserving_interpolation(longitudes, latitudes, field_values, grid_lon, grid_lat)
    elif method == 'multiscale_rbf':
        grid_field = multiscale_rbf_interpolation(longitudes, latitudes, field_values, grid_lon, grid_lat)
    elif method == 'anisotropic_kernel':
        grid_field = anisotropic_kernel_interpolation(longitudes, latitudes, field_values, grid_lon, grid_lat)
    elif method == 'directional_smoothing':
        grid_field = directional_smoothing_interpolation(longitudes, latitudes, field_values, grid_lon, grid_lat)
    elif method == 'adaptive_grid':
        grid_field = adaptive_grid_interpolation(longitudes, latitudes, field_values, grid_lon, grid_lat)
    elif method == 'high_quality_rbf':
        grid_field = high_quality_rbf_interpolation(longitudes, latitudes, field_values, grid_lon, grid_lat)
    else:
        # Fallback to linear interpolation
        grid_field = griddata(
            (longitudes, latitudes), 
            field_values, 
            (grid_lon, grid_lat), 
            method='linear', 
            fill_value=np.nan
        )
    
    return grid_lat, grid_lon, grid_field


def parallel_lines_rbf_interpolation(x, y, z, grid_x, grid_y):
    """RBF interpolation optimized for parallel flight lines"""
    try:
        # Clean input data - remove NaN values
        valid_mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
        if not np.any(valid_mask):
            return np.full(grid_x.shape, np.nan)
        
        x_clean = x[valid_mask]
        y_clean = y[valid_mask] 
        z_clean = z[valid_mask]
        
        # Limit dataset size for computational efficiency
        max_points = 2000
        if len(x_clean) > max_points:
            indices = np.random.choice(len(x_clean), max_points, replace=False)
            x_clean = x_clean[indices]
            y_clean = y_clean[indices]
            z_clean = z_clean[indices]
        
        # Calculate appropriate epsilon based on data scale (for future use)
        # x_range = np.ptp(x_clean)
        # y_range = np.ptp(y_clean)
        # epsilon = min(x_range, y_range) * 0.1  # 10% of smaller coordinate range
        
        # Use thin_plate_spline which is more stable than multiquadric
        # Adjust smoothing based on data noise level
        z_std = np.std(z_clean)
        adaptive_smoothing = min(0.01, z_std * 0.001)  # Very low smoothing for high quality
        
        rbf_interpolator = RBFInterpolator(
            np.column_stack([x_clean, y_clean]), z_clean, 
            kernel='thin_plate_spline',
            smoothing=adaptive_smoothing,  # Adaptive smoothing for better fit
        )
        
        # Interpolate to grid
        grid_z = rbf_interpolator(np.column_stack([grid_x.ravel(), grid_y.ravel()]))
        return grid_z.reshape(grid_x.shape)
        
    except MemoryError:
        print("MemoryError in RBF interpolation - falling back to linear")
        return griddata((x, y), z, (grid_x, grid_y), method='linear', fill_value=np.nan)
    except np.linalg.LinAlgError:
        print("Linear algebra error in RBF interpolation - falling back to linear")  
        return griddata((x, y), z, (grid_x, grid_y), method='linear', fill_value=np.nan)
    except Exception as e:
        print(f"Unexpected error in RBF interpolation: {e} - falling back to linear")
        # Fallback to linear interpolation
        return griddata((x, y), z, (grid_x, grid_y), method='linear', fill_value=np.nan)


def directional_kriging_interpolation(x, y, z, grid_x, grid_y):
    """Gaussian Process regression with directional kernel for flight line patterns"""
    try:
        # Clean input data - remove NaN values
        valid_mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
        if not np.any(valid_mask):
            return np.full(grid_x.shape, np.nan)
        
        x_clean = x[valid_mask]
        y_clean = y[valid_mask] 
        z_clean = z[valid_mask]
        
        # Determine flight line direction by analyzing data distribution
        x_range = np.ptp(x_clean)
        y_range = np.ptp(y_clean)
        
        # Create anisotropic kernel based on flight line direction
        if x_range > y_range:
            # More variation in x-direction (east-west flight lines)
            length_scale = [x_range * 0.1, y_range * 0.5]
        else:
            # More variation in y-direction (north-south flight lines)
            length_scale = [x_range * 0.5, y_range * 0.1]
        
        kernel = RBF(length_scale=length_scale, length_scale_bounds=(1e-5, 1e5)) + WhiteKernel(noise_level=1.0)
        
        # Aggressive sampling for computational efficiency (GP is O(n³))
        n_samples = min(len(x_clean), 5000)  # Reduced from 1000 for memory efficiency
        if len(x_clean) > n_samples:
            indices = np.random.choice(len(x_clean), n_samples, replace=False)
            x_sample = x_clean[indices]
            y_sample = y_clean[indices]
            z_sample = z_clean[indices]
        else:
            x_sample = x_clean
            y_sample = y_clean
            z_sample = z_clean
        
        print(f"Directional kriging with {len(x_sample)} sample points...")
        
        gpr = GaussianProcessRegressor(
            kernel=kernel, 
            alpha=1e-6, 
            normalize_y=True,
            n_restarts_optimizer=0  # Disable optimization restarts for speed
        )
        gpr.fit(np.column_stack([x_sample, y_sample]), z_sample)
        
        # Predict on grid
        grid_z, _ = gpr.predict(np.column_stack([grid_x.ravel(), grid_y.ravel()]), return_std=True)
        print("Directional kriging completed successfully")
        return grid_z.reshape(grid_x.shape)
        
    except MemoryError:
        print("MemoryError in directional kriging - falling back to linear")
        return griddata((x, y), z, (grid_x, grid_y), method='linear', fill_value=np.nan)
    except Exception as e:
        print(f"Error in directional kriging: {e} - falling back to linear")
        return griddata((x, y), z, (grid_x, grid_y), method='linear', fill_value=np.nan)


def edge_preserving_interpolation(x, y, z, grid_x, grid_y):
    """Edge-aware interpolation with convex hull masking and distance-weighted smoothing"""
    try:
        # First, perform standard interpolation
        grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic', fill_value=np.nan)
        
        # Create convex hull of data points to mask extrapolation
        try:
            points = np.column_stack([x, y])
            hull = ConvexHull(points)
            hull_path = plt.matplotlib.path.Path(points[hull.vertices])
            
            # Create mask for points inside convex hull
            grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
            inside_hull = hull_path.contains_points(grid_points)
            mask = inside_hull.reshape(grid_x.shape)
            
            # Apply mask
            grid_z[~mask] = np.nan
        except Exception:
            pass  # Continue without hull masking if it fails
        
        # Apply edge-preserving smoothing using distance weighting
        tree = cKDTree(np.column_stack([x, y]))
        
        # For each grid point, find nearby data points and apply distance weighting
        valid_mask = ~np.isnan(grid_z)
        for i in range(grid_x.shape[0]):
            for j in range(grid_x.shape[1]):
                if valid_mask[i, j]:
                    # Find nearest neighbors
                    distances, indices = tree.query([grid_x[i, j], grid_y[i, j]], k=min(10, len(x)))
                    
                    # Apply distance-weighted averaging with nearby points
                    if np.max(distances) > 0:
                        weights = 1.0 / (distances + 1e-8)
                        weights /= np.sum(weights)
                        grid_z[i, j] = np.sum(weights * z[indices])
        
        return grid_z
    except Exception:
        return griddata((x, y), z, (grid_x, grid_y), method='linear', fill_value=np.nan)


def multiscale_rbf_interpolation(x, y, z, grid_x, grid_y):
    """Multi-scale RBF interpolation combining different scales for better edge handling"""
    try:
        # Clean input data - remove NaN values
        valid_mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
        if not np.any(valid_mask):
            return np.full(grid_x.shape, np.nan)
        
        x_clean = x[valid_mask]
        y_clean = y[valid_mask] 
        z_clean = z[valid_mask]
        
        # Aggressive dataset limiting for multi-scale approach
        max_points = 10000  # Reduced from 2000 since we're doing multiple scales
        if len(x_clean) > max_points:
            indices = np.random.choice(len(x_clean), max_points, replace=False)
            x_clean = x_clean[indices]
            y_clean = y_clean[indices]
            z_clean = z_clean[indices]
        
        # Use fewer scales and simpler approach for memory efficiency
        scales = [0.5, 1.5]  # Reduced from 3 scales to 2
        weights = [0.4, 0.6]  # Weights for combining scales
        
        combined_result = np.zeros_like(grid_x, dtype=float)
        total_weight = 0
        successful_scales = 0
        
        # Pre-calculate distance weights once
        tree = cKDTree(np.column_stack([x_clean, y_clean]))
        distances, _ = tree.query(np.column_stack([grid_x.ravel(), grid_y.ravel()]), k=1)
        distance_weights = 1.0 / (distances.reshape(grid_x.shape) + 1e-6)
        distance_weights /= np.max(distance_weights)
        
        for scale, weight in zip(scales, weights):
            try:
                print(f"Processing scale {scale} with {len(x_clean)} points...")
                
                # Adaptive smoothing for higher quality
                z_std = np.std(z_clean)
                adaptive_smoothing = min(0.05, z_std * 0.001) * scale
                
                rbf = RBFInterpolator(
                    np.column_stack([x_clean, y_clean]), z_clean,
                    kernel='thin_plate_spline',
                    smoothing=adaptive_smoothing,  # Adaptive smoothing for better quality
                )
                
                scale_result = rbf(np.column_stack([grid_x.ravel(), grid_y.ravel()]))
                scale_result = scale_result.reshape(grid_x.shape)
                
                combined_result += weight * scale_result * distance_weights
                total_weight += weight * distance_weights
                successful_scales += 1
                
                print(f"Scale {scale} completed successfully")
                
            except MemoryError:
                print(f"MemoryError at scale {scale} - skipping")
                continue
            except Exception as e:
                print(f"Error at scale {scale}: {e} - skipping")
                continue
        
        # Check if we have any successful scales
        if successful_scales == 0:
            print("All scales failed - falling back to simple linear interpolation")
            return griddata((x, y), z, (grid_x, grid_y), method='linear', fill_value=np.nan)
        
        # Normalize by total weights
        valid_mask = total_weight > 0
        combined_result[valid_mask] /= total_weight[valid_mask]
        combined_result[np.logical_not(valid_mask)] = np.nan
        
        print(f"Multiscale RBF completed with {successful_scales}/{len(scales)} successful scales")
        return combined_result
        
    except MemoryError:
        print("MemoryError in multiscale RBF interpolation - falling back to linear")
        return griddata((x, y), z, (grid_x, grid_y), method='linear', fill_value=np.nan)
    except Exception as e:
        print(f"Unexpected error in multiscale RBF interpolation: {e} - falling back to linear")
        return griddata((x, y), z, (grid_x, grid_y), method='linear', fill_value=np.nan)


def anisotropic_kernel_interpolation(x, y, z, grid_x, grid_y):
    """Anisotropic kernel interpolation adapted to flight line geometry"""
    try:
        # Clean input data - remove NaN values
        valid_mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
        if not np.any(valid_mask):
            return np.full(grid_x.shape, np.nan)
        
        x_clean = x[valid_mask]
        y_clean = y[valid_mask] 
        z_clean = z[valid_mask]
        
        # Limit dataset size for computational efficiency
        max_points = 1500
        if len(x_clean) > max_points:
            indices = np.random.choice(len(x_clean), max_points, replace=False)
            x_clean = x_clean[indices]
            y_clean = y_clean[indices]
            z_clean = z_clean[indices]
        
        print(f"Anisotropic kernel with {len(x_clean)} points...")
        
        # Analyze flight line pattern to determine anisotropy
        coords = np.column_stack([x_clean, y_clean])
        centered_coords = coords - np.mean(coords, axis=0)
        
        # Compute covariance matrix and eigenvalues/eigenvectors
        cov_matrix = np.cov(centered_coords.T)
        eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
        
        # Check for numerical stability
        if eigenvals[0] <= 1e-10:
            print("Eigenvalues too small - falling back to simple RBF")
            rbf = RBFInterpolator(
                coords, z_clean,
                kernel='thin_plate_spline',
                smoothing=0.05
            )
            grid_z = rbf(np.column_stack([grid_x.ravel(), grid_y.ravel()]))
            return grid_z.reshape(grid_x.shape)
        
        # Determine anisotropy ratio and principal directions
        anisotropy_ratio = np.sqrt(eigenvals[1] / eigenvals[0])
        
        # Create anisotropic RBF with orientation based on flight lines
        # Transform coordinates to align with principal directions
        rotation_matrix = eigenvecs
        transformed_coords = centered_coords @ rotation_matrix
        
        # Scale coordinates to account for anisotropy (limit extreme ratios)
        anisotropy_ratio = np.clip(anisotropy_ratio, 0.1, 10.0)  # Prevent extreme scaling
        scale_factors = [1.0, anisotropy_ratio]
        transformed_coords[:, 0] *= scale_factors[0]
        transformed_coords[:, 1] *= scale_factors[1]
        
        # Perform RBF interpolation in transformed space
        # Adaptive smoothing for better quality
        z_std = np.std(z_clean)
        adaptive_smoothing = min(0.01, z_std * 0.001)
        
        rbf = RBFInterpolator(
            transformed_coords, z_clean,
            kernel='thin_plate_spline',  # More stable than multiquadric
            smoothing=adaptive_smoothing
        )
        
        # Transform grid coordinates
        grid_coords = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        centered_grid = grid_coords - np.mean(coords, axis=0)
        transformed_grid = centered_grid @ rotation_matrix
        transformed_grid[:, 0] *= scale_factors[0]
        transformed_grid[:, 1] *= scale_factors[1]
        
        # Interpolate
        grid_z = rbf(transformed_grid)
        print("Anisotropic kernel completed successfully")
        return grid_z.reshape(grid_x.shape)
        
    except MemoryError:
        print("MemoryError in anisotropic kernel - falling back to linear")
        return griddata((x, y), z, (grid_x, grid_y), method='linear', fill_value=np.nan)
    except np.linalg.LinAlgError:
        print("Linear algebra error in anisotropic kernel - falling back to linear")
        return griddata((x, y), z, (grid_x, grid_y), method='linear', fill_value=np.nan)
    except Exception as e:
        print(f"Error in anisotropic kernel: {e} - falling back to linear")
        return griddata((x, y), z, (grid_x, grid_y), method='linear', fill_value=np.nan)


def directional_smoothing_interpolation(x, y, z, grid_x, grid_y):
    """Directional smoothing interpolation using principal direction analysis"""
    try:
        # Analyze flight line pattern to determine principal direction
        coords = np.column_stack([x, y])
        centered_coords = coords - np.mean(coords, axis=0)
        
        # Compute covariance matrix and eigenvalues/eigenvectors
        cov_matrix = np.cov(centered_coords.T)
        _, eigenvecs = np.linalg.eigh(cov_matrix)
        
        # Principal direction (direction of maximum variance - along flight lines)
        principal_direction = eigenvecs[:, 1]
        
        # First, perform standard RBF interpolation with adaptive smoothing
        z_std = np.std(z)
        adaptive_smoothing = min(0.01, z_std * 0.001)
        
        rbf = RBFInterpolator(
            np.column_stack([x, y]), z,
            kernel='thin_plate_spline',
            smoothing=adaptive_smoothing
        )
        grid_z_initial = rbf(np.column_stack([grid_x.ravel(), grid_y.ravel()]))
        grid_z_initial = grid_z_initial.reshape(grid_x.shape)
        
        # Apply directional smoothing using Gaussian filter
        # Create directional kernels - stronger smoothing perpendicular to flight lines
        sigma_along = 0.5    # Less smoothing along flight direction
        sigma_across = 2.0   # More smoothing across flight direction
        
        # Apply anisotropic Gaussian smoothing
        # Approximate directional smoothing by applying different sigmas
        if abs(principal_direction[0]) > abs(principal_direction[1]):
            # Predominantly horizontal flight lines
            sigma_x = sigma_along
            sigma_y = sigma_across
        else:
            # Predominantly vertical flight lines
            sigma_x = sigma_across
            sigma_y = sigma_along
        
        # Apply 2D Gaussian filter with directional emphasis
        grid_z_smoothed = gaussian_filter(grid_z_initial, sigma=[sigma_y, sigma_x], mode='nearest')
        
        # Blend original and smoothed versions based on distance to data points
        tree = cKDTree(np.column_stack([x, y]))
        distances, _ = tree.query(np.column_stack([grid_x.ravel(), grid_y.ravel()]), k=1)
        distance_weights = np.exp(-distances.reshape(grid_x.shape) / np.mean(distances))
        
        # Near data points: use more of original, far from data: use more smoothed
        grid_z_final = distance_weights * grid_z_initial + (1 - distance_weights) * grid_z_smoothed
        
        return grid_z_final
        
    except Exception:
        return griddata((x, y), z, (grid_x, grid_y), method='linear', fill_value=np.nan)


def adaptive_grid_interpolation(x, y, z, grid_x, grid_y):
    """Adaptive grid resolution interpolation based on flight line direction"""
    try:
        # Analyze flight line pattern to determine principal direction
        coords = np.column_stack([x, y])
        centered_coords = coords - np.mean(coords, axis=0)
        
        # Compute covariance matrix and eigenvalues/eigenvectors
        cov_matrix = np.cov(centered_coords.T)
        _, eigenvecs = np.linalg.eigh(cov_matrix)
        
        # Principal direction (direction of maximum variance)
        principal_direction = eigenvecs[:, 1]
        
        # Determine if flight lines are more horizontal or vertical
        is_horizontal = abs(principal_direction[0]) > abs(principal_direction[1])
        
        # Create adaptive grid with higher resolution along flight line direction
        if is_horizontal:
            # Higher resolution in longitude (x), coarser in latitude (y)
            fine_res = int(GRID_RESOLUTION * 1.2)
            coarse_res = int(GRID_RESOLUTION * 0.8)
            xi = np.linspace(grid_x.min(), grid_x.max(), fine_res)
            yi = np.linspace(grid_y.min(), grid_y.max(), coarse_res)
        else:
            # Higher resolution in latitude (y), coarser in longitude (x)
            fine_res = int(GRID_RESOLUTION * 0.8)
            coarse_res = int(GRID_RESOLUTION * 1.2)
            xi = np.linspace(grid_x.min(), grid_x.max(), fine_res)
            yi = np.linspace(grid_y.min(), grid_y.max(), coarse_res)
        
        # Create the adaptive mesh
        xi_mesh, yi_mesh = np.meshgrid(xi, yi)
        
        # Perform interpolation on adaptive grid using cubic method
        zi_adaptive = griddata((x, y), z, (xi_mesh, yi_mesh), method='cubic', fill_value=np.nan)
        
        # Interpolate back to the standard grid for consistency with other methods
        valid_mask = ~np.isnan(zi_adaptive)
        if np.any(valid_mask):
            # Use the adaptive result to interpolate to standard grid
            grid_z = griddata(
                (xi_mesh[valid_mask], yi_mesh[valid_mask]), 
                zi_adaptive[valid_mask],
                (grid_x, grid_y), 
                method='linear', 
                fill_value=np.nan
            )
            
            # Fill remaining NaNs with cubic interpolation from original data
            nan_mask = np.isnan(grid_z)
            if np.any(nan_mask):
                grid_z_cubic = griddata((x, y), z, (grid_x, grid_y), method='cubic', fill_value=np.nan)
                grid_z[nan_mask] = grid_z_cubic[nan_mask]
            
            return grid_z
        else:
            # Fallback to standard cubic interpolation
            return griddata((x, y), z, (grid_x, grid_y), method='cubic', fill_value=np.nan)
        
    except Exception:
        return griddata((x, y), z, (grid_x, grid_y), method='linear', fill_value=np.nan)


def high_quality_rbf_interpolation(x, y, z, grid_x, grid_y):
    """High-quality RBF interpolation optimized for magnetic survey data"""
    try:
        # Clean input data - remove NaN values
        valid_mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
        if not np.any(valid_mask):
            return np.full(grid_x.shape, np.nan)
        
        x_clean = x[valid_mask]
        y_clean = y[valid_mask] 
        z_clean = z[valid_mask]
        
        # Optimize dataset size vs quality tradeoff
        max_points = 3000  # Higher than other methods for better quality
        if len(x_clean) > max_points:
            # Use stratified sampling to preserve spatial distribution
            n_strata = 10
            strata_size = len(x_clean) // n_strata
            indices = []
            for i in range(n_strata):
                start_idx = i * strata_size
                end_idx = min((i + 1) * strata_size, len(x_clean))
                stratum_indices = np.arange(start_idx, end_idx)
                sample_size = min(max_points // n_strata, len(stratum_indices))
                if sample_size > 0:
                    selected = np.random.choice(stratum_indices, sample_size, replace=False)
                    indices.extend(selected)
            
            indices = np.array(indices)
            x_clean = x_clean[indices]
            y_clean = y_clean[indices]
            z_clean = z_clean[indices]
        
        print(f"High-quality RBF with {len(x_clean)} points...")
        
        # Calculate optimal smoothing based on data characteristics
        z_range = np.ptp(z_clean)
        z_std = np.std(z_clean)
        
        # Very conservative smoothing for high quality
        noise_estimate = z_std * 0.1  # Assume 10% of std is noise
        optimal_smoothing = max(1e-6, min(0.001, noise_estimate / z_range))
        
        # Calculate spatial density for adaptive epsilon
        coords = np.column_stack([x_clean, y_clean])
        tree = cKDTree(coords)
        distances, _ = tree.query(coords, k=min(10, len(coords)))
        median_distance = np.median(distances[:, 1:])  # Exclude self-distance
        
        # Use multiquadric with optimized parameters
        rbf_interpolator = RBFInterpolator(
            coords, z_clean,
            kernel='multiquadric',
            epsilon=median_distance * 0.5,  # Based on local point density
            smoothing=optimal_smoothing,
        )
        
        # Interpolate to grid
        grid_z = rbf_interpolator(np.column_stack([grid_x.ravel(), grid_y.ravel()]))
        result = grid_z.reshape(grid_x.shape)
        
        print(f"High-quality RBF completed with smoothing={optimal_smoothing:.2e}, epsilon={median_distance * 0.5:.2e}")
        return result
        
    except MemoryError:
        print("MemoryError in high-quality RBF - falling back to standard RBF")
        # Fallback to simpler RBF with fewer points
        max_fallback = 1000
        if len(x) > max_fallback:
            indices = np.random.choice(len(x), max_fallback, replace=False)
            x_fallback = x[indices]
            y_fallback = y[indices]
            z_fallback = z[indices]
        else:
            x_fallback, y_fallback, z_fallback = x, y, z
            
        rbf_simple = RBFInterpolator(
            np.column_stack([x_fallback, y_fallback]), z_fallback,
            kernel='thin_plate_spline',
            smoothing=0.01
        )
        grid_z = rbf_simple(np.column_stack([grid_x.ravel(), grid_y.ravel()]))
        return grid_z.reshape(grid_x.shape)
        
    except Exception as e:
        print(f"Error in high-quality RBF: {e} - falling back to linear")
        return griddata((x, y), z, (grid_x, grid_y), method='linear', fill_value=np.nan)

# =============================================
# CREATE RASTER OVERLAY FROM INTERPOLATED DATA
# =============================================

def create_raster_overlay(grid_field, vmin, vmax, alpha=0.6):
    """Create a raster overlay from interpolated data using viridis colormap"""
    
    # Normalize the data to 0-1 range for colormap
    normalized_data = (grid_field - vmin) / (vmax - vmin)
    
    # Handle NaN values
    mask = ~np.isnan(normalized_data)
    
    # Create RGBA image using viridis colormap
    import matplotlib.cm as cm
    
    # Get viridis colormap
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


# =============================================
# SAVE INTERPOLATED FIELD AS PNG
# =============================================

def save_interpolated_field_as_png(grid_lat, grid_lon, grid_field, heading, filename_prefix, method):
    """Save the interpolated field as a PNG image using the viridis colormap"""
    logger = logging.getLogger(__name__)
    
    # Plot the interpolated field with viridis colormap
    plt.figure(figsize=(12, 8))
    im = plt.imshow(grid_field, extent=(grid_lon.min(), grid_lon.max(), grid_lat.min(), grid_lat.max()), 
                   origin='lower', cmap='viridis', aspect='auto')
    plt.colorbar(im, label='Field Value (nT)', shrink=0.8)
    plt.title(f"Interpolated Field ({method}) for Heading {heading}")
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    # Save the plot as PNG
    output_dir = Path(INPUT_DIRECTORY)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = output_dir / f"{filename_prefix}_Heading_{heading}_{method}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved PNG image of the interpolated field to {output_filename}")
    return output_filename

# =============================================
# CREATE INTERACTIVE MAPS USING FOLIUM
# =============================================

def create_interactive_map_with_layers(heading, filename_prefix, input_directory, interpolation_results, heading_data=None):
    """Create an interactive Folium map with multiple interpolation methods as toggleable raster layers"""
    logger = logging.getLogger(__name__)
    
    # Get the first interpolation result to determine map center
    first_result = list(interpolation_results.values())[0]
    grid_lat, grid_lon, _ = first_result
    
    # Initialize the map (satellite base tiles)
    map_center = [grid_lat.mean(), grid_lon.mean()]
    m = folium.Map(location=map_center, zoom_start=14)

    # Add the Mapbox satellite base layer with attribution
    folium.TileLayer(
        tiles=f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/tiles/{{z}}/{{x}}/{{y}}?access_token={MAPBOX_TOKEN}",
        attr="Mapbox Attribution: &copy; <a href='https://www.mapbox.com/about/maps/'>Mapbox</a>",
        name="Satellite"
    ).add_to(m)
    
    # Find global min/max for consistent colormap across all methods
    all_values = []
    for grid_lat, grid_lon, grid_field in interpolation_results.values():
        all_values.extend(grid_field[~np.isnan(grid_field)].flatten())
    
    global_min = np.min(all_values)
    global_max = np.max(all_values)
    
    # Define a custom colormap using viridis colors and linear scale
    cm_field = linear.viridis.scale(global_min, global_max)
    cm_field.caption = "Interpolated Magnetic Field (nT)"
    cm_field.add_to(m)
    
    # Method descriptions for better user understanding
    method_descriptions = {
        'linear': 'Linear Interpolation - Simple triangular interpolation',
        'nearest': 'Nearest Neighbor - Assigns nearest data point value',
        'cubic': 'Cubic Interpolation - Smooth cubic interpolation',
        'parallel_lines_rbf': 'Parallel Lines RBF - Radial basis function optimized for flight lines',
        'directional_kriging': 'Directional Kriging - Gaussian process with anisotropic kernel',
        'edge_preserving': 'Edge Preserving - Convex hull masking with distance weighting',
        'multiscale_rbf': 'Multi-scale RBF - Combined multiple scale interpolation',
        'anisotropic_kernel': 'Anisotropic Kernel - Adapted to flight line geometry',
        'directional_smoothing': 'Directional Smoothing - Stronger smoothing perpendicular to flight lines',
        'adaptive_grid': 'Adaptive Grid - Higher resolution along flight line direction',
        'high_quality_rbf': 'High Quality RBF - Optimized parameters with stratified sampling'
    }
    
    # Add each interpolation method as a separate raster layer
    for method, (grid_lat, grid_lon, grid_field) in interpolation_results.items():
        logger.info(f"Creating raster overlay for {method} interpolation")
        
        try:
            # Create raster overlay with higher opacity for better visibility
            img_str = create_raster_overlay(grid_field, global_min, global_max, alpha=0.9)
            
            # Define bounds for the raster
            bounds = [
                [grid_lat.min(), grid_lon.min()],  # Southwest corner
                [grid_lat.max(), grid_lon.max()]   # Northeast corner
            ]
            
            # Get method description
            description = method_descriptions.get(method, f"{method.capitalize()} Interpolation")
            
            # Add raster overlay to map with high opacity for better visibility
            folium.raster_layers.ImageOverlay(
                image=f"data:image/png;base64,{img_str}",
                bounds=bounds,
                opacity=0.95,  # High opacity for better field visibility
                interactive=True,
                cross_origin=False,
                name=description
            ).add_to(m)
            
            logger.info(f"Successfully added {method} layer to map")
            
        except Exception as e:
            logger.error(f"Failed to create raster overlay for {method}: {e}")
            continue
    
    # Add flight path visualization if heading_data is provided
    if heading_data is not None and not heading_data.empty:
        logger.info("Adding flight path visualization to map")
        
        # Create flight path feature group
        flight_path_group = folium.FeatureGroup(name="Flight Path")
        
        # Extract coordinates for the flight path
        path_coords = []
        field_values_for_path = []
        
        # Sort data by timestamp to ensure proper path order
        if 'Timestamp [ms]' in heading_data.columns:
            heading_data_sorted = heading_data.sort_values('Timestamp [ms]')
        else:
            heading_data_sorted = heading_data
        
        for _, row in heading_data_sorted.iterrows():
            lat = row['Latitude [Decimal Degrees]']
            lon = row['Longitude [Decimal Degrees]']
            path_coords.append([lat, lon])
            
            # Get field value for coloring
            if 'R1 [nT]' in row:
                field_values_for_path.append(row['R1 [nT]'])
            else:
                field_values_for_path.append(row['Btotal1 [nT]'])
        
        # Create colormap for flight path
        field_min = min(field_values_for_path)
        field_max = max(field_values_for_path)
        cm_path = linear.viridis.scale(field_min, field_max)
        
        # Add the flight path as a polyline
        folium.PolyLine(
            locations=path_coords,
            color='white',
            weight=3,
            opacity=0.8,
            tooltip=f"Flight Path - {heading}"
        ).add_to(flight_path_group)
        
        # Add data points along the flight path
        subsample_rate = max(1, len(heading_data_sorted) // 200)  # Limit to ~200 points for performance
        for i, (_, row) in enumerate(heading_data_sorted.iterrows()):
            if i % subsample_rate == 0:  # Subsample for performance
                lat = row['Latitude [Decimal Degrees]']
                lon = row['Longitude [Decimal Degrees]']
                
                if 'R1 [nT]' in row:
                    field_value = row['R1 [nT]']
                    field_name = 'R1 [nT]'
                else:
                    field_value = row['Btotal1 [nT]']
                    field_name = 'Btotal1 [nT]'
                
                # Color based on field value
                color = cm_path(field_value)
                
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=3,
                    color='white',
                    weight=1,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.8,
                    popup=f'{field_name}: {field_value:.2f} nT<br>Lat: {lat:.6f}<br>Lon: {lon:.6f}',
                    tooltip=f'{field_value:.2f} nT'
                ).add_to(flight_path_group)
        
        # Add flight path group to map
        flight_path_group.add_to(m)
        
        # Add colormap for flight path
        cm_path.caption = f"Flight Path {heading} - Field Values (nT)"
        cm_path.add_to(m)
        
        logger.info(f"Added flight path with {len(path_coords)} points")
    
    # Add layer control for toggling between interpolation methods
    folium.LayerControl().add_to(m)
    
    # Add a fullscreen button
    from folium.plugins import Fullscreen
    Fullscreen().add_to(m)
    
    # Get the directory from the input CSV directory
    output_dir = Path(input_directory)
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as an HTML file in the same directory as the input CSV files
    output_filename = output_dir / f"{filename_prefix}_Heading_{heading}_interactive_field_map.html"
    m.save(output_filename)
    
    logger.info(f"Saved interactive field map with all interpolation methods to {output_filename}")
    return output_filename

# =============================================
# ADD ORIGINAL DATA POINTS TO MAP
# =============================================

def add_original_data_points(m, df, global_min, global_max):
    """Add original data points as a separate toggleable layer"""
    
    # Create feature group for original data
    data_points = folium.FeatureGroup(name="Original Data Points")
    
    # Get viridis colormap for consistency
    cm_field = linear.viridis.scale(global_min, global_max)
    
    # Add each data point
    for _, row in df.iterrows():
        lat = row['Latitude [Decimal Degrees]']
        lon = row['Longitude [Decimal Degrees]']
        value = row['Btotal1 [nT]']
        
        color = cm_field(value)
        
        folium.CircleMarker(
            location=[lat, lon],
            radius=4,
            color='black',
            weight=1,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            popup=f'Original Data<br>Value: {value:.2f} nT'
        ).add_to(data_points)
    
    data_points.add_to(m)

# =============================================
# MAIN PROCESSING FUNCTION
# =============================================

def main():
    logger = setup_logging()
    
    # Load data from CSV files in the directory
    df = load_data_from_csv(INPUT_DIRECTORY)
    if df is None:
        logger.error("No data loaded. Exiting script.")
        return
    
    # Group data by heading (filename)
    headings = df['Filename'].unique()
    
    for heading in headings:
        logger.info(f"Processing heading: {heading}")
        
        # Filter data for this heading
        heading_data = df[df['Filename'] == heading]
        
        # Store interpolation results for all methods
        interpolation_results = {}
        
        # Process each interpolation method for this heading
        for method in INTERPOLATION_METHODS:
            logger.info(f"Performing {method} interpolation for heading {heading}")
            
            try:
                # Interpolate the full field
                grid_lat, grid_lon, grid_field = interpolate_full_field(heading_data, method)
                
                # Check if interpolation was successful
                if grid_field is not None and not np.all(np.isnan(grid_field)):
                    interpolation_results[method] = (grid_lat, grid_lon, grid_field)
                    
                    # Extract filename prefix
                    filename_prefix = heading.split('_')[0] if '_' in heading else heading
                    
                    # Save the interpolated field as PNG using viridis colormap
                    save_interpolated_field_as_png(grid_lat, grid_lon, grid_field, heading, filename_prefix, method)
                    logger.info(f"Successfully completed {method} interpolation")
                else:
                    logger.warning(f"Failed to interpolate using {method} - all values are NaN")
                    
            except Exception as e:
                logger.error(f"Error in {method} interpolation: {e}")
                # Continue with other methods even if one fails
        
        # Create one interactive map with all interpolation methods as raster layers
        if interpolation_results:
            filename_prefix = heading.split('_')[0] if '_' in heading else heading
            map_filename = create_interactive_map_with_layers(heading, filename_prefix, INPUT_DIRECTORY, interpolation_results, heading_data)
            
            logger.info(f"Interactive field map created for {heading}: {map_filename}")
            logger.info(f"Available interpolation methods: {list(interpolation_results.keys())}")
        else:
            logger.warning(f"No successful interpolations for heading {heading} - skipping map creation")

if __name__ == "__main__":
    main()
