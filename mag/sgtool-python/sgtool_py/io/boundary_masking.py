"""
Boundary Masking for Interpolated Grids
=======================================

Create boundary masks to prevent interpolation outside data coverage.
Adapted from grid_interpolator.py boundary masking functionality.
"""

import numpy as np
import logging
from typing import Tuple, Optional, Union
from scipy.spatial import ConvexHull, distance
from scipy.spatial.distance import cdist

try:
    from scipy.spatial import cKDTree
    CKDTREE_AVAILABLE = True
except ImportError:
    CKDTREE_AVAILABLE = False

try:
    import alphashape
    ALPHASHAPE_AVAILABLE = True
except ImportError:
    ALPHASHAPE_AVAILABLE = False

logger = logging.getLogger(__name__)


def create_boundary_mask(x: np.ndarray, y: np.ndarray, 
                        grid_x: np.ndarray, grid_y: np.ndarray,
                        method: str = 'convex_hull',
                        buffer_distance: Optional[float] = None,
                        alpha: float = 0.1) -> np.ndarray:
    """
    Create a boundary mask to prevent interpolation outside data coverage.
    
    Parameters:
        x, y (np.ndarray): Data point coordinates
        grid_x, grid_y (np.ndarray): Grid meshgrid coordinates
        method (str): 'convex_hull', 'alpha_shape', or 'distance'
        buffer_distance (Optional[float]): Buffer distance in coordinate units
        alpha (float): Alpha parameter for alpha shape (smaller = tighter fit)
        
    Returns:
        np.ndarray: Boolean mask (True = valid area, False = masked)
    """
    try:
        if method == 'convex_hull':
            return _convex_hull_mask(x, y, grid_x, grid_y, buffer_distance)
        elif method == 'alpha_shape':
            return _alpha_shape_mask(x, y, grid_x, grid_y, alpha, buffer_distance)
        elif method == 'distance':
            return _distance_mask(x, y, grid_x, grid_y, buffer_distance)
        else:
            logger.warning(f"Unknown boundary method: {method}, using convex_hull")
            return _convex_hull_mask(x, y, grid_x, grid_y, buffer_distance)
            
    except Exception as e:
        logger.error(f"Boundary masking failed: {e}, using no mask")
        return np.ones(grid_x.shape, dtype=bool)


def _convex_hull_mask(x: np.ndarray, y: np.ndarray,
                     grid_x: np.ndarray, grid_y: np.ndarray,
                     buffer_distance: Optional[float] = None) -> np.ndarray:
    """Create boundary mask using convex hull."""
    
    # Create convex hull of data points
    points = np.column_stack((x, y))
    
    try:
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        
        # Apply buffer if specified
        if buffer_distance:
            hull_points = _apply_buffer_to_polygon(hull_points, buffer_distance)
        
        # Test which grid points are inside the hull
        grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
        mask = _points_in_polygon(grid_points, hull_points)
        
        return mask.reshape(grid_x.shape)
        
    except Exception as e:
        logger.warning(f"Convex hull creation failed: {e}")
        return np.ones(grid_x.shape, dtype=bool)


def _alpha_shape_mask(x: np.ndarray, y: np.ndarray,
                     grid_x: np.ndarray, grid_y: np.ndarray,
                     alpha: float = 0.1,
                     buffer_distance: Optional[float] = None) -> np.ndarray:
    """Create boundary mask using alpha shape (requires alphashape package)."""
    
    if not ALPHASHAPE_AVAILABLE:
        logger.warning("alphashape package not available, falling back to convex hull")
        return _convex_hull_mask(x, y, grid_x, grid_y, buffer_distance)
    
    try:
        points = np.column_stack((x, y))
        
        # Create alpha shape
        alpha_shape = alphashape.alphashape(points, alpha)
        
        if alpha_shape.is_empty:
            logger.warning("Alpha shape is empty, falling back to convex hull")
            return _convex_hull_mask(x, y, grid_x, grid_y, buffer_distance)
        
        # Convert to boundary points
        if hasattr(alpha_shape, 'exterior'):
            # Polygon
            boundary_points = np.array(alpha_shape.exterior.coords)
        else:
            # MultiPolygon or other - use convex hull as fallback
            logger.warning("Complex alpha shape, falling back to convex hull")
            return _convex_hull_mask(x, y, grid_x, grid_y, buffer_distance)
        
        # Apply buffer if specified
        if buffer_distance:
            boundary_points = _apply_buffer_to_polygon(boundary_points, buffer_distance)
        
        # Test which grid points are inside
        grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
        mask = _points_in_polygon(grid_points, boundary_points)
        
        return mask.reshape(grid_x.shape)
        
    except Exception as e:
        logger.warning(f"Alpha shape creation failed: {e}, falling back to convex hull")
        return _convex_hull_mask(x, y, grid_x, grid_y, buffer_distance)


def _distance_mask(x: np.ndarray, y: np.ndarray,
                  grid_x: np.ndarray, grid_y: np.ndarray,
                  max_distance: Optional[float] = None) -> np.ndarray:
    """Create boundary mask using distance from nearest data point."""
    
    if max_distance is None:
        # Auto-calculate reasonable distance based on data density
        points = np.column_stack((x, y))
        if len(points) > 1000:
            # Sample for performance
            sample_indices = np.random.choice(len(points), 1000, replace=False)
            sample_points = points[sample_indices]
        else:
            sample_points = points
        
        # Calculate median nearest neighbor distance
        distances = cdist(sample_points, sample_points)
        np.fill_diagonal(distances, np.inf)
        nearest_distances = np.min(distances, axis=1)
        max_distance = np.median(nearest_distances) * 3  # 3x median spacing
    
    try:
        # Use KDTree for efficient distance queries if available
        if CKDTREE_AVAILABLE:
            tree = cKDTree(np.column_stack((x, y)))
            grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
            distances, _ = tree.query(grid_points)
            mask = distances <= max_distance
        else:
            # Fallback to direct distance calculation (slower)
            mask = np.zeros(grid_x.shape, dtype=bool)
            for i in range(grid_x.shape[0]):
                for j in range(grid_x.shape[1]):
                    grid_point = np.array([grid_x[i, j], grid_y[i, j]])
                    min_dist = np.min(np.sqrt((x - grid_point[0])**2 + (y - grid_point[1])**2))
                    mask[i, j] = min_dist <= max_distance
            return mask
        
        return mask.reshape(grid_x.shape)
        
    except Exception as e:
        logger.warning(f"Distance masking failed: {e}")
        return np.ones(grid_x.shape, dtype=bool)


def _points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """
    Test which points are inside a polygon using ray casting algorithm.
    
    Parameters:
        points (np.ndarray): Points to test (N, 2)
        polygon (np.ndarray): Polygon vertices (M, 2)
        
    Returns:
        np.ndarray: Boolean array (N,) indicating which points are inside
    """
    n_points = len(points)
    inside = np.zeros(n_points, dtype=bool)
    
    j = len(polygon) - 1
    for i in range(len(polygon)):
        for p in range(n_points):
            if (((polygon[i, 1] > points[p, 1]) != (polygon[j, 1] > points[p, 1])) and
                (points[p, 0] < (polygon[j, 0] - polygon[i, 0]) * (points[p, 1] - polygon[i, 1]) / 
                 (polygon[j, 1] - polygon[i, 1]) + polygon[i, 0])):
                inside[p] = not inside[p]
        j = i
    
    return inside


def _apply_buffer_to_polygon(polygon: np.ndarray, buffer_distance: float) -> np.ndarray:
    """
    Apply buffer to polygon (simplified implementation).
    
    This is a basic implementation that expands the polygon outward.
    For production use, consider using shapely for more robust buffering.
    """
    try:
        # Calculate centroid
        centroid = np.mean(polygon, axis=0)
        
        # Expand each vertex away from centroid
        buffered_polygon = np.zeros_like(polygon)
        for i, vertex in enumerate(polygon):
            direction = vertex - centroid
            direction_norm = np.linalg.norm(direction)
            if direction_norm > 0:
                direction = direction / direction_norm
                buffered_polygon[i] = vertex + direction * buffer_distance
            else:
                buffered_polygon[i] = vertex
        
        return buffered_polygon
        
    except Exception as e:
        logger.warning(f"Buffer application failed: {e}, using original polygon")
        return polygon


def apply_boundary_mask(grid_data: np.ndarray, boundary_mask: np.ndarray) -> np.ndarray:
    """
    Apply boundary mask to grid data.
    
    Parameters:
        grid_data (np.ndarray): Grid data to mask
        boundary_mask (np.ndarray): Boolean mask (True = keep, False = mask out)
        
    Returns:
        np.ndarray: Masked grid data (NaN where masked)
    """
    masked_data = grid_data.copy()
    masked_data[~boundary_mask] = np.nan
    return masked_data


def get_data_coverage_info(x: np.ndarray, y: np.ndarray) -> dict:
    """
    Get information about data coverage for boundary masking decisions.
    
    Parameters:
        x, y (np.ndarray): Data point coordinates
        
    Returns:
        dict: Coverage information
    """
    points = np.column_stack((x, y))
    
    info = {
        'n_points': len(points),
        'x_range': (x.min(), x.max()),
        'y_range': (y.min(), y.max()),
        'extent': [x.min(), x.max(), y.min(), y.max()],
    }
    
    # Calculate approximate data density
    area = (x.max() - x.min()) * (y.max() - y.min())
    if area > 0:
        info['density_points_per_unit'] = len(points) / area
    else:
        info['density_points_per_unit'] = 0
    
    # Calculate convex hull area vs bounding box area (measure of concavity)
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume  # In 2D, volume is area
        bbox_area = area
        info['concavity_ratio'] = hull_area / bbox_area if bbox_area > 0 else 0
        info['hull_area'] = hull_area
    except:
        info['concavity_ratio'] = 1.0
        info['hull_area'] = area
    
    return info