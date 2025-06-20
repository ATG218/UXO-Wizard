"""
Batch Processing Pipeline
========================

Main processing pipeline for handling directories of CSV files.
Integrates grid interpolation, SGTool filtering, and interactive visualization.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import logging
from scipy.interpolate import griddata
from pykrige import OrdinaryKriging
import warnings
import matplotlib.pyplot as plt
import folium

from ..io.csv_reader import CSVReader
from ..io.minimum_curvature import minimum_curvature_interpolation
from ..io.boundary_masking import create_boundary_mask, apply_boundary_mask
from ..core.geophysical_processor import GeophysicalProcessor
from ..core.frequency_filters import FrequencyFilters
from ..core.gradient_filters import GradientFilters
from ..visualization.interactive_maps import InteractiveMaps

try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    NUMBA_AVAILABLE = False

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class BatchProcessor:
    """
    Main batch processing class for magnetic survey data.
    
    Handles directory-level processing with grid interpolation,
    SGTool filtering, and interactive visualization.
    """
    
    def __init__(self, input_directory: Union[str, Path], 
                 output_directory: Optional[Union[str, Path]] = None,
                 grid_resolution: int = 300,
                 max_kriging_points: int = 15000):
        """
        Initialize batch processor.
        
        Parameters:
            input_directory (Union[str, Path]): Directory containing CSV files
            output_directory (Optional[Union[str, Path]]): Output directory
            grid_resolution (int): Grid resolution for interpolation
            max_kriging_points (int): Maximum points for kriging
        """
        self.input_directory = Path(input_directory)
        self.output_directory = Path(output_directory) if output_directory else self.input_directory / "sgtool_results"
        self.grid_resolution = grid_resolution
        self.max_kriging_points = max_kriging_points
        
        # Create output directory
        self.output_directory.mkdir(exist_ok=True)
        
        # Initialize components
        self.csv_reader = CSVReader()
        self.interactive_maps = InteractiveMaps()
        
        # Processing parameters
        self.default_filters = ['rtp', 'thg', 'analytic_signal', 'tilt_angle']
        self.magnetic_params = {
            'inclination': 70.0,  # Default for northern regions
            'declination': 2.0    # Default declination
        }
        
        # Boundary masking parameters
        self.enable_boundary_masking = True
        self.boundary_method = 'convex_hull'
        self.boundary_buffer_distance = None
        
        logger.info(f"Batch processor initialized for: {self.input_directory}")
    
    def load_data(self) -> pd.DataFrame:
        """
        Load and combine all CSV files from input directory.
        
        Returns:
            pd.DataFrame: Combined dataset
        """
        logger.info("Loading CSV files from directory...")
        
        # Read all CSV files
        dataframes = self.csv_reader.read_csv_directory(self.input_directory)
        
        if not dataframes:
            raise ValueError(f"No valid CSV files found in {self.input_directory}")
        
        # Combine dataframes
        combined_data = self.csv_reader.combine_dataframes(dataframes)
        
        # Get data summary
        summary = self.csv_reader.get_data_summary(combined_data)
        logger.info(f"Loaded {summary['total_points']} points from {summary['unique_files']} files")
        
        return combined_data
    
    def create_interpolation_grid(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple[float, float, float, float]]:
        """
        Create regular grid for interpolation with precise UTM to lat/lon conversion.
        
        Parameters:
            df (pd.DataFrame): Input data
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple]: UTM grids, lat/lon grids, and extent
        """
        # Get exact data bounds (no buffer for precise positioning)
        x_min, x_max = df['x'].min(), df['x'].max()
        y_min, y_max = df['y'].min(), df['y'].max()
        
        logger.info(f"Data UTM bounds: X=[{x_min:.1f}, {x_max:.1f}], Y=[{y_min:.1f}, {y_max:.1f}]")
        
        # Create UTM grid
        x_grid = np.linspace(x_min, x_max, self.grid_resolution)
        y_grid = np.linspace(y_min, y_max, self.grid_resolution)
        grid_x_utm, grid_y_utm = np.meshgrid(x_grid, y_grid)
        
        # Convert UTM grid to lat/lon for folium mapping with proper precision
        try:
            import pyproj
            # Use UTM Zone 33N (EPSG:32633) - common for Norway
            utm_crs = pyproj.CRS('EPSG:32633')
            geographic_crs = pyproj.CRS('EPSG:4326')
            transformer = pyproj.Transformer.from_crs(utm_crs, geographic_crs, always_xy=True)
            
            # Transform corner points first to check bounds
            corner_lons, corner_lats = transformer.transform(
                [x_min, x_max, x_min, x_max],
                [y_min, y_min, y_max, y_max]
            )
            
            logger.info(f"Corner coordinates: lat=[{min(corner_lats):.6f}, {max(corner_lats):.6f}], lon=[{min(corner_lons):.6f}, {max(corner_lons):.6f}]")
            
            # Transform the entire grid
            grid_lon, grid_lat = transformer.transform(grid_x_utm, grid_y_utm)
            
            logger.info(f"Grid transformed: lat range [{grid_lat.min():.6f}, {grid_lat.max():.6f}], lon range [{grid_lon.min():.6f}, {grid_lon.max():.6f}]")
            
        except ImportError:
            logger.warning("pyproj not available, using simplified conversion")
            # Fallback simplified conversion for Norwegian UTM Zone 33N
            grid_lat = (grid_y_utm - 7000000) / 111000 + 63
            grid_lon = (grid_x_utm - 500000) / 111000 + 15
            
            # Clamp to reasonable bounds
            grid_lat = np.clip(grid_lat, 58, 72)
            grid_lon = np.clip(grid_lon, 4, 32)
        
        extent = (x_min, x_max, y_min, y_max)
        
        logger.info(f"Created {self.grid_resolution}x{self.grid_resolution} interpolation grid")
        
        return grid_x_utm, grid_y_utm, grid_lat, grid_lon, extent
    
    def downsample_for_kriging(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                              max_points: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Downsample data for kriging performance.
        
        Parameters:
            x, y, z (np.ndarray): Input coordinates and values
            max_points (int): Maximum points to keep
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Downsampled data
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
    
    def interpolate_data(self, df: pd.DataFrame, grid_x: np.ndarray, grid_y: np.ndarray,
                        method: str = 'kriging') -> np.ndarray:
        """
        Interpolate data to regular grid.
        
        Parameters:
            df (pd.DataFrame): Input data
            grid_x, grid_y (np.ndarray): Grid coordinates
            method (str): Interpolation method ('kriging', 'linear', 'cubic')
            
        Returns:
            np.ndarray: Interpolated grid
        """
        # Extract coordinates and values
        x = df['x'].values
        y = df['y'].values
        z = df['magnetic_field'].values
        
        # Remove NaN values
        valid_mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
        x, y, z = x[valid_mask], y[valid_mask], z[valid_mask]
        
        if len(x) == 0:
            logger.error("No valid data points for interpolation")
            return np.full(grid_x.shape, np.nan)
        
        if method == 'minimum_curvature':
            try:
                # Use vectorized minimum curvature interpolation
                grid_z = minimum_curvature_interpolation(
                    x, y, z, grid_x, grid_y,
                    max_iterations=100,
                    tolerance=1e-3,
                    omega=0.5,
                    max_points=None  # No limit - use all points
                )
                
                logger.info("Minimum curvature interpolation completed")
                return grid_z
                
            except Exception as e:
                logger.warning(f"Minimum curvature failed: {e}, falling back to kriging")
                method = 'kriging'
        
        if method == 'kriging':
            try:
                # Downsample for performance
                if len(x) > self.max_kriging_points:
                    x, y, z = self.downsample_for_kriging(x, y, z, self.max_kriging_points)
                
                # Create kriging object
                ok = OrdinaryKriging(
                    x, y, z,
                    variogram_model='gaussian',
                    weight=True,
                    exact_values=True,
                    verbose=False,
                    enable_plotting=False
                )
                
                # Execute kriging
                grid_z, _ = ok.execute('grid', 
                                     grid_x[0, :], grid_y[:, 0])
                
                logger.info("Kriging interpolation completed")
                return grid_z
                
            except Exception as e:
                logger.warning(f"Kriging failed: {e}, falling back to linear interpolation")
                method = 'linear'
        
        if method in ['linear', 'cubic']:
            try:
                grid_z = griddata((x, y), z, (grid_x, grid_y), 
                                method=method, fill_value=np.nan)
                logger.info(f"{method.capitalize()} interpolation completed")
                return grid_z
                
            except Exception as e:
                logger.error(f"{method} interpolation failed: {e}")
                return np.full(grid_x.shape, np.nan)
        
        raise ValueError(f"Unknown interpolation method: {method}")
    
    def apply_sgtool_filters(self, grid_data: np.ndarray, dx: float, dy: float,
                           filters: List[str]) -> Dict[str, np.ndarray]:
        """
        Apply SGTool filters to gridded data.
        
        Parameters:
            grid_data (np.ndarray): Input grid
            dx, dy (float): Grid spacing
            filters (List[str]): List of filters to apply
            
        Returns:
            Dict[str, np.ndarray]: Dictionary of filtered results
        """
        results = {}
        
        # Initialize processors
        geo_processor = GeophysicalProcessor(dx, dy)
        freq_filters = FrequencyFilters(dx, dy)
        grad_filters = GradientFilters(dx, dy)
        
        logger.info(f"Applying {len(filters)} SGTool filters...")
        
        for filter_name in filters:
            try:
                if filter_name == 'rtp':
                    # Reduction to Pole
                    filtered = geo_processor.reduction_to_pole(
                        grid_data, 
                        self.magnetic_params['inclination'],
                        self.magnetic_params['declination']
                    )
                    results['RTP'] = self._validate_filter_result(filtered, 'RTP')
                    
                elif filter_name == 'rte':
                    # Reduction to Equator
                    filtered = geo_processor.reduction_to_equator(
                        grid_data,
                        self.magnetic_params['inclination'],
                        self.magnetic_params['declination']
                    )
                    results['RTE'] = self._validate_filter_result(filtered, 'RTE')
                    
                elif filter_name == 'upward_continuation':
                    # Upward continuation (100m)
                    filtered = geo_processor.upward_continuation(grid_data, 100.0)
                    results['Upward_Continuation_100m'] = self._validate_filter_result(filtered, 'Upward_Continuation_100m')
                    
                elif filter_name == 'vertical_integration':
                    # Vertical integration (pseudogravity)
                    filtered = geo_processor.vertical_integration(grid_data)
                    results['Vertical_Integration'] = self._validate_filter_result(filtered, 'Vertical_Integration')
                    
                elif filter_name == 'thg':
                    # Total Horizontal Gradient
                    filtered = grad_filters.total_horizontal_gradient(grid_data)
                    results['THG'] = self._validate_filter_result(filtered, 'THG')
                    
                elif filter_name == 'analytic_signal':
                    # Analytic Signal
                    filtered = grad_filters.analytic_signal(grid_data)
                    results['Analytic_Signal'] = self._validate_filter_result(filtered, 'Analytic_Signal')
                    
                elif filter_name == 'tilt_angle':
                    # Tilt Angle
                    filtered = grad_filters.tilt_angle_degrees(grid_data)
                    results['Tilt_Angle'] = self._validate_filter_result(filtered, 'Tilt_Angle')
                    
                elif filter_name == 'high_pass':
                    # High-pass filter
                    wavelength = min(dx, dy) * 50  # 50 grid cells
                    filtered = freq_filters.high_pass_filter(grid_data, wavelength)
                    results['High_Pass'] = self._validate_filter_result(filtered, 'High_Pass')
                    
                elif filter_name == 'low_pass':
                    # Low-pass filter
                    wavelength = min(dx, dy) * 10  # 10 grid cells
                    filtered = freq_filters.low_pass_filter(grid_data, wavelength)
                    results['Low_Pass'] = self._validate_filter_result(filtered, 'Low_Pass')
                    
                elif filter_name == 'remove_regional':
                    # Remove regional trend
                    filtered = geo_processor.remove_regional_trend(grid_data, order=1)
                    results['Regional_Removed'] = self._validate_filter_result(filtered, 'Regional_Removed')
                    
                else:
                    logger.warning(f"Unknown filter: {filter_name}")
                    continue
                    
                logger.info(f"Applied filter: {filter_name}")
                
            except Exception as e:
                logger.error(f"Failed to apply filter {filter_name}: {e}")
                continue
        
        return results
    
    def create_interactive_map(self, original_data: pd.DataFrame, 
                             grid_results: Dict[str, np.ndarray],
                             grid_lat: np.ndarray, grid_lon: np.ndarray) -> str:
        """
        Create comprehensive interactive map using grid_interpolator.py approach.
        
        Parameters:
            original_data (pd.DataFrame): Original point data
            grid_results (Dict[str, np.ndarray]): Filtered grid results
            grid_lat, grid_lon (np.ndarray): Lat/lon grids for mapping
            
        Returns:
            str: Path to HTML map file
        """
        logger.info("Creating interactive map...")
        
        # Calculate map center from grid coordinates
        center_lat = grid_lat.mean()
        center_lon = grid_lon.mean()
        
        logger.info(f"Map center: lat={center_lat:.6f}, lon={center_lon:.6f}")
        
        # Create base map
        m = self.interactive_maps.create_base_map(center_lat, center_lon, zoom_start=14)
        
        # Find global min/max for consistent colormap
        all_values = []
        for grid_data in grid_results.values():
            valid_data = grid_data[~np.isnan(grid_data) & np.isfinite(grid_data)]
            if len(valid_data) > 0:
                all_values.extend(valid_data.flatten())
        
        if len(all_values) == 0:
            logger.error("No valid values found for colormap")
            return None
        
        global_min = np.min(all_values)
        global_max = np.max(all_values)
        
        # Ensure min < max
        if global_min >= global_max:
            global_max = global_min + 1.0
        
        logger.info(f"Colormap range: {global_min:.2f} to {global_max:.2f}")
        
        # Create dynamic colorbars that show/hide with layers
        from branca.colormap import linear
        
        # Prepare colorbar data for JavaScript control (don't add to map yet)
        layer_colorbars = {}
        for name, grid_data in grid_results.items():
            valid_data = grid_data[~np.isnan(grid_data)]
            if len(valid_data) > 0:
                # Remove extreme outliers for better range calculation
                mean_val = np.mean(valid_data)
                std_val = np.std(valid_data)
                outlier_mask = np.abs(valid_data - mean_val) <= 5 * std_val
                
                if np.any(outlier_mask):
                    filtered_data = valid_data[outlier_mask]
                    layer_min = np.percentile(filtered_data, 2)
                    layer_max = np.percentile(filtered_data, 98)
                    
                    outliers_removed = len(valid_data) - len(filtered_data)
                    if outliers_removed > 0:
                        logger.warning(f"{name}: Removed {outliers_removed} extreme outliers for colorbar range")
                else:
                    layer_min = np.percentile(valid_data, 2)
                    layer_max = np.percentile(valid_data, 98)
                
                # Create appropriate colormap and units using available branca colormaps
                if 'RTP' in name or 'Original' in name:
                    cm_layer = linear.viridis.scale(layer_min, layer_max)
                    unit = "nT"
                elif 'THG' in name or 'Analytic' in name:
                    cm_layer = linear.plasma.scale(layer_min, layer_max)  # Use plasma for distinction
                    unit = "nT/m"
                elif 'Tilt' in name:
                    cm_layer = linear.inferno.scale(layer_min, layer_max)  # Use inferno for tilt
                    unit = "degrees"
                else:
                    cm_layer = linear.magma.scale(layer_min, layer_max)  # Use magma (available colormap)
                    unit = ""
                
                cm_layer.caption = f"{name}: {layer_min:.1f} to {layer_max:.1f} {unit}"
                layer_colorbars[name] = cm_layer
                
                logger.info(f"{name} colorbar range: {layer_min:.2f} to {layer_max:.2f} {unit}")
        
        # Add grid overlays with proper positioning
        layers_added = 0
        for name, grid_data in grid_results.items():
            logger.info(f"Adding {name} layer to map")
            try:
                # Add grid overlay with precise positioning
                self.interactive_maps.add_grid_overlay(
                    m, grid_data, grid_lat, grid_lon, 
                    layer_name=name
                )
                layers_added += 1
            except Exception as e:
                logger.error(f"Failed to add {name} layer: {e}")
        
        logger.info(f"Successfully added {layers_added} data layers")
        
        # Add layer control with grouping
        folium.LayerControl(position='topright', collapsed=False).add_to(m)
        
        # Add all colorbars positioned in bottom-right corner
        for name, colorbar in layer_colorbars.items():
            colorbar.add_to(m)
        
        # Add custom CSS to header for proper colorbar positioning
        colorbar_css = '''
        <style>
        /* Target ONLY branca colormap elements - don't break other controls */
        div[id*="color_map"] {
            background: rgba(255, 255, 255, 0.9) !important;
            border: 1px solid #ccc !important;
            border-radius: 3px !important;
            padding: 5px !important;
        }
        
        /* Style SVG elements inside colormaps */
        div[id*="color_map"] svg {
            background: rgba(255, 255, 255, 0.9) !important;
            border: 1px solid #ccc !important;
            border-radius: 3px !important;
            padding: 5px !important;
        }
        </style>
        '''
        
        # Add CSS to map header using proper Folium method
        m.get_root().header.add_child(folium.Element(colorbar_css))
        
        # Add toggle button HTML
        toggle_html = '''
        <div style="position: fixed; 
                    bottom: 10px; left: 10px; width: 100px; height: 35px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:11px; text-align: center; padding: 3px; border-radius: 5px;">
        <button onclick="toggleAllLayers()" style="width: 100%; height: 28px; font-size: 10px;">Toggle All</button>
        </div>
        '''
        
        # Add HTML to body
        m.get_root().html.add_child(folium.Element(toggle_html))
        
        # Add JavaScript to script section using proper method
        toggle_js = '''
        function toggleAllLayers() {
            var checkboxes = document.querySelectorAll('.leaflet-control-layers-overlays input[type="checkbox"]');
            var allChecked = Array.from(checkboxes).every(cb => cb.checked);
            
            checkboxes.forEach(function(checkbox) {
                if (allChecked) {
                    if (checkbox.checked) checkbox.click();
                } else {
                    if (!checkbox.checked) checkbox.click();
                }
            });
        }
        
        // Function to position ONLY colorbars in bottom-right corner
        function positionAllColorbars() {
            // Find ONLY colormap divs, not other controls
            var colormaps = document.querySelectorAll('div[id*="color_map"]');
            var bottomStart = 60;
            var spacing = 50;
            
            colormaps.forEach(function(colormap, index) {
                colormap.style.position = 'fixed';
                colormap.style.right = '10px';
                colormap.style.bottom = (bottomStart + (index * spacing)) + 'px';
                colormap.style.zIndex = '1000';
            });
            
            console.log('Positioned ' + colormaps.length + ' colorbars (without breaking other controls)');
        }
        
        // Run positioning after page loads and after short delay to ensure colorbars are created
        document.addEventListener('DOMContentLoaded', function() {
            setTimeout(positionAllColorbars, 500);  // Wait 500ms for colorbars to be created
            setTimeout(positionAllColorbars, 1500); // Try again after 1.5s to be sure
        });
        '''
        
        # Add JavaScript to script section
        m.get_root().script.add_child(folium.Element(toggle_js))
        
        # Add fullscreen button
        try:
            from folium.plugins import Fullscreen
            Fullscreen().add_to(m)
        except ImportError:
            logger.warning("Fullscreen plugin not available")
        
        # Save map
        output_file = self.output_directory / "interactive_map.html"
        m.save(str(output_file))
        
        logger.info(f"Interactive map saved to: {output_file}")
        return str(output_file)
    
    def save_results(self, grid_results: Dict[str, np.ndarray], 
                    extent: Tuple[float, float, float, float]) -> Dict[str, str]:
        """
        Save processing results to files.
        
        Parameters:
            grid_results (Dict[str, np.ndarray]): Processing results
            extent (Tuple): Grid extent
            
        Returns:
            Dict[str, str]: Mapping of result names to file paths
        """
        saved_files = {}
        
        for name, grid_data in grid_results.items():
            # Save as CSV with coordinates
            csv_file = self.output_directory / f"{name}.csv"
            ny, nx = grid_data.shape
            x_coords = np.linspace(extent[0], extent[1], nx)
            y_coords = np.linspace(extent[2], extent[3], ny)
            
            # Create coordinate meshgrid
            xx, yy = np.meshgrid(x_coords, y_coords)
            
            # Flatten and create DataFrame
            df_export = pd.DataFrame({
                'x': xx.flatten(),
                'y': yy.flatten(),
                'value': grid_data.flatten()
            })
            
            # Remove NaN values
            df_export = df_export.dropna()
            df_export.to_csv(csv_file, index=False)
            saved_files[f"{name}_csv"] = str(csv_file)
            
            # Save as PNG
            png_file = self.output_directory / f"{name}.png"
            self._create_png_visualization(grid_data, extent, name, png_file)
            saved_files[f"{name}_png"] = str(png_file)
        
        logger.info(f"Saved {len(grid_results)} results to {self.output_directory}")
        
        return saved_files
    
    def run(self, filters: Optional[List[str]] = None, 
           interpolation_method: str = 'kriging',
           create_map: bool = True) -> Dict[str, Union[str, Dict]]:
        """
        Run complete batch processing pipeline.
        
        Parameters:
            filters (Optional[List[str]]): Filters to apply
            interpolation_method (str): Interpolation method
            create_map (bool): Whether to create interactive map
            
        Returns:
            Dict: Processing results summary
        """
        if filters is None:
            filters = self.default_filters
        
        logger.info("Starting batch processing pipeline...")
        
        try:
            # 1. Load data
            combined_data = self.load_data()
            
            # 2. Create interpolation grid
            grid_x_utm, grid_y_utm, grid_lat, grid_lon, extent = self.create_interpolation_grid(combined_data)
            
            # 3. Interpolate data
            original_grid = self.interpolate_data(combined_data, grid_x_utm, grid_y_utm, interpolation_method)
            
            # 4. Apply boundary masking if enabled
            if self.enable_boundary_masking:
                logger.info(f"Applying boundary masking using {self.boundary_method} method")
                boundary_mask = create_boundary_mask(
                    combined_data['x'].values, combined_data['y'].values,
                    grid_x_utm, grid_y_utm,
                    method=self.boundary_method,
                    buffer_distance=self.boundary_buffer_distance
                )
                original_grid = apply_boundary_mask(original_grid, boundary_mask)
                logger.info("Boundary masking applied successfully")
            
            # Calculate grid spacing
            dx = (extent[1] - extent[0]) / (self.grid_resolution - 1)
            dy = (extent[3] - extent[2]) / (self.grid_resolution - 1)
            
            # 5. Apply SGTool filters
            filter_results = self.apply_sgtool_filters(original_grid, dx, dy, filters)
            
            # Add original grid to results
            filter_results['Original_Grid'] = original_grid
            
            # 5. Save results
            saved_files = self.save_results(filter_results, extent)
            
            # 6. Create interactive map
            map_file = None
            if create_map:
                map_file = self.create_interactive_map(combined_data, filter_results, grid_lat, grid_lon)
            
            # Prepare summary
            summary = {
                'input_directory': str(self.input_directory),
                'output_directory': str(self.output_directory),
                'total_points': len(combined_data),
                'grid_resolution': self.grid_resolution,
                'interpolation_method': interpolation_method,
                'filters_applied': list(filter_results.keys()),
                'saved_files': saved_files,
                'interactive_map': map_file,
                'extent': extent,
                'grid_spacing': (dx, dy)
            }
            
            logger.info("Batch processing completed successfully!")
            return summary
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            raise
    
    def _validate_filter_result(self, filtered_data: np.ndarray, filter_name: str) -> np.ndarray:
        """
        Validate and clean filter results to prevent extreme outliers.
        
        Parameters:
            filtered_data (np.ndarray): Filter result data
            filter_name (str): Name of the filter for logging
            
        Returns:
            np.ndarray: Validated and cleaned data
        """
        valid_mask = ~np.isnan(filtered_data)
        if not np.any(valid_mask):
            logger.warning(f"{filter_name}: No valid data after filtering")
            return filtered_data
        
        valid_data = filtered_data[valid_mask]
        mean_val = np.mean(valid_data)
        std_val = np.std(valid_data)
        
        # Check for extreme outliers (beyond 10 standard deviations)
        extreme_mask = np.abs(valid_data - mean_val) > 10 * std_val
        if np.any(extreme_mask):
            n_extreme = np.sum(extreme_mask)
            extreme_min = np.min(valid_data[extreme_mask])
            extreme_max = np.max(valid_data[extreme_mask])
            
            logger.warning(f"{filter_name}: Found {n_extreme} extreme outliers (range: {extreme_min:.2e} to {extreme_max:.2e})")
            logger.warning(f"{filter_name}: Data stats - mean: {mean_val:.2e}, std: {std_val:.2e}")
            
            # Clip extreme values to 5 standard deviations
            cleaned_data = filtered_data.copy()
            outlier_mask = np.abs(cleaned_data - mean_val) > 5 * std_val
            cleaned_data[outlier_mask] = np.nan
            
            logger.info(f"{filter_name}: Clipped {np.sum(outlier_mask)} extreme values")
            return cleaned_data
        
        logger.info(f"{filter_name}: Data range [{np.min(valid_data):.2f}, {np.max(valid_data):.2f}], std: {std_val:.2f}")
        return filtered_data
    
    def _create_png_visualization(self, grid_data: np.ndarray, extent: Tuple[float, float, float, float],
                                 name: str, output_file: Path) -> None:
        """
        Create PNG visualization of grid data.
        
        Parameters:
            grid_data (np.ndarray): Grid data to visualize
            extent (Tuple): Grid extent (x_min, x_max, y_min, y_max)
            name (str): Name of the filter/data
            output_file (Path): Output PNG file path
        """
        try:
            plt.figure(figsize=(12, 10))
            
            # Choose colormap based on filter type
            if 'RTP' in name or 'Original' in name:
                cmap = 'RdYlBu_r'
                label = 'Magnetic Field (nT)'
            elif 'THG' in name or 'Analytic' in name:
                cmap = 'viridis'
                label = 'Amplitude (nT/m)'
            elif 'Tilt' in name:
                cmap = 'RdBu_r'
                label = 'Tilt Angle (degrees)'
            else:
                cmap = 'plasma'
                label = 'Value'
            
            # Calculate robust statistics for color scaling
            valid_data = grid_data[~np.isnan(grid_data)]
            if len(valid_data) > 0:
                vmin = np.percentile(valid_data, 2)
                vmax = np.percentile(valid_data, 98)
            else:
                vmin, vmax = 0, 1
            
            # Create the plot
            im = plt.imshow(grid_data, extent=extent, origin='lower', 
                          cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
            
            # Add colorbar
            cbar = plt.colorbar(im, shrink=0.8)
            cbar.set_label(label, fontsize=12)
            
            # Add labels and title
            plt.xlabel('Easting (m)', fontsize=12)
            plt.ylabel('Northing (m)', fontsize=12)
            plt.title(f'{name.replace("_", " ")}', fontsize=14, fontweight='bold')
            
            # Add grid
            plt.grid(True, alpha=0.3)
            
            # Tight layout
            plt.tight_layout()
            
            # Save with high DPI
            plt.savefig(output_file, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
            
            logger.info(f"Created PNG visualization: {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to create PNG for {name}: {e}")
            plt.close()  # Ensure figure is closed even on error