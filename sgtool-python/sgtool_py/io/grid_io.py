"""
Grid I/O Operations
==================

Handle reading and writing of grid/raster data formats.
"""

import numpy as np
from pathlib import Path
from typing import Union, Tuple, Optional, Dict
import logging

try:
    import rasterio
    from rasterio.transform import from_bounds
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

logger = logging.getLogger(__name__)


class GridIO:
    """
    Grid input/output operations for geophysical data.
    
    Handles various grid formats including GeoTIFF, numpy arrays, and ASCII grids.
    """
    
    def __init__(self):
        """Initialize GridIO."""
        self.supported_formats = ['.tif', '.tiff', '.npy', '.asc', '.txt']
    
    def read_grid(self, file_path: Union[str, Path]) -> Tuple[np.ndarray, Dict]:
        """
        Read grid data from file.
        
        Parameters:
            file_path (Union[str, Path]): Path to grid file
            
        Returns:
            Tuple[np.ndarray, Dict]: Grid data and metadata
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Grid file not found: {file_path}")
        
        suffix = file_path.suffix.lower()
        
        if suffix in ['.tif', '.tiff'] and RASTERIO_AVAILABLE:
            return self._read_geotiff(file_path)
        elif suffix == '.npy':
            return self._read_numpy(file_path)
        elif suffix in ['.asc', '.txt']:
            return self._read_ascii(file_path)
        else:
            raise ValueError(f"Unsupported grid format: {suffix}")
    
    def write_grid(self, data: np.ndarray, file_path: Union[str, Path],
                  extent: Optional[Tuple[float, float, float, float]] = None,
                  crs: Optional[str] = None) -> None:
        """
        Write grid data to file.
        
        Parameters:
            data (np.ndarray): Grid data
            file_path (Union[str, Path]): Output file path
            extent (Optional[Tuple]): Spatial extent (xmin, xmax, ymin, ymax)
            crs (Optional[str]): Coordinate reference system
        """
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()
        
        if suffix in ['.tif', '.tiff'] and RASTERIO_AVAILABLE:
            self._write_geotiff(data, file_path, extent, crs)
        elif suffix == '.npy':
            self._write_numpy(data, file_path)
        elif suffix in ['.asc', '.txt']:
            self._write_ascii(data, file_path, extent)
        else:
            raise ValueError(f"Unsupported output format: {suffix}")
    
    def _read_geotiff(self, file_path: Path) -> Tuple[np.ndarray, Dict]:
        """Read GeoTIFF file using rasterio."""
        with rasterio.open(file_path) as src:
            data = src.read(1)  # Read first band
            
            # Create metadata
            metadata = {
                'transform': src.transform,
                'crs': src.crs,
                'extent': src.bounds,
                'shape': data.shape,
                'nodata': src.nodata
            }
            
            # Handle nodata values
            if src.nodata is not None:
                data = np.where(data == src.nodata, np.nan, data)
        
        logger.info(f"Read GeoTIFF: {file_path.name}, shape: {data.shape}")
        return data, metadata
    
    def _read_numpy(self, file_path: Path) -> Tuple[np.ndarray, Dict]:
        """Read numpy array file."""
        data = np.load(file_path)
        
        metadata = {
            'shape': data.shape,
            'format': 'numpy'
        }
        
        logger.info(f"Read numpy array: {file_path.name}, shape: {data.shape}")
        return data, metadata
    
    def _read_ascii(self, file_path: Path) -> Tuple[np.ndarray, Dict]:
        """Read ASCII grid file."""
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Parse header
        header = {}
        data_start = 0
        
        for i, line in enumerate(lines):
            if line.strip().startswith(('ncols', 'nrows', 'xllcorner', 
                                       'yllcorner', 'cellsize', 'nodata')):
                key, value = line.strip().split()
                if key.lower() in ['ncols', 'nrows']:
                    header[key.lower()] = int(value)
                elif key.lower() == 'nodata':
                    header[key.lower()] = float(value)
                else:
                    header[key.lower()] = float(value)
                data_start = i + 1
            else:
                break
        
        # Read data
        data_lines = lines[data_start:]
        data = []
        for line in data_lines:
            if line.strip():
                row = [float(x) for x in line.strip().split()]
                data.append(row)
        
        data = np.array(data)
        
        # Handle nodata
        if 'nodata' in header:
            data = np.where(data == header['nodata'], np.nan, data)
        
        # Calculate extent if available
        metadata = {'header': header, 'format': 'ascii'}
        if all(k in header for k in ['xllcorner', 'yllcorner', 'cellsize', 'ncols', 'nrows']):
            xmin = header['xllcorner']
            ymin = header['yllcorner']
            cellsize = header['cellsize']
            xmax = xmin + header['ncols'] * cellsize
            ymax = ymin + header['nrows'] * cellsize
            metadata['extent'] = (xmin, xmax, ymin, ymax)
        
        logger.info(f"Read ASCII grid: {file_path.name}, shape: {data.shape}")
        return data, metadata
    
    def _write_geotiff(self, data: np.ndarray, file_path: Path,
                      extent: Optional[Tuple[float, float, float, float]] = None,
                      crs: Optional[str] = None) -> None:
        """Write data as GeoTIFF using rasterio."""
        if not RASTERIO_AVAILABLE:
            raise ImportError("rasterio is required for GeoTIFF output")
        
        # Default parameters
        if extent is None:
            extent = (0, data.shape[1], 0, data.shape[0])
        if crs is None:
            crs = 'EPSG:4326'
        
        # Create transform
        transform = from_bounds(extent[0], extent[2], extent[1], extent[3],
                              data.shape[1], data.shape[0])
        
        # Write file
        with rasterio.open(
            file_path,
            'w',
            driver='GTiff',
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype=data.dtype,
            crs=crs,
            transform=transform,
            nodata=np.nan
        ) as dst:
            dst.write(data, 1)
        
        logger.info(f"Wrote GeoTIFF: {file_path}")
    
    def _write_numpy(self, data: np.ndarray, file_path: Path) -> None:
        """Write data as numpy array."""
        np.save(file_path, data)
        logger.info(f"Wrote numpy array: {file_path}")
    
    def _write_ascii(self, data: np.ndarray, file_path: Path,
                    extent: Optional[Tuple[float, float, float, float]] = None) -> None:
        """Write data as ASCII grid."""
        with open(file_path, 'w') as f:
            # Write header
            f.write(f"ncols {data.shape[1]}\n")
            f.write(f"nrows {data.shape[0]}\n")
            
            if extent is not None:
                cellsize_x = (extent[1] - extent[0]) / data.shape[1]
                cellsize_y = (extent[3] - extent[2]) / data.shape[0]
                cellsize = min(cellsize_x, cellsize_y)
                
                f.write(f"xllcorner {extent[0]}\n")
                f.write(f"yllcorner {extent[2]}\n")
                f.write(f"cellsize {cellsize}\n")
            else:
                f.write("xllcorner 0\n")
                f.write("yllcorner 0\n")
                f.write("cellsize 1\n")
            
            f.write("nodata -9999\n")
            
            # Write data
            for row in data:
                row_str = ' '.join(['-9999' if np.isnan(x) else f'{x:.6f}' for x in row])
                f.write(row_str + '\n')
        
        logger.info(f"Wrote ASCII grid: {file_path}")
    
    def get_grid_info(self, file_path: Union[str, Path]) -> Dict:
        """
        Get information about a grid file without loading the full data.
        
        Parameters:
            file_path (Union[str, Path]): Path to grid file
            
        Returns:
            Dict: Grid information
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Grid file not found: {file_path}")
        
        suffix = file_path.suffix.lower()
        
        if suffix in ['.tif', '.tiff'] and RASTERIO_AVAILABLE:
            with rasterio.open(file_path) as src:
                return {
                    'format': 'GeoTIFF',
                    'shape': (src.height, src.width),
                    'dtype': src.dtypes[0],
                    'crs': str(src.crs) if src.crs else None,
                    'extent': src.bounds,
                    'nodata': src.nodata,
                    'bands': src.count
                }
        
        elif suffix == '.npy':
            # For numpy files, we need to load to get info
            data = np.load(file_path)
            return {
                'format': 'NumPy',
                'shape': data.shape,
                'dtype': str(data.dtype),
                'size_mb': data.nbytes / (1024 * 1024)
            }
        
        else:
            return {'format': 'Unknown', 'file_size': file_path.stat().st_size}