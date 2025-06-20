"""
Geophysical Colormaps
====================

Specialized colormaps for geophysical data visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Dict, List, Tuple


class GeophysicalColormaps:
    """
    Collection of colormaps optimized for geophysical data visualization.
    """
    
    @staticmethod
    def magnetic_anomaly() -> mcolors.LinearSegmentedColormap:
        """
        Colormap for magnetic anomaly data (diverging, red-blue).
        
        Returns:
            matplotlib.colors.LinearSegmentedColormap: Magnetic anomaly colormap
        """
        colors = ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0', 
                 '#f7f7f7', '#fdbf6f', '#fd8d3c', '#e31a1c', '#b10026']
        return mcolors.LinearSegmentedColormap.from_list('magnetic_anomaly', colors)
    
    @staticmethod
    def gravity_anomaly() -> mcolors.LinearSegmentedColormap:
        """
        Colormap for gravity anomaly data (blue-white-red).
        
        Returns:
            matplotlib.colors.LinearSegmentedColormap: Gravity anomaly colormap
        """
        colors = ['#053061', '#2166ac', '#4393c3', '#92c5de', '#d1e5f0',
                 '#ffffff', '#fdbf6f', '#fd8d3c', '#e31a1c', '#67001f']
        return mcolors.LinearSegmentedColormap.from_list('gravity_anomaly', colors)
    
    @staticmethod
    def gradient_magnitude() -> mcolors.LinearSegmentedColormap:
        """
        Colormap for gradient magnitude data (sequential, white to dark).
        
        Returns:
            matplotlib.colors.LinearSegmentedColormap: Gradient magnitude colormap
        """
        colors = ['#ffffff', '#f0f0f0', '#d9d9d9', '#bdbdbd', 
                 '#969696', '#737373', '#525252', '#252525']
        return mcolors.LinearSegmentedColormap.from_list('gradient_magnitude', colors)
    
    @staticmethod
    def tilt_angle() -> mcolors.LinearSegmentedColormap:
        """
        Colormap for tilt angle data (diverging around zero).
        
        Returns:
            matplotlib.colors.LinearSegmentedColormap: Tilt angle colormap
        """
        colors = ['#5e3c99', '#b2abd2', '#e66101', '#fdb863', 
                 '#ffffff', '#e66101', '#b2abd2', '#5e3c99']
        return mcolors.LinearSegmentedColormap.from_list('tilt_angle', colors)
    
    @staticmethod
    def get_colormap_dict() -> Dict[str, str]:
        """
        Get dictionary of recommended colormaps for different data types.
        
        Returns:
            Dict[str, str]: Mapping of data types to colormap names
        """
        return {
            'magnetic_field': 'RdYlBu_r',
            'magnetic_anomaly': 'RdBu_r', 
            'gravity_anomaly': 'RdBu_r',
            'total_horizontal_gradient': 'viridis',
            'analytic_signal': 'plasma',
            'tilt_angle': 'RdBu_r',
            'rtp': 'RdYlBu_r',
            'rte': 'RdYlBu_r',
            'upward_continuation': 'RdYlBu_r',
            'vertical_integration': 'RdYlBu_r',
            'high_pass': 'RdBu_r',
            'low_pass': 'RdYlBu_r'
        }