"""
Basic tests for SGTool Python
=============================

Simple test suite to verify core functionality.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sgtool_py.core.geophysical_processor import GeophysicalProcessor
from sgtool_py.core.frequency_filters import FrequencyFilters
from sgtool_py.core.gradient_filters import GradientFilters


class TestGeophysicalProcessor:
    """Test core geophysical processor."""
    
    def setup_method(self):
        """Setup test data."""
        self.dx = 1.0
        self.dy = 1.0
        self.processor = GeophysicalProcessor(self.dx, self.dy)
        
        # Create simple test grid
        x = np.arange(50)
        y = np.arange(50)
        xx, yy = np.meshgrid(x, y)
        self.test_data = np.sin(0.1 * xx) * np.cos(0.1 * yy) + np.random.normal(0, 0.1, (50, 50))
    
    def test_initialization(self):
        """Test processor initialization."""
        assert self.processor.dx == 1.0
        assert self.processor.dy == 1.0
    
    def test_wavenumber_grids(self):
        """Test wavenumber grid creation."""
        kx, ky = self.processor.create_wavenumber_grids(50, 50)
        assert kx.shape == (50, 50)
        assert ky.shape == (50, 50)
    
    def test_fill_nan_values(self):
        """Test NaN value filling."""
        data_with_nan = self.test_data.copy()
        data_with_nan[10:15, 10:15] = np.nan
        
        filled = self.processor.fill_nan_values(data_with_nan)
        assert not np.any(np.isnan(filled))
    
    def test_rtp_filter(self):
        """Test Reduction to Pole filter."""
        result = self.processor.reduction_to_pole(self.test_data, 70.0, 2.0)
        assert result.shape == self.test_data.shape
        assert not np.all(np.isnan(result))
    
    def test_upward_continuation(self):
        """Test upward continuation."""
        result = self.processor.upward_continuation(self.test_data, 10.0)
        assert result.shape == self.test_data.shape
        # Upward continuation should smooth the data
        assert np.std(result) <= np.std(self.test_data)


class TestFrequencyFilters:
    """Test frequency domain filters."""
    
    def setup_method(self):
        """Setup test data."""
        self.dx = 1.0
        self.dy = 1.0
        self.filters = FrequencyFilters(self.dx, self.dy)
        
        # Create test data with known frequencies
        x = np.arange(100)
        y = np.arange(100)
        xx, yy = np.meshgrid(x, y)
        self.test_data = (np.sin(0.05 * xx) + np.sin(0.2 * xx) + 
                         np.cos(0.05 * yy) + np.cos(0.2 * yy))
    
    def test_high_pass_filter(self):
        """Test high-pass filter."""
        cutoff_wavelength = 20.0
        result = self.filters.high_pass_filter(self.test_data, cutoff_wavelength)
        assert result.shape == self.test_data.shape
        # High-pass should reduce low-frequency content
        assert np.mean(np.abs(result)) < np.mean(np.abs(self.test_data))
    
    def test_low_pass_filter(self):
        """Test low-pass filter."""
        cutoff_wavelength = 10.0
        result = self.filters.low_pass_filter(self.test_data, cutoff_wavelength)
        assert result.shape == self.test_data.shape
        # Low-pass should smooth the data
        assert np.std(result) <= np.std(self.test_data)
    
    def test_band_pass_filter(self):
        """Test band-pass filter."""
        result = self.filters.band_pass_filter(self.test_data, 30.0, 10.0)
        assert result.shape == self.test_data.shape


class TestGradientFilters:
    """Test gradient-based filters."""
    
    def setup_method(self):
        """Setup test data."""
        self.dx = 1.0
        self.dy = 1.0
        self.filters = GradientFilters(self.dx, self.dy)
        
        # Create test data with known gradients
        x = np.arange(50)
        y = np.arange(50)
        xx, yy = np.meshgrid(x, y)
        self.test_data = xx + 2 * yy + 0.1 * np.sin(0.2 * xx)
    
    def test_derivatives(self):
        """Test derivative calculations."""
        dx, dy, dz = self.filters.calculate_derivatives(self.test_data)
        assert dx.shape == self.test_data.shape
        assert dy.shape == self.test_data.shape
        assert dz.shape == self.test_data.shape
    
    def test_total_horizontal_gradient(self):
        """Test Total Horizontal Gradient."""
        thg = self.filters.total_horizontal_gradient(self.test_data)
        assert thg.shape == self.test_data.shape
        assert np.all(thg >= 0)  # THG should be non-negative
    
    def test_analytic_signal(self):
        """Test Analytic Signal."""
        analytic_sig = self.filters.analytic_signal(self.test_data)
        assert analytic_sig.shape == self.test_data.shape
        assert np.all(analytic_sig >= 0)  # AS should be non-negative
    
    def test_tilt_angle(self):
        """Test Tilt Angle calculation."""
        tilt = self.filters.tilt_angle_degrees(self.test_data)
        assert tilt.shape == self.test_data.shape
        # Tilt angle should be between -90 and 90 degrees
        assert np.all(np.abs(tilt[~np.isnan(tilt)]) <= 90)


if __name__ == '__main__':
    # Run basic tests if script is executed directly
    import subprocess
    import sys
    
    try:
        result = subprocess.run([sys.executable, '-m', 'pytest', __file__, '-v'], 
                              capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        sys.exit(result.returncode)
    except FileNotFoundError:
        print("pytest not available, running simple tests...")
        
        # Simple manual test runner
        test_classes = [TestGeophysicalProcessor, TestFrequencyFilters, TestGradientFilters]
        
        for test_class in test_classes:
            print(f"\nTesting {test_class.__name__}:")
            instance = test_class()
            instance.setup_method()
            
            # Get all test methods
            test_methods = [method for method in dir(instance) if method.startswith('test_')]
            
            for method_name in test_methods:
                try:
                    method = getattr(instance, method_name)
                    method()
                    print(f"  ✓ {method_name}")
                except Exception as e:
                    print(f"  ✗ {method_name}: {e}")
        
        print("\nBasic tests completed!")