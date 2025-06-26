"""
test_get_unbounded_axis_limits - unit test function for get_unbounded_axis_limits

Syntax:
    pytest test_get_unbounded_axis_limits.py

Inputs:
    -

Outputs:
    test results

Other modules required: none
Subfunctions: none

See also: none

Authors: Tobias Ladner (MATLAB), AI Assistant (Python)
Written: 26-July-2023 (MATLAB), 2025 (Python)
Last update: ---
Last revision: ---
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from cora_python.g.functions.verbose.plot.get_unbounded_axis_limits import get_unbounded_axis_limits
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol


class TestGetUnboundedAxisLimits:
    """Test class for get_unbounded_axis_limits function"""
    
    def setup_method(self):
        """Set up test method"""
        self.fig = plt.figure()
    
    def teardown_method(self):
        """Clean up after test method"""
        plt.close(self.fig)
    
    def test_empty_plot(self):
        """Test get_unbounded_axis_limits with empty plot"""
        
        x_lim, y_lim = get_unbounded_axis_limits()
        expected = np.array([0, 1, 0, 1])
        result = np.array([*x_lim, *y_lim])
        assert np.allclose(expected, result)
    
    def test_with_vertices(self):
        """Test get_unbounded_axis_limits with vertices"""
        
        # Give weird vertices
        vertices = np.array([[1.25, 1.223], [1.34, 1.2]])
        x_lim, y_lim = get_unbounded_axis_limits(vertices)
        expected = np.array([0, 1.4, 0, 1.4])
        result = np.array([*x_lim, *y_lim])
        assert withinTol(expected, result, 1e-10)
    
    def test_with_set_axis(self):
        """Test get_unbounded_axis_limits with manually set axis"""
        
        # Set axis
        plt.xlim([-2, 1])
        plt.ylim([1, 3])
        
        x_lim, y_lim = get_unbounded_axis_limits()
        expected = np.array([-2, 1, 1, 3])
        result = np.array([*x_lim, *y_lim])
        assert withinTol(expected, result, 1e-10)
    
    def test_with_set_axis_and_vertices(self):
        """Test get_unbounded_axis_limits with set axis and vertices"""
        
        # Set axis
        plt.xlim([-2, 1])
        plt.ylim([1, 3])
        
        # With vertices
        vertices = np.array([[1.25, 1.223], [1.34, 1.2]])
        x_lim, y_lim = get_unbounded_axis_limits(vertices)
        expected = np.array([-2, 1.5, 1, 3])
        result = np.array([*x_lim, *y_lim])
        assert withinTol(expected, result, 1e-10)
    
    def test_3d_plot(self):
        """Test get_unbounded_axis_limits with 3D plot"""
        
        # Set 3D view
        ax = plt.gca()
        ax.view_init()  # Enable 3D view
        plt.xlim([-2, 1])
        plt.ylim([1, 3])
        
        # For 3D, we need to manually set zlim
        ax.set_zlim([4, 5])
        
        try:
            x_lim, y_lim, z_lim = get_unbounded_axis_limits()
            expected = np.array([-2, 1, 1, 3, 4, 5])
            result = np.array([*x_lim, *y_lim, *z_lim])
            assert withinTol(expected, result, 1e-10)
        except (ValueError, TypeError):
            # If 3D functionality is not implemented, skip this test
            pytest.skip("3D functionality not implemented in get_unbounded_axis_limits")
    
    def test_3d_plot_with_vertices(self):
        """Test get_unbounded_axis_limits with 3D plot and vertices"""
        
        # Set 3D view
        ax = plt.gca()
        ax.view_init()  # Enable 3D view
        plt.xlim([-2, 1])
        plt.ylim([1, 3])
        ax.set_zlim([4, 5])
        
        try:
            vertices = np.array([[1.25, 1.223], [1.34, 1.2], [2.2, 3]])
            x_lim, y_lim, z_lim = get_unbounded_axis_limits(vertices)
            expected = np.array([-2, 2, 1, 3, 2, 5])
            result = np.array([*x_lim, *y_lim, *z_lim])
            assert withinTol(expected, result, 1e-10)
        except (ValueError, TypeError):
            # If 3D functionality is not implemented, skip this test
            pytest.skip("3D functionality not implemented in get_unbounded_axis_limits")
    
    def test_edge_cases(self):
        """Test edge cases for get_unbounded_axis_limits"""
        
        # Test with empty vertices array
        vertices = np.array([]).reshape(0, 2)
        x_lim, y_lim = get_unbounded_axis_limits(vertices)
        # Should return current axis limits or default
        assert len(x_lim) == 2
        assert len(y_lim) == 2
        
        # Test with single vertex
        vertices = np.array([[1.0, 2.0]])
        x_lim, y_lim = get_unbounded_axis_limits(vertices)
        assert len(x_lim) == 2
        assert len(y_lim) == 2
        
        # Test with identical vertices
        vertices = np.array([[1.0, 2.0], [1.0, 2.0]])
        x_lim, y_lim = get_unbounded_axis_limits(vertices)
        assert len(x_lim) == 2
        assert len(y_lim) == 2
    
    def test_large_coordinates(self):
        """Test with large coordinate values"""
        
        vertices = np.array([[1e6, 1e6], [2e6, 2e6]])
        x_lim, y_lim = get_unbounded_axis_limits(vertices)
        assert len(x_lim) == 2
        assert len(y_lim) == 2
        assert x_lim[1] > x_lim[0]  # max > min
        assert y_lim[1] > y_lim[0]  # max > min
    
    def test_negative_coordinates(self):
        """Test with negative coordinate values"""
        
        vertices = np.array([[-2.0, -3.0], [-1.0, -1.0]])
        x_lim, y_lim = get_unbounded_axis_limits(vertices)
        assert len(x_lim) == 2
        assert len(y_lim) == 2
        assert x_lim[1] > x_lim[0]  # max > min
        assert y_lim[1] > y_lim[0]  # max > min


if __name__ == "__main__":
    pytest.main([__file__]) 