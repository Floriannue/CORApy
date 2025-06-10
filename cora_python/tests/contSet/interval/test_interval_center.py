"""
test_interval_center - unit test function for interval center method

This module tests the center method for intervals, following the MATLAB
test patterns from test_interval_center.m.

Authors: Python translation by AI Assistant
Written: 2025
"""

import pytest
import numpy as np
import sys
import os

# Add the parent directory to the path to import cora_python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from cora_python.contSet import Interval
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAError


class TestIntervalCenter:
    """Test class for interval center method"""
    
    def test_center_empty_set(self):
        """Test center of empty set"""
        n = 2
        I = Interval.empty(n)
        c = I.center()
        
        assert c.size == 0
        assert c.shape == (n, 0)
        assert isinstance(c, np.ndarray)
    
    def test_center_bounded(self):
        """Test center of bounded intervals"""
        tol = 1e-9
        
        I = Interval([-5.0, -4.0, -3, 0, 0, 5], [-2, 0.0, 2.0, 0, 5, 8])
        c = I.center()
        c_true = np.array([-3.5, -2, -0.5, 0, 2.5, 6.5])
        
        assert np.allclose(c, c_true, atol=tol)
    
    def test_center_partially_unbounded(self):
        """Test center of partially unbounded intervals"""
        # Left unbounded
        I = Interval([-np.inf], [2])
        c = I.center()
        assert np.isnan(c[0])
        
        # Right unbounded
        I = Interval([2], [np.inf])
        c = I.center()
        assert np.isnan(c[0])
    
    def test_center_fully_unbounded(self):
        """Test center of fully unbounded intervals"""
        I = Interval([-np.inf], [np.inf])
        c = I.center()
        assert c[0] == 0
    
    def test_center_scalar_Interval(self):
        """Test center of scalar intervals"""
        # Regular scalar
        I = Interval([5], [7])
        c = I.center()
        assert np.isclose(c[0], 6.0)
        
        # Point scalar
        I = Interval([3], [3])
        c = I.center()
        assert np.isclose(c[0], 3.0)
        
        # Unbounded scalar
        I = Interval([-np.inf], [np.inf])
        c = I.center()
        assert c[0] == 0
    
    def test_center_point_Interval(self):
        """Test center of point intervals"""
        I = Interval([1, 2, 3], [1, 2, 3])
        c = I.center()
        expected = np.array([1, 2, 3])
        assert np.allclose(c, expected)
    
    def test_center_matrix_intervals(self):
        """Test center of matrix intervals"""
        # 2x2 matrix
        inf_mat = np.array([[-2, -1], [0, 2]])
        sup_mat = np.array([[3, 5], [2, 3]])
        I = Interval(inf_mat, sup_mat)
        c = I.center()
        
        expected = (inf_mat + sup_mat) / 2
        assert np.allclose(c, expected)
    
    def test_center_nd_arrays(self):
        """Test center of n-dimensional arrays"""
        # Create 2x2x2x3 arrays as in MATLAB test
        lb = np.array([1.000, 3.000, 2.000, 5.000, -3.000, 0.000, 2.000, 1.000, 
                      0.000, -2.000, -1.000, 3.000, 0.000, 0.000, 0.000, 0.000, 
                      1.000, -1.000, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000]).reshape(2, 2, 2, 3)
        ub = np.array([1.500, 4.000, 4.000, 10.000, -1.000, 0.000, 3.000, 2.000, 
                      1.000, 0.000, 2.000, 4.000, 0.000, 0.000, 0.000, 0.000, 
                      2.000, -0.500, 3.000, 2.000, 0.000, 0.000, 0.000, 0.000]).reshape(2, 2, 2, 3)
        
        I = Interval(lb, ub)
        c = I.center()
        c_true = (lb + ub) / 2
        
        assert np.allclose(c, c_true)
    
    def test_center_mixed_bounded_unbounded(self):
        """Test center with mixed bounded and unbounded dimensions"""
        I = Interval([-np.inf, -2, 1], [2, np.inf, 3])
        c = I.center()
        
        assert np.isnan(c[0])  # -inf to 2
        assert np.isnan(c[1])  # -2 to inf
        assert np.isclose(c[2], 2.0)  # 1 to 3
    
    def test_center_zero_width_intervals(self):
        """Test center of zero-width intervals"""
        I = Interval([0, -1, 5], [0, -1, 5])
        c = I.center()
        expected = np.array([0, -1, 5])
        assert np.allclose(c, expected)
    
    def test_center_large_intervals(self):
        """Test center with large values"""
        I = Interval([-1e10, -1e6], [1e10, 1e6])
        c = I.center()
        expected = np.array([0, 0])
        assert np.allclose(c, expected)
    
    def test_center_small_intervals(self):
        """Test center with very small intervals"""
        I = Interval([1e-10, -1e-10], [2e-10, 1e-10])
        c = I.center()
        expected = np.array([1.5e-10, 0])
        assert np.allclose(c, expected)
    
    def test_center_consistency(self):
        """Test that center is consistent with inf and sup"""
        I = Interval([-3, -1, 0, 2], [5, 4, 0, 8])
        c = I.center()
        
        # Center should be exactly halfway between inf and sup
        expected = (I.inf + I.sup) / 2
        assert np.allclose(c, expected)
        
        # For each dimension, center should be between inf and sup
        assert np.all(c >= I.inf)
        assert np.all(c <= I.sup)
    
    def test_center_origin_Interval(self):
        """Test center of origin interval"""
        I = Interval.origin(3)
        c = I.center()
        expected = np.array([0, 0, 0])
        assert np.allclose(c, expected)
    
    def test_center_symmetric_intervals(self):
        """Test center of symmetric intervals"""
        # Symmetric around origin
        I = Interval([-2, -3, -1], [2, 3, 1])
        c = I.center()
        expected = np.array([0, 0, 0])
        assert np.allclose(c, expected)
        
        # Symmetric around other point
        I = Interval([3, 7], [7, 11])
        c = I.center()
        expected = np.array([5, 9])
        assert np.allclose(c, expected)


if __name__ == '__main__':
    pytest.main([__file__]) 
