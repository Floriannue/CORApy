"""
test_interval_rad - unit test function for interval rad method

This module tests the rad (radius) method for intervals, following the MATLAB
test patterns from test_interval_rad.m.

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
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


class TestIntervalRad:
    """Test class for interval rad method"""
    
    def test_rad_basic(self):
        """Test basic radius calculation"""
        I = Interval([-1, 1], [1, 2])
        r = I.rad()
        expected = np.array([1.0, 0.5])
        assert np.allclose(r, expected)
    
    def test_rad_point_Interval(self):
        """Test radius of point intervals"""
        I = Interval([1, 2, 3], [1, 2, 3])
        r = I.rad()
        expected = np.array([0, 0, 0])
        assert np.allclose(r, expected)
    
    def test_rad_scalar_Interval(self):
        """Test radius of scalar intervals"""
        # Regular scalar
        I = Interval([5], [7])
        r = I.rad()
        assert np.isclose(r[0], 1.0)
        
        # Point scalar
        I = Interval([3], [3])
        r = I.rad()
        assert np.isclose(r[0], 0.0)
    
    def test_rad_matrix_intervals(self):
        """Test radius of matrix intervals"""
        # 2x2 matrix
        inf_mat = np.array([[-2, -1], [0, 2]])
        sup_mat = np.array([[3, 5], [2, 3]])
        I = Interval(inf_mat, sup_mat)
        r = I.rad()
        
        expected = (sup_mat - inf_mat) / 2
        assert np.allclose(r, expected)
    
    def test_rad_unbounded_intervals(self):
        """Test radius of unbounded intervals"""
        # Left unbounded
        I = Interval([-np.inf], [2])
        r = I.rad()
        assert np.isinf(r[0])
        
        # Right unbounded
        I = Interval([2], [np.inf])
        r = I.rad()
        assert np.isinf(r[0])
        
        # Fully unbounded
        I = Interval([-np.inf], [np.inf])
        r = I.rad()
        assert np.isinf(r[0])
    
    def test_rad_mixed_bounded_unbounded(self):
        """Test radius with mixed bounded and unbounded dimensions"""
        I = Interval([-np.inf, -2, 1], [2, np.inf, 3])
        r = I.rad()
        
        assert np.isinf(r[0])  # -inf to 2
        assert np.isinf(r[1])  # -2 to inf
        assert np.isclose(r[2], 1.0)  # 1 to 3
    
    def test_rad_large_intervals(self):
        """Test radius with large values"""
        I = Interval([-1e10, -1e6], [1e10, 1e6])
        r = I.rad()
        expected = np.array([1e10, 1e6])
        assert np.allclose(r, expected)
    
    def test_rad_small_intervals(self):
        """Test radius with very small intervals"""
        I = Interval([1e-10, -1e-10], [2e-10, 1e-10])
        r = I.rad()
        expected = np.array([0.5e-10, 1e-10])
        assert np.allclose(r, expected)
    
    def test_rad_zero_width_intervals(self):
        """Test radius of zero-width intervals"""
        I = Interval([0, -1, 5], [0, -1, 5])
        r = I.rad()
        expected = np.array([0, 0, 0])
        assert np.allclose(r, expected)
    
    def test_rad_symmetric_intervals(self):
        """Test radius of symmetric intervals"""
        # Symmetric around origin
        I = Interval([-2, -3, -1], [2, 3, 1])
        r = I.rad()
        expected = np.array([2, 3, 1])
        assert np.allclose(r, expected)
        
        # Symmetric around other point
        I = Interval([3, 7], [7, 11])
        r = I.rad()
        expected = np.array([2, 2])
        assert np.allclose(r, expected)
    
    def test_rad_consistency(self):
        """Test that radius is consistent with inf and sup"""
        I = Interval([-3, -1, 0, 2], [5, 4, 0, 8])
        r = I.rad()
        
        # Radius should be exactly half the width
        expected = (I.sup - I.inf) / 2
        assert np.allclose(r, expected)
        
        # Radius should always be non-negative
        assert np.all(r >= 0)
    
    def test_rad_origin_Interval(self):
        """Test radius of origin interval"""
        I = Interval.origin(3)
        r = I.rad()
        expected = np.array([0, 0, 0])
        assert np.allclose(r, expected)
    
    def test_rad_empty_Interval(self):
        """Test radius of empty interval"""
        n = 2
        I = Interval.empty(n)
        r = I.rad()
        
        # Empty interval should have empty radius
        assert r.size == 0
        assert r.shape == (n, 0)
    
    def test_rad_fullspace_Interval(self):
        """Test radius of fullspace interval"""
        I = Interval.Inf(2)
        r = I.rad()
        
        # Fullspace should have infinite radius
        assert np.all(np.isinf(r))
    
    def test_rad_nd_arrays(self):
        """Test radius of n-dimensional arrays"""
        # Create 2x2x2 arrays
        lb = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        ub = np.array([[[3, 4], [5, 6]], [[7, 8], [9, 10]]])
        
        I = Interval(lb, ub)
        r = I.rad()
        r_true = (ub - lb) / 2
        
        assert np.allclose(r, r_true)
    
    def test_rad_relationship_with_center(self):
        """Test relationship between radius and center"""
        I = Interval([-5, -2, 1], [3, 4, 7])
        r = I.rad()
        c = I.center()
        
        # Check that inf = center - radius and sup = center + radius
        assert np.allclose(I.inf, c - r)
        assert np.allclose(I.sup, c + r)
    
    def test_rad_positive_values(self):
        """Test that radius is always non-negative"""
        # Various interval types
        intervals = [
            Interval([-5, -2], [3, 4]),
            Interval([0, 0], [0, 0]),
            Interval([1, 2], [1, 2]),
            Interval([-10], [10]),
            Interval([5], [15])
        ]
        
        for I in intervals:
            r = I.rad()
            assert np.all(r >= 0), f"Radius should be non-negative, got {r}"


if __name__ == '__main__':
    pytest.main([__file__]) 
