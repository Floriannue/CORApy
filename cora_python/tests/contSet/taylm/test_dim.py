"""
test_dim - unit test function for taylm dim

Tests the dimension functionality for Taylm objects.

Authors:       Tobias Ladner (MATLAB)
               Python translation by AI Assistant
Written:       02-August-2023
Last update:   ---
Last revision: ---
"""

import pytest
import numpy as np
from cora_python.contSet.taylm.taylm import Taylm
from cora_python.contSet.interval.interval import Interval


class TestTaylmDim:
    """Test class for taylm dim method"""

    def test_dim_scalar_case(self):
        """Test dimension of scalar taylor model"""
        # Scalar case
        tay = Taylm(Interval(np.array([1]), np.array([1])))
        assert tay.dim() == 1

    def test_dim_higher_dimensional(self):
        """Test dimension of higher-dimensional taylor model"""
        # Higher-dimensional case
        lb = np.array([-3, -2, -5])
        ub = np.array([4, 2, 1])
        I = Interval(lb, ub)
        tay = Taylm(I)
        
        # Compute dimension
        assert tay.dim() == 3

    def test_dim_various_dimensions(self):
        """Test dimension for various sizes"""
        for n in [1, 2, 4, 5, 10]:
            I = Interval(-np.ones(n), np.ones(n))
            tay = Taylm(I)
            assert tay.dim() == n

    def test_dim_return_type(self):
        """Test that dim returns integer"""
        I = Interval(np.array([0, 1]), np.array([2, 3]))
        tay = Taylm(I)
        
        result = tay.dim()
        assert isinstance(result, int)
        assert result == 2

    def test_dim_consistency(self):
        """Test that dim is consistent across calls"""
        I = Interval(np.array([1, 2, 3]), np.array([4, 5, 6]))
        tay = Taylm(I)
        
        # Multiple calls should return same result
        dim1 = tay.dim()
        dim2 = tay.dim()
        dim3 = tay.dim()
        
        assert dim1 == dim2 == dim3 == 3

    def test_dim_large_dimension(self):
        """Test dimension for large dimensions"""
        n = 50
        I = Interval(np.zeros(n), np.ones(n))
        tay = Taylm(I)
        assert tay.dim() == n 