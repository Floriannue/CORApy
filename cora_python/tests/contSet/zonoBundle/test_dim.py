"""
test_dim - unit test function for zonoBundle dim

Tests the dimension functionality for ZonoBundle objects.

Authors:       Matthias Althoff (MATLAB)
               Python translation by AI Assistant
Written:       02-August-2023
Last update:   ---
Last revision: ---
"""

import pytest
import numpy as np
from cora_python.contSet.zonoBundle.zonoBundle import ZonoBundle
from cora_python.contSet.zonotope.zonotope import Zonotope


class TestZonoBundleDim:
    """Test class for zonoBundle dim method"""

    def test_dim_basic(self):
        """Test basic dimension computation"""
        Z1 = Zonotope(np.array([1, 2]), np.eye(2))
        Z2 = Zonotope(np.array([2, 3]), np.eye(2))
        zB = ZonoBundle([Z1, Z2])
        
        assert zB.dim() == 2

    def test_dim_various_dimensions(self):
        """Test dimension for various sizes"""
        for n in [1, 3, 5, 10]:
            Z1 = Zonotope(np.zeros(n), np.eye(n))
            Z2 = Zonotope(np.ones(n), np.eye(n))
            zB = ZonoBundle([Z1, Z2])
            assert zB.dim() == n

    def test_dim_single_zonotope(self):
        """Test dimension with single zonotope"""
        Z = Zonotope(np.array([1, 2, 3]), np.eye(3))
        zB = ZonoBundle([Z])
        assert zB.dim() == 3

    def test_dim_return_type(self):
        """Test that dim returns integer"""
        Z1 = Zonotope(np.array([0, 1]), np.eye(2))
        Z2 = Zonotope(np.array([1, 0]), np.eye(2))
        zB = ZonoBundle([Z1, Z2])
        
        result = zB.dim()
        assert isinstance(result, int)
        assert result == 2

    def test_dim_consistency(self):
        """Test that dim is consistent across calls"""
        Z1 = Zonotope(np.array([1, 2, 3]), np.eye(3))
        Z2 = Zonotope(np.array([2, 3, 4]), np.eye(3))
        zB = ZonoBundle([Z1, Z2])
        
        # Multiple calls should return same result
        dim1 = zB.dim()
        dim2 = zB.dim()
        dim3 = zB.dim()
        
        assert dim1 == dim2 == dim3 == 3

    def test_dim_multiple_zonotopes(self):
        """Test dimension with many zonotopes"""
        zonotopes = []
        n = 4
        for i in range(10):
            Z = Zonotope(np.random.randn(n), np.random.randn(n, 3))
            zonotopes.append(Z)
        
        zB = ZonoBundle(zonotopes)
        assert zB.dim() == n

    def test_dim_large_dimension(self):
        """Test dimension for large dimensions"""
        n = 50
        Z1 = Zonotope(np.zeros(n), np.eye(n))
        Z2 = Zonotope(np.ones(n), np.eye(n))
        zB = ZonoBundle([Z1, Z2])
        assert zB.dim() == n 