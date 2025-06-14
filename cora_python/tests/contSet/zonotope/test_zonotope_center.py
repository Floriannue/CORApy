"""
test_zonotope_center - unit test function of center

Tests the center method for zonotope objects.

Syntax:
    pytest cora_python/tests/contSet/zonotope/test_zonotope_center.py

Authors: Florian Lercher (translated from MATLAB)
Date: 2025-01-11
"""

import pytest
import numpy as np
from cora_python.contSet.zonotope.zonotope import Zonotope


class TestZonotopeCenter:
    def test_center_basic(self):
        """Test basic center functionality"""
        # Simple 2D zonotope
        c = np.array([[1], [2]])
        G = np.array([[1, 0], [0, 1]])
        Z = Zonotope(c, G)
        
        center_result = Z.center()
        assert np.allclose(center_result, c)

    def test_center_different_dimensions(self):
        """Test center for different dimensions"""
        # 1D case
        c_1d = np.array([[5]])
        G_1d = np.array([[2]])
        Z_1d = Zonotope(c_1d, G_1d)
        assert np.allclose(Z_1d.center(), c_1d)
        
        # 3D case
        c_3d = np.array([[1], [2], [3]])
        G_3d = np.array([[1, 0, 2], [0, 1, 1], [0, 0, 1]])
        Z_3d = Zonotope(c_3d, G_3d)
        assert np.allclose(Z_3d.center(), c_3d)

    def test_center_empty_zonotope(self):
        """Test center of empty zonotope"""
        Z_empty = Zonotope.empty(2)
        center_result = Z_empty.center()
        
        # Empty zonotope should return empty center
        assert center_result.size == 0
        assert center_result.shape == (2, 0)

    def test_center_point_zonotope(self):
        """Test center of point zonotope (no generators)"""
        c = np.array([[3], [-1], [2]])
        Z_point = Zonotope(c)
        
        center_result = Z_point.center()
        assert np.allclose(center_result, c)

    def test_center_return_type(self):
        """Test that center returns the stored center vector"""
        c = np.array([[1.5], [-2.3], [0.7]])
        G = np.array([[1, 2], [0, 1], [1, 0]])
        Z = Zonotope(c, G)
        
        center_result = Z.center()
        
        # Should return exactly the center vector
        assert np.array_equal(center_result, c)
        assert center_result.shape == c.shape

    def test_center_with_large_generators(self):
        """Test that center is independent of generators"""
        c = np.array([[0], [0]])
        G_small = np.array([[0.1, 0.05], [0.05, 0.1]])
        G_large = np.array([[10, 5], [5, 10]])
        
        Z_small = Zonotope(c, G_small)
        Z_large = Zonotope(c, G_large)
        
        # Center should be the same regardless of generator size
        assert np.allclose(Z_small.center(), Z_large.center())
        assert np.allclose(Z_small.center(), c)


if __name__ == "__main__":
    pytest.main([__file__]) 