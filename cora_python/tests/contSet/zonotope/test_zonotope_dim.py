"""
test_zonotope_dim - unit test function of dim

Tests the dim method for zonotope objects to check dimension calculation.

Syntax:
    pytest cora_python/tests/contSet/zonotope/test_zonotope_dim.py

Authors: Florian Lercher (translated from MATLAB)
Date: 2025-01-11
"""

import pytest
import numpy as np
from cora_python.contSet.zonotope.zonotope import Zonotope


class TestZonotopeDim:
    def test_dim_basic_2d(self):
        """Test dimension of basic 2D zonotope"""
        c = np.array([[1], [2]])
        G = np.array([[1, 0], [0, 1]])
        Z = Zonotope(c, G)
        assert Z.dim() == 2

    def test_dim_basic_3d(self):
        """Test dimension of basic 3D zonotope"""
        c = np.array([[1], [2], [3]])
        G = np.array([[1, 0, 2], [0, 1, 1], [0, 0, 1]])
        Z = Zonotope(c, G)
        assert Z.dim() == 3

    def test_dim_1d(self):
        """Test dimension of 1D zonotope"""
        c = np.array([[5]])
        G = np.array([[2]])
        Z = Zonotope(c, G)
        assert Z.dim() == 1

    def test_dim_high_dimension(self):
        """Test dimension of high-dimensional zonotope"""
        n = 10
        c = np.ones((n, 1))
        G = np.eye(n)
        Z = Zonotope(c, G)
        assert Z.dim() == n

    def test_dim_point_zonotope(self):
        """Test dimension of point zonotope (no generators)"""
        c = np.array([[3], [-1], [2], [0]])
        Z = Zonotope(c)
        assert Z.dim() == 4

    def test_dim_empty_zonotope(self):
        """Test dimension of empty zonotope"""
        Z_empty = Zonotope.empty(5)
        assert Z_empty.dim() == 5

    def test_dim_with_many_generators(self):
        """Test dimension with more generators than dimensions"""
        c = np.array([[0], [0]])
        G = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])  # 4 generators for 2D space
        Z = Zonotope(c, G)
        assert Z.dim() == 2

    def test_dim_with_fewer_generators(self):
        """Test dimension with fewer generators than dimensions"""
        c = np.array([[0], [0], [0]])
        G = np.array([[1, 2], [3, 4], [5, 6]])  # 2 generators for 3D space
        Z = Zonotope(c, G)
        assert Z.dim() == 3

    def test_dim_zero_generators(self):
        """Test dimension with zero generators"""
        c = np.array([[1], [2], [3]])
        G = np.zeros((3, 5))  # All zero generators
        Z = Zonotope(c, G)
        assert Z.dim() == 3

    def test_dim_from_center_only(self):
        """Test that dimension is determined by center vector"""
        # Dimension should be determined by center, not generators
        c = np.array([[1], [2], [3], [4], [5], [6]])
        Z = Zonotope(c)
        assert Z.dim() == 6

    def test_dim_consistency(self):
        """Test dimension consistency across operations"""
        c = np.array([[1], [2]])
        G = np.array([[1, 2], [3, 4]])
        Z = Zonotope(c, G)
        
        # Dimension should be consistent
        assert Z.dim() == 2
        assert Z.c.shape[0] == 2
        assert Z.G.shape[0] == 2

    def test_dim_return_type(self):
        """Test that dim returns integer"""
        c = np.array([[1], [2], [3]])
        G = np.array([[1], [2], [3]])
        Z = Zonotope(c, G)
        
        dim_result = Z.dim()
        assert isinstance(dim_result, int)
        assert dim_result == 3


if __name__ == "__main__":
    pytest.main([__file__]) 