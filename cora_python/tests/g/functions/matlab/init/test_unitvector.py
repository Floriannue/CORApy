"""
test_unitvector - unit test function for unitvector

Tests the unitvector function for creating standard unit vectors.

Authors: Python translation by AI Assistant
Date: 2025
"""

import pytest
import numpy as np
from cora_python.g.functions.matlab.init.unitvector import unitvector


class TestUnitvector:
    def test_unitvector_basic(self):
        """Test basic unitvector functionality"""
        # Test 3D unit vectors
        v1 = unitvector(1, 3)
        expected1 = np.array([[1], [0], [0]])
        assert np.allclose(v1, expected1)
        
        v2 = unitvector(2, 3)
        expected2 = np.array([[0], [1], [0]])
        assert np.allclose(v2, expected2)
        
        v3 = unitvector(3, 3)
        expected3 = np.array([[0], [0], [1]])
        assert np.allclose(v3, expected3)
    
    def test_unitvector_2d(self):
        """Test 2D unit vectors"""
        v1 = unitvector(1, 2)
        expected1 = np.array([[1], [0]])
        assert np.allclose(v1, expected1)
        
        v2 = unitvector(2, 2)
        expected2 = np.array([[0], [1]])
        assert np.allclose(v2, expected2)
    
    def test_unitvector_1d(self):
        """Test 1D unit vector"""
        v = unitvector(1, 1)
        expected = np.array([[1]])
        assert np.allclose(v, expected)
    
    def test_unitvector_zero_dimension(self):
        """Test zero dimension case"""
        v = unitvector(1, 0)
        expected = np.array([]).reshape(0, 1)
        assert v.shape == expected.shape
        assert v.size == 0
    
    def test_unitvector_large_dimension(self):
        """Test larger dimensions"""
        n = 10
        for i in range(1, n+1):
            v = unitvector(i, n)
            # Check that only the i-th element is 1
            assert v[i-1, 0] == 1
            # Check that all other elements are 0
            mask = np.ones(n, dtype=bool)
            mask[i-1] = False
            assert np.all(v[mask, 0] == 0)
            # Check shape
            assert v.shape == (n, 1)
    
    def test_unitvector_properties(self):
        """Test mathematical properties of unit vectors"""
        n = 5
        for i in range(1, n+1):
            v = unitvector(i, n)
            # Check that it's a unit vector (norm = 1)
            assert np.allclose(np.linalg.norm(v), 1.0)
            # Check that it's sparse (only one non-zero element)
            assert np.sum(v != 0) == 1


if __name__ == "__main__":
    pytest.main([__file__]) 