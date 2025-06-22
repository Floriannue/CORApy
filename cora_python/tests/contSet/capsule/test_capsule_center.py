"""
test_capsule_center - unit test function for capsule center method

Tests the center method of capsule objects.

Syntax:
    pytest cora_python/tests/contSet/capsule/test_capsule_center.py

Authors: Florian Lercher (translated from MATLAB)
Date: 2025-01-11
"""

import pytest
import numpy as np
from cora_python.contSet.capsule.capsule import Capsule


class TestCapsuleCenter:
    def test_center_empty_capsule(self):
        """Test center of empty capsule"""
        n = 2
        C = Capsule.empty(n)
        c = C.center()
        
        # Empty capsule should return empty center with correct dimensions
        assert c.size == 0
        assert c.shape == (2, 0)

    def test_center_3d_capsule(self):
        """Test center of 3D capsule"""
        c_true = np.array([[2], [0], [-1]])
        g = np.array([[1], [-1], [2]])
        r = 0.5
        
        C = Capsule(c_true.flatten(), g.flatten(), r)
        c = C.center()
        
        assert np.allclose(c, c_true)

    def test_center_2d_capsule(self):
        """Test center of 2D capsule"""
        c_true = np.array([[1], [3]])
        g = np.array([[2], [-1]])
        r = 1.0
        
        C = Capsule(c_true.flatten(), g.flatten(), r)
        c = C.center()
        
        assert np.allclose(c, c_true)

    def test_center_point_capsule(self):
        """Test center of point capsule (r=0, g=0)"""
        c_true = np.array([[5], [-2], [1]])
        g = np.array([[0], [0], [0]])
        r = 0
        
        C = Capsule(c_true.flatten(), g.flatten(), r)
        c = C.center()
        
        assert np.allclose(c, c_true)

    def test_center_1d_capsule(self):
        """Test center of 1D capsule"""
        c_true = np.array([[7]])
        g = np.array([[3]])
        r = 0.2
        
        C = Capsule(c_true.flatten(), g.flatten(), r)
        c = C.center()
        
        assert np.allclose(c, c_true)

    def test_center_origin_capsule(self):
        """Test center of origin capsule"""
        n = 3
        C = Capsule.origin(n)
        c = C.center()
        
        expected_c = np.zeros((n, 1))
        assert np.allclose(c, expected_c)

    def test_center_high_dimension(self):
        """Test center of high-dimensional capsule"""
        n = 10
        c_true = np.random.rand(n, 1)
        g = np.random.rand(n, 1)
        r = 0.1
        
        C = Capsule(c_true.flatten(), g.flatten(), r)
        c = C.center()
        
        assert np.allclose(c, c_true)

    def test_center_return_type(self):
        """Test that center returns correct array type and shape"""
        c_input = np.array([1, 2, 3])
        g_input = np.array([0, 1, 0])
        r = 0.5
        
        C = Capsule(c_input, g_input, r)
        c = C.center()
        
        # Should return column vector
        assert c.ndim == 2
        assert c.shape[1] == 1
        assert c.shape[0] == len(c_input)


if __name__ == "__main__":
    pytest.main([__file__]) 