"""
test_ellipsoid_center - unit tests for ellipsoid/center

Syntax:
    python -m pytest cora_python/tests/contSet/ellipsoid/test_ellipsoid_center.py

Authors: Python translation by AI Assistant
Written: 2025
"""

import numpy as np
import pytest
from cora_python.contSet.ellipsoid import Ellipsoid


class TestEllipsoidCenter:
    """Test class for ellipsoid center method"""
    
    def test_center_2d(self):
        """Test center of 2D ellipsoid"""
        Q = np.array([[2.7, -0.2], [-0.2, 2.4]])
        q = np.array([[1], [2]])
        E = Ellipsoid(Q, q)
        
        center = E.center()
        expected = np.array([[1], [2]])
        np.testing.assert_array_equal(center, expected)
        
    def test_center_origin(self):
        """Test center at origin"""
        Q = np.eye(3)
        q = np.zeros((3, 1))
        E = Ellipsoid(Q, q)
        
        center = E.center()
        expected = np.zeros((3, 1))
        np.testing.assert_array_equal(center, expected)
        
    def test_center_1d(self):
        """Test center of 1D ellipsoid"""
        Q = np.array([[4.0]])
        q = np.array([[-1.5]])
        E = Ellipsoid(Q, q)
        
        center = E.center()
        expected = np.array([[-1.5]])
        np.testing.assert_array_equal(center, expected)

    def test_center_empty(self):
        """Test center of an empty ellipsoid"""
        n = 2
        E_empty = Ellipsoid.empty(n)
        c = E_empty.center()
        assert c.shape == (n, 0)
        assert c.dtype == np.float64 # Ensure it's a numeric empty array


if __name__ == '__main__':
    pytest.main([__file__]) 