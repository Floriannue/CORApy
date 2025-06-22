"""
test_ellipsoid_dim - unit tests for ellipsoid/dim

Syntax:
    python -m pytest cora_python/tests/contSet/ellipsoid/test_ellipsoid_dim.py

Authors: Python translation by AI Assistant
Written: 2025
"""

import numpy as np
import pytest
from cora_python.contSet.ellipsoid import Ellipsoid


class TestEllipsoidDim:
    """Test class for ellipsoid dim method"""
    
    def test_dim_2d(self):
        """Test dimension of 2D ellipsoid"""
        Q = np.array([[2.7, -0.2], [-0.2, 2.4]])
        q = np.array([[1], [2]])
        E = Ellipsoid(Q, q)
        
        assert E.dim() == 2
        
    def test_dim_3d(self):
        """Test dimension of 3D ellipsoid"""
        Q = np.eye(3)
        q = np.zeros((3, 1))
        E = Ellipsoid(Q, q)
        
        assert E.dim() == 3
        
    def test_dim_1d(self):
        """Test dimension of 1D ellipsoid"""
        Q = np.array([[1.0]])
        q = np.array([[0.0]])
        E = Ellipsoid(Q, q)
        
        assert E.dim() == 1
        
    def test_dim_empty(self):
        """Test dimension of empty ellipsoid"""
        Q = np.array([]).reshape(0, 0)
        q = np.array([]).reshape(0, 1)
        E = Ellipsoid(Q, q)
        
        assert E.dim() == 0


if __name__ == '__main__':
    pytest.main([__file__]) 