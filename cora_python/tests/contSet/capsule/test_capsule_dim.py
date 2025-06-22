"""
test_capsule_dim - unit tests for capsule/dim

Syntax:
    python -m pytest cora_python/tests/contSet/capsule/test_capsule_dim.py

Authors: Python translation by AI Assistant
Written: 2025
"""

import numpy as np
import pytest
from cora_python.contSet.capsule import Capsule


class TestCapsuleDim:
    """Test class for capsule dim method"""
    
    def test_dim_2d(self):
        """Test dimension of 2D capsule"""
        c = np.array([[1], [2]])
        g = np.array([[0.5], [-1]])
        r = 0.5
        C = Capsule(c, g, r)
        
        assert C.dim() == 2
        
    def test_dim_3d(self):
        """Test dimension of 3D capsule"""
        c = np.array([[1], [1], [0]])
        g = np.array([[0.5], [-1], [1]])
        r = 0.5
        C = Capsule(c, g, r)
        
        assert C.dim() == 3
        
    def test_dim_1d(self):
        """Test dimension of 1D capsule"""
        c = np.array([[2]])
        g = np.array([[1]])
        r = 0.1
        C = Capsule(c, g, r)
        
        assert C.dim() == 1


if __name__ == '__main__':
    pytest.main([__file__]) 