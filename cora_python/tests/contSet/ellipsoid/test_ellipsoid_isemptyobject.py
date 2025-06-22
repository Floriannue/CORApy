"""
test_ellipsoid_isemptyobject - unit tests for ellipsoid/isemptyobject

Syntax:
    python -m pytest cora_python/tests/contSet/ellipsoid/test_ellipsoid_isemptyobject.py

Authors: Python translation by AI Assistant
Written: 2025
"""

import numpy as np
import pytest
from cora_python.contSet.ellipsoid import Ellipsoid


class TestEllipsoidIsEmptyObject:
    """Test class for ellipsoid isemptyobject method"""
    
    def test_isemptyobject_false_normal(self):
        """Test isemptyobject returns False for normal ellipsoid"""
        Q = np.array([[2.7, -0.2], [-0.2, 2.4]])
        q = np.array([[1], [2]])
        E = Ellipsoid(Q, q)
        
        assert E.isemptyobject() == False
        
    def test_isemptyobject_false_unit(self):
        """Test isemptyobject returns False for unit ellipsoid"""
        Q = np.eye(2)
        q = np.zeros((2, 1))
        E = Ellipsoid(Q, q)
        
        assert E.isemptyobject() == False
        
    def test_isemptyobject_true_empty(self):
        """Test isemptyobject returns True for empty ellipsoid"""
        Q = np.array([]).reshape(0, 0)
        q = np.array([]).reshape(0, 1)
        E = Ellipsoid(Q, q)
        
        # For empty ellipsoids, this should return True
        # The function checks if Q and q are empty and TOL is default
        assert E.isemptyobject() == True
        
    def test_isemptyobject_false_1d(self):
        """Test isemptyobject returns False for 1D ellipsoid"""
        Q = np.array([[1.0]])
        q = np.array([[0.0]])
        E = Ellipsoid(Q, q)
        
        assert E.isemptyobject() == False


if __name__ == '__main__':
    pytest.main([__file__]) 