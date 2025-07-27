"""
test_zonotope_plus - unit test function of plus

This module tests the plus method (Minkowski addition) for zonotope objects.

Authors:       Matthias Althoff, Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       26-July-2016 (MATLAB)
Python translation: 2025
"""

import pytest
import numpy as np
from cora_python.contSet.zonotope import Zonotope

def test_zonotope_plus():
    """Unit test function of plus - mirrors MATLAB test_zonotope_plus.m"""
    
    # 2D zonotopes
    Z1 = Zonotope(np.array([[-4], [1]]), np.array([[-3, -2, -1], [2, 3, 4]]))
    Z2 = Zonotope(np.array([[1], [-1]]), np.array([[10], [-10]]))
    Z_plus = Z1 + Z2  # Using + operator as in MATLAB
    
    # compare to true result
    c_plus = np.array([[-3], [0]])
    G_plus = np.array([[-3, -2, -1, 10], [2, 3, 4, -10]])
    
    assert np.allclose(Z_plus.c, c_plus)
    assert np.allclose(Z_plus.G, G_plus)
    
    # Minkowski sum with empty set
    Z_empty = Zonotope.empty(2)
    Z_plus = Z1 + Z_empty  # Using + operator as in MATLAB
    assert Z_plus.representsa_('emptySet') 