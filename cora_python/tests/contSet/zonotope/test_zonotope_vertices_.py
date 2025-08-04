"""
Test file for zonotope vertices_ method - translated from MATLAB

Authors: Matthias Althoff, Tobias Ladner (MATLAB)
         Python translation by AI Assistant
Written: 26-July-2016 (MATLAB)
Last update: 01-June-2023 (MATLAB)
               2025 (Tiange Yang, Florian Nüssel, Python translation by AI Assistant)
"""

import pytest
import numpy as np
from cora_python.contSet.zonotope.zonotope import Zonotope
from cora_python.contSet.zonotope.vertices_ import vertices_
from cora_python.g.functions.matlab.validate.check import compareMatrices


def test_zonotope_vertices():
    """Test zonotope vertices_ method - translated from MATLAB"""
    
    # empty set
    Z = Zonotope.empty(2)
    V = vertices_(Z)
    assert V.size == 0 and V.shape == (2, 0)
    
    # simple zonotope
    c = np.array([0, 0])
    G = np.array([[1, 0, 1], [1, 1, 0]])
    Z = Zonotope(c, G)
    V = vertices_(Z)
    V0 = np.array([[2, 0, -2, -2, 0, 2], 
                   [2, 2, 0, -2, -2, 0]])
    assert compareMatrices(V, V0)
    
    # zonotope
    c = np.array([-4, 1])
    G = np.array([[-3, -2, -1], [2, 3, 4]])
    Z = Zonotope(c, G)
    V = vertices_(Z)
    V0 = np.array([[-8, 0, 2, -4, -4, -10], 
                   [2, 0, -8, -4, 6, 10]])
    assert compareMatrices(V, V0)
    
    # 1d zonotope
    c = np.array([2])
    G = np.array([[1]])
    Z = Zonotope(c, G)
    V = vertices_(Z)
    V0 = np.array([[1, 3]])
    assert compareMatrices(V, V0)
    
    # degenerate case
    c = np.array([1, 2, 3])
    G = np.array([[1, 1], [0, 1], [1, 1]])
    Z = Zonotope(c, G)
    V = vertices_(Z)
    V0 = np.array([[1.000, 3.000, 1.000, -1.000], 
                   [3.000, 3.000, 1.000, 1.000], 
                   [3.000, 5.000, 3.000, 1.000]])
    assert compareMatrices(V, V0)
    
    # another degenerate case
    c = np.array([1, 2, 3])
    G = np.array([[1, 1, 2, -1], [0, 1, 2, 0], [1, 1, 2, -1]])
    Z = Zonotope(c, G)
    V = vertices_(Z)
    V0 = np.array([[2.000, -4.000, 6.000, 0.000], 
                   [5.000, -1.000, 5.000, -1.000], 
                   [4.000, -2.000, 8.000, 2.000]])
    assert compareMatrices(V, V0)
    
    # case with zero dimension
    c = np.array([1, 2, 0])
    G = np.array([[1, 1], [0, 1], [0, 0]])
    Z = Zonotope(c, G)
    V = vertices_(Z)
    V0 = np.array([[3.000, 1.000, 1.000, -1.000], 
                   [3.000, 3.000, 1.000, 1.000], 
                   [0.000, 0.000, 0.000, 0.000]])
    assert compareMatrices(V, V0)


if __name__ == '__main__':
    test_zonotope_vertices()
    print("All tests passed!") 