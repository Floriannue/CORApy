import pytest
import numpy as np

from cora_python.contSet.zonotope.zonotope import Zonotope
from cora_python.g.functions.matlab.validate.check import compareMatrices

def test_contSet_projVertices():
    # Test 1: 2D zonotope
    c = np.array([0, 0])
    G = np.array([[1.5, -1.5, 0.5], [1, 0.5, -1]])
    Z = Zonotope(c, G)

    V = Z.vertices()
    V_proj = Z.projVertices()
    
    # For a 2D set, projVertices should return the same as vertices
    assert compareMatrices(V, V_proj, 1e-12)

    # Test 2: 3D zonotope
    c = np.array([0, 0, 1])
    G = np.array([[1.5, -1.5, 0.5], [1, 0.5, -1], [0, -0.5, -1]])
    Z = Zonotope(c, G)
    
    V = Z.vertices()
    dims_list = [[1, 2], [2, 3], [1, 3]]
    
    for dims in dims_list:
        V_proj = Z.projVertices(dims)
        dims_0_based = [d - 1 for d in dims]
        
        # The projected vertices must be a subset of the original vertices
        # projected onto the same dimensions.
        assert compareMatrices(V_proj, V[dims_0_based, :], 1e-12, 'subset')

    # Test 3: Degenerate zonotope (line)
    Z = Zonotope(np.array([1, 1]), np.array([[1], [-1]]))
    
    V_true = np.array([[2, 0], [0, 2]])
    V_proj = Z.projVertices()

    assert compareMatrices(V_true, V_proj, 1e-12)

    # Test 4: Degenerate zonotope (point)
    Z = Zonotope(np.array([1, 1]), np.empty((2, 0))) # no generators
    
    V_true = np.array([[1], [1]])
    V_proj = Z.projVertices()
    
    assert compareMatrices(V_true, V_proj, 1e-12) 