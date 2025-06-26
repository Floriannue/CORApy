"""
test_dim - unit test function for polyZonotope dim

Tests the dimension determination functionality for polyZonotopes.

Authors: Python: AI Assistant
"""

import numpy as np
import pytest
from cora_python.contSet.polyZonotope.polyZonotope import PolyZonotope


class TestPolyZonotropeDim:
    """Test class for polyZonotope dim method"""
    
    def test_dim_empty_polyzonotope(self):
        """Test dim with empty polyZonotope"""
        for n in [1, 2, 3, 5, 10]:
            pZ = PolyZonotope.empty(n)
            assert pZ.dim() == n
    
    def test_dim_origin_polyzonotope(self):
        """Test dim with origin polyZonotope"""
        for n in [1, 2, 3, 5, 10]:
            pZ = PolyZonotope.origin(n)
            assert pZ.dim() == n
    
    def test_dim_random_polyzonotope(self):
        """Test dim with random polyZonotope"""
        for n in [1, 2, 3, 5, 10]:
            pZ = PolyZonotope.generateRandom(Dimension=n)
            assert pZ.dim() == n
    
    def test_dim_constructed_polyzonotope(self):
        """Test dim with manually constructed polyZonotope"""
        # 1D case
        pZ = PolyZonotope(np.array([[1]]))
        assert pZ.dim() == 1
        
        # 2D case
        c = np.array([[0], [0]])
        G = np.array([[1, 0], [0, 1]])
        E = np.array([[1, 0], [0, 1]])
        pZ = PolyZonotope(c, G, None, E)
        assert pZ.dim() == 2
        
        # 3D case with independent generators
        c = np.array([[1], [2], [3]])
        GI = np.array([[1, 0], [0, 1], [0, 0]])
        pZ = PolyZonotope(c, None, GI, None)
        assert pZ.dim() == 3
    
    def test_dim_complex_polyzonotope(self):
        """Test dim with complex polyZonotope"""
        c = np.array([[0], [0], [0]])
        G = np.array([[0, 4, 1, -1, 2], [1, 2, -1, -1, 1], [2, 1, 0, 1, -1]])
        GI = np.array([[-7, 1, 1], [15, 1, -1], [0, 0, 1]])
        E = np.array([[1, 0, 0, 0, 1], [0, 1, 0, 3, 2], [0, 0, 1, 1, 0]])
        pZ = PolyZonotope(c, G, GI, E)
        assert pZ.dim() == 3
    
    def test_dim_edge_cases(self):
        """Test dim with edge cases"""
        # Minimum dimension
        pZ = PolyZonotope.origin(1)
        assert pZ.dim() == 1
        
        # High dimension
        pZ = PolyZonotope.origin(50)
        assert pZ.dim() == 50
    
    def test_dim_consistency(self):
        """Test dim consistency across operations"""
        n = 4
        pZ_empty = PolyZonotope.empty(n)
        pZ_origin = PolyZonotope.origin(n)
        pZ_random = PolyZonotope.generateRandom(Dimension=n)
        
        assert pZ_empty.dim() == n
        assert pZ_origin.dim() == n
        assert pZ_random.dim() == n 