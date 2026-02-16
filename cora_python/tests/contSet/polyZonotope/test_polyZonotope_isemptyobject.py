"""
test_isemptyobject - unit test function for polyZonotope isemptyobject

Tests the empty object checking functionality for polyZonotopes.

Authors: MATLAB: Mark Wetzlinger
         Python: AI Assistant
"""

import numpy as np
import pytest
from cora_python.contSet.polyZonotope.polyZonotope import PolyZonotope


class TestPolyZonotopeIsEmptyObject:
    """Test class for polyZonotope isemptyobject method"""
    
    def test_isemptyobject_empty_polyzonotope(self):
        """Test isemptyobject with empty polyZonotope"""
        pZ = PolyZonotope.empty(2)
        assert pZ.isemptyobject()
    
    def test_isemptyobject_non_empty_polyzonotope(self):
        """Test isemptyobject with non-empty polyZonotope"""
        c = np.array([[0], [0]])
        G = np.array([[0, 4, 1, -1, 2], [1, 2, -1, -1, 1]])
        GI = np.array([[-7, 1, 1], [15, 1, -1]])
        E = np.array([[1, 0, 0, 0, 1], [0, 1, 0, 3, 2], [0, 0, 1, 1, 0]])
        pZ = PolyZonotope(c, G, GI, E)
        assert not pZ.isemptyobject()
    
    def test_isemptyobject_various_empty_dimensions(self):
        """Test isemptyobject with various empty dimensions"""
        for n in [1, 2, 3, 5, 10]:
            pZ = PolyZonotope.empty(n)
            assert pZ.isemptyobject()
    
    def test_isemptyobject_origin_polyzonotope(self):
        """Test isemptyobject with origin polyZonotope"""
        for n in [1, 2, 3, 5]:
            pZ = PolyZonotope.origin(n)
            assert not pZ.isemptyobject()
    
    def test_isemptyobject_random_polyzonotope(self):
        """Test isemptyobject with random polyZonotope"""
        pZ = PolyZonotope.generateRandom(dimension=3, nr_generators=2, nr_indep_generators=2)
        assert not pZ.isemptyobject()
    
    def test_isemptyobject_simple_cases(self):
        """Test isemptyobject with simple cases"""
        # Simple 1D polyZonotope
        pZ = PolyZonotope(np.array([[1]]))
        assert not pZ.isemptyobject()
        
        # Simple 2D polyZonotope with generators
        c = np.array([[0], [0]])
        G = np.array([[1, 0], [0, 1]])
        E = np.array([[1, 0], [0, 1]])
        pZ = PolyZonotope(c, G, None, E)
        assert not pZ.isemptyobject()
    
    def test_isemptyobject_edge_cases(self):
        """Test isemptyobject with edge cases"""
        # PolyZonotope with only center
        pZ = PolyZonotope(np.array([[2], [3]]))
        assert not pZ.isemptyobject()
        
        # PolyZonotope with center and independent generators only
        c = np.array([[1], [2]])
        GI = np.array([[1, 0], [0, 1]])
        pZ = PolyZonotope(c, None, GI, None)
        assert not pZ.isemptyobject()
    
    def test_isemptyobject_consistency(self):
        """Test consistency of isemptyobject"""
        # Empty polyZonotope should be consistent
        pZ_empty = PolyZonotope.empty(3)
        assert pZ_empty.isemptyobject()
        assert pZ_empty.representsa_('emptySet', tol=1e-10)
        
        # Non-empty polyZonotope should be consistent
        pZ_origin = PolyZonotope.origin(3)
        assert not pZ_origin.isemptyobject()
        assert not pZ_origin.representsa_('emptySet', tol=1e-10) 