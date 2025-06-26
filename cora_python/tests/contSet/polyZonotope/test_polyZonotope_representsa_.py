"""
test_representsa_ - unit test function for polyZonotope representsa_

Tests the representation checking functionality for polyZonotopes.

Authors: MATLAB: Mark Wetzlinger
         Python: AI Assistant
"""

import numpy as np
import pytest
from cora_python.contSet.polyZonotope.polyZonotope import PolyZonotope


class TestPolyZonotopeRepresentsa:
    """Test class for polyZonotope representsa_ method"""
    
    def test_representsa_emptyset(self):
        """Test representsa_ with emptySet comparison"""
        # Non-empty polyZonotopes
        c = np.array([[2], [-3]])
        G = np.array([[2, 3], [-1, 0]])
        GI = np.array([[1, 2, 4], [5, 6, 0]])
        E = np.array([[2, 0], [0, 1]])
        
        pZ1 = PolyZonotope(c, G, GI, E)
        assert not pZ1.representsa_('emptySet')
        
        pZ2 = PolyZonotope(c, G, None, E)
        assert not pZ2.representsa_('emptySet')
        
        # Empty polyZonotope
        pZ3 = PolyZonotope.empty(2)
        assert pZ3.representsa_('emptySet')
    
    def test_representsa_origin(self):
        """Test representsa_ with origin comparison"""
        # Empty polyZonotope should not represent origin
        pZ = PolyZonotope.empty(2)
        assert not pZ.representsa_('origin')
        
        # Only origin
        pZ = PolyZonotope(np.zeros((3, 1)))
        assert pZ.representsa_('origin')
        
        # Shifted center
        pZ = PolyZonotope(0.01 * np.ones((4, 1)))
        assert not pZ.representsa_('origin')
        
        # With tolerance
        tol = 0.02
        assert pZ.representsa_('origin', tol)
        
        # Include dependent generator matrix
        pZ = PolyZonotope(np.ones((2, 1)), 0.1 * np.eye(2))
        tol = 2
        assert pZ.representsa_('origin', tol)
        
        # Dependent and independent generators
        c = np.zeros((2, 1))
        G = np.array([[0.01, -0.02], [0.03, 0.01]])
        GI = np.array([[0.05], [-0.02]])
        E = np.array([[2, 0], [0, 1]])
        pZ = PolyZonotope(c, G, GI, E)
        tol = 0.1
        assert pZ.representsa_('origin', tol)
    
    def test_representsa_zonotope(self):
        """Test representsa_ with zonotope comparison"""
        c = np.array([[2], [-1]])
        G = np.array([[1], [-1]])
        GI = np.array([[2, 0, 1], [-1, 1, 0]])
        
        # Only independent generators
        pZ = PolyZonotope(c, None, GI)
        result, Z = pZ.representsa_('zonotope')
        assert result
        # Expected: zonotope with center c and generators GI
        
        # Dependent and independent generators
        pZ = PolyZonotope(c, G, GI)
        result, Z = pZ.representsa_('zonotope')
        assert result
        # Expected: zonotope with center c and generators [G, GI]
        
        # With exponent matrix
        E = np.array([[3], [0]])
        pZ = PolyZonotope(c, G, GI, E)
        result, Z = pZ.representsa_('zonotope')
        assert result
        # Expected: zonotope with center c and generators [G, GI]
    
    def test_representsa_point(self):
        """Test representsa_ with point comparison"""
        c = np.array([[2], [-1]])
        G = np.array([[1], [-1]])
        G_zeros = np.zeros((2, 2))
        GI = np.array([[2, 0, 1], [-1, 1, 0]])
        GI_zeros = np.zeros((2, 3))
        
        # Only center
        pZ = PolyZonotope(c)
        assert pZ.representsa_('point')
        
        # With dependent generators
        pZ = PolyZonotope(c, G)
        assert not pZ.representsa_('point')
        
        # With zero dependent generators
        pZ = PolyZonotope(c, G_zeros)
        assert pZ.representsa_('point')
        
        # With independent generators
        pZ = PolyZonotope(c, None, GI)
        assert not pZ.representsa_('point')
        
        # With zero independent generators
        pZ = PolyZonotope(c, None, GI_zeros)
        assert pZ.representsa_('point')
    
    def test_representsa_fullspace(self):
        """Test representsa_ with fullspace comparison"""
        # Most polyZonotopes should not represent fullspace
        pZ = PolyZonotope.origin(2)
        assert not pZ.representsa_('fullspace')
        
        pZ = PolyZonotope.generateRandom(Dimension=3)
        assert not pZ.representsa_('fullspace')
    
    def test_representsa_edge_cases(self):
        """Test representsa_ edge cases"""
        # Test with various dimensions
        for n in [1, 2, 3, 5]:
            pZ_empty = PolyZonotope.empty(n)
            assert pZ_empty.representsa_('emptySet')
            assert not pZ_empty.representsa_('origin')
            
            pZ_origin = PolyZonotope.origin(n)
            assert not pZ_origin.representsa_('emptySet')
            assert pZ_origin.representsa_('origin')
            assert pZ_origin.representsa_('point')
    
    def test_representsa_tolerance_parameter(self):
        """Test representsa_ with tolerance parameter"""
        # Small deviation from origin
        c = np.array([[0.005], [0.003]])
        pZ = PolyZonotope(c)
        
        assert not pZ.representsa_('origin')  # Without tolerance
        assert pZ.representsa_('origin', 0.01)  # With sufficient tolerance
        assert not pZ.representsa_('origin', 0.001)  # With insufficient tolerance
    
    def test_representsa_invalid_type(self):
        """Test representsa_ with invalid type"""
        pZ = PolyZonotope.origin(2)
        
        with pytest.raises(Exception):
            pZ.representsa_('invalid_type')
    
    def test_representsa_consistency(self):
        """Test representsa_ consistency"""
        # Empty polyZonotope consistency
        pZ_empty = PolyZonotope.empty(3)
        assert pZ_empty.representsa_('emptySet')
        assert pZ_empty.isemptyobject()
        
        # Origin polyZonotope consistency
        pZ_origin = PolyZonotope.origin(3)
        assert pZ_origin.representsa_('origin')
        assert pZ_origin.representsa_('point')
        assert not pZ_origin.representsa_('emptySet')
        
        # Random polyZonotope should not be empty
        pZ_random = PolyZonotope.generateRandom(Dimension=3)
        assert not pZ_random.representsa_('emptySet') 