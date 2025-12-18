"""
test_origin - unit test function for polyZonotope origin

Tests the origin polyZonotope instantiation functionality.

Authors: MATLAB: Mark Wetzlinger
         Python: AI Assistant
"""

import numpy as np
import pytest
from cora_python.contSet.polyZonotope.polyZonotope import PolyZonotope


class TestPolyZonotopeOrigin:
    """Test class for polyZonotope origin method"""
    
    def test_origin_1d(self):
        """Test origin instantiation for 1D polyZonotope"""
        pZ = PolyZonotope.origin(1)
        pZ_true = PolyZonotope(np.array([[0]]))
        assert pZ == pZ_true
        # Note: contains method would need to be implemented to fully test this
    
    def test_origin_2d(self):
        """Test origin instantiation for 2D polyZonotope"""
        pZ = PolyZonotope.origin(2)
        pZ_true = PolyZonotope(np.zeros((2, 1)))
        assert pZ == pZ_true
        # Note: contains method would need to be implemented to fully test this
    
    def test_origin_various_dimensions(self):
        """Test origin instantiation for various dimensions"""
        for n in [1, 2, 3, 5, 10]:
            pZ = PolyZonotope.origin(n)
            pZ_true = PolyZonotope(np.zeros((n, 1)))
            assert pZ == pZ_true
            assert pZ.dim() == n
    
    def test_origin_properties(self):
        """Test properties of origin polyZonotope"""
        pZ = PolyZonotope.origin(3)
        
        # Should have correct dimension
        assert pZ.dim() == 3
        
        # Should not be empty object
        assert not pZ.isemptyobject()
        
        # Should represent origin
        assert pZ.representsa_('origin', tol=1e-10)
    
    def test_origin_wrong_calls(self):
        """Test wrong calls to origin method"""
        # Zero dimension
        with pytest.raises(Exception):
            PolyZonotope.origin(0)
        
        # Negative dimension
        with pytest.raises(Exception):
            PolyZonotope.origin(-1)
        
        # Non-integer dimension
        with pytest.raises(Exception):
            PolyZonotope.origin(0.5)
        
        # Multiple values
        with pytest.raises(Exception):
            PolyZonotope.origin([1, 2])
        
        # String input
        with pytest.raises(Exception):
            PolyZonotope.origin('text')
    
    def test_origin_edge_cases(self):
        """Test edge cases for origin instantiation"""
        # Minimum valid dimension
        pZ = PolyZonotope.origin(1)
        assert pZ.dim() == 1
        assert pZ.representsa_('origin', tol=1e-10)
        
        # Higher dimension
        pZ = PolyZonotope.origin(20)
        assert pZ.dim() == 20
        assert pZ.representsa_('origin', tol=1e-10) 