"""
test_empty - unit test function for polyZonotope empty instantiation

Tests the empty polyZonotope instantiation functionality.

Authors: MATLAB: Tobias Ladner
         Python: AI Assistant
"""

import numpy as np
import pytest
from cora_python.contSet.polyZonotope.polyZonotope import PolyZonotope


class TestPolyZonotopeEmpty:
    """Test class for polyZonotope empty method"""
    
    def test_empty_1d(self):
        """Test empty instantiation for 1D polyZonotope"""
        n = 1
        pZ = PolyZonotope.empty(n)
        assert pZ.representsa_('emptySet') and pZ.dim() == 1
    
    def test_empty_5d(self):
        """Test empty instantiation for 5D polyZonotope"""
        n = 5
        pZ = PolyZonotope.empty(n)
        assert pZ.representsa_('emptySet') and pZ.dim() == 5
    
    def test_empty_various_dimensions(self):
        """Test empty instantiation for various dimensions"""
        for n in [2, 3, 4, 10]:
            pZ = PolyZonotope.empty(n)
            assert pZ.representsa_('emptySet') and pZ.dim() == n
    
    def test_empty_properties(self):
        """Test properties of empty polyZonotope"""
        pZ = PolyZonotope.empty(3)
        
        # Should be empty object
        assert pZ.isemptyobject()
        
        # Should have correct dimension
        assert pZ.dim() == 3
        
        # Check that it represents empty set
        assert pZ.representsa_('emptySet')
    
    def test_empty_edge_cases(self):
        """Test edge cases for empty instantiation"""
        # Test minimum dimension
        pZ = PolyZonotope.empty(1)
        assert pZ.dim() == 1
        assert pZ.representsa_('emptySet')
        
        # Test higher dimension
        pZ = PolyZonotope.empty(20)
        assert pZ.dim() == 20
        assert pZ.representsa_('emptySet') 