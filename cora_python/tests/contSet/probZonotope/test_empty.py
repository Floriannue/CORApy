"""
test_empty - unit test function for probZonotope empty instantiation

Tests the empty probZonotope instantiation functionality.

Authors: Python: AI Assistant
"""

import numpy as np
import pytest
from cora_python.contSet.probZonotope.probZonotope import ProbZonotope


class TestProbZonotopeEmpty:
    """Test class for probZonotope empty method"""
    
    def test_empty_1d(self):
        """Test empty instantiation for 1D probZonotope"""
        n = 1
        pZ = ProbZonotope.empty(n)
        assert pZ.dim() == 1
        assert pZ.isemptyobject()
    
    def test_empty_2d(self):
        """Test empty instantiation for 2D probZonotope"""
        n = 2
        pZ = ProbZonotope.empty(n)
        assert pZ.dim() == 2
        assert pZ.isemptyobject()
    
    def test_empty_various_dimensions(self):
        """Test empty instantiation for various dimensions"""
        for n in [1, 2, 3, 5, 10]:
            pZ = ProbZonotope.empty(n)
            assert pZ.dim() == n
            assert pZ.isemptyobject()
    
    def test_empty_properties(self):
        """Test properties of empty probZonotope"""
        pZ = ProbZonotope.empty(3)
        
        # Should be empty object
        assert pZ.isemptyobject()
        
        # Should have correct dimension
        assert pZ.dim() == 3
    
    def test_empty_edge_cases(self):
        """Test edge cases for empty instantiation"""
        # Test minimum dimension
        pZ = ProbZonotope.empty(1)
        assert pZ.dim() == 1
        assert pZ.isemptyobject()
        
        # Test higher dimension
        pZ = ProbZonotope.empty(20)
        assert pZ.dim() == 20
        assert pZ.isemptyobject() 