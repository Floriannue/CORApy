"""
test_isemptyobject - unit test function for fullspace isemptyobject

Tests the empty object check functionality for fullspace objects.

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       25-July-2023
Last update:   ---
Last revision: ---
"""

import pytest
import numpy as np
from cora_python.contSet.fullspace.fullspace import Fullspace


class TestFullspaceIsemptyobject:
    """Test class for fullspace isemptyobject method"""

    def test_basic_isemptyobject(self):
        """Test basic empty object check"""
        n = 2
        fs = Fullspace(n)
        
        # Fullspace should never be empty
        assert not fs.isemptyobject()

    def test_isemptyobject_different_dimensions(self):
        """Test empty object check for different dimensions"""
        for n in [1, 3, 5, 10]:
            fs = Fullspace(n)
            
            # Fullspace should never be empty regardless of dimension
            assert not fs.isemptyobject()

    def test_isemptyobject_zero_dimension(self):
        """Test empty object check for zero-dimensional fullspace"""
        n = 0
        fs = Fullspace(n)
        
        # Zero-dimensional fullspace is empty (matches MATLAB behavior)
        assert fs.isemptyobject()

    def test_isemptyobject_consistency(self):
        """Test that isemptyobject is consistent"""
        n = 3
        fs = Fullspace(n)
        
        # Multiple calls should return the same result
        result1 = fs.isemptyobject()
        result2 = fs.isemptyobject()
        result3 = fs.isemptyobject()
        
        assert result1 == result2 == result3 == False

    def test_isemptyobject_return_type(self):
        """Test that isemptyobject returns boolean"""
        n = 2
        fs = Fullspace(n)
        
        result = fs.isemptyobject()
        
        # Should return boolean
        assert isinstance(result, bool)
        assert result is False 