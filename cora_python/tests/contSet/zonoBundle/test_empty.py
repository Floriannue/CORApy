"""
test_empty - unit test function for zonoBundle empty

Tests the empty bundle creation functionality for ZonoBundle objects.

Authors:       Matthias Althoff (MATLAB)
               Python translation by AI Assistant
Written:       02-August-2023
Last update:   ---
Last revision: ---
"""

import pytest
import numpy as np
from cora_python.contSet.zonoBundle.zonoBundle import ZonoBundle


class TestZonoBundleEmpty:
    """Test class for zonoBundle empty method"""

    def test_empty_basic(self):
        """Test basic empty creation"""
        n = 2
        empty_zB = ZonoBundle.empty(n)
        
        # Should create a valid ZonoBundle object
        assert isinstance(empty_zB, ZonoBundle)
        assert empty_zB.dim() == n

    def test_empty_various_dimensions(self):
        """Test empty creation for various dimensions"""
        for n in [1, 3, 5, 10]:
            empty_zB = ZonoBundle.empty(n)
            assert isinstance(empty_zB, ZonoBundle)
            assert empty_zB.dim() == n

    def test_empty_zero_dimension(self):
        """Test empty creation for zero dimension"""
        empty_zB = ZonoBundle.empty(0)
        assert isinstance(empty_zB, ZonoBundle)
        assert empty_zB.dim() == 0

    def test_empty_static_method(self):
        """Test that empty is a static method"""
        # Should be callable without instance
        empty_zB = ZonoBundle.empty(3)
        assert isinstance(empty_zB, ZonoBundle)

    def test_empty_invalid_dimension(self):
        """Test empty creation with invalid dimension"""
        # Negative dimension should raise error
        with pytest.raises((ValueError, Exception)):
            ZonoBundle.empty(-1)

    def test_empty_large_dimension(self):
        """Test empty creation for large dimension"""
        n = 100
        empty_zB = ZonoBundle.empty(n)
        assert isinstance(empty_zB, ZonoBundle)
        assert empty_zB.dim() == n

    def test_empty_consistency(self):
        """Test that empty creation is consistent"""
        n = 4
        empty1 = ZonoBundle.empty(n)
        empty2 = ZonoBundle.empty(n)
        
        # Both should be valid ZonoBundle objects with same dimension
        assert isinstance(empty1, ZonoBundle)
        assert isinstance(empty2, ZonoBundle)
        assert empty1.dim() == empty2.dim() == n

    def test_empty_properties(self):
        """Test properties of empty ZonoBundle"""
        n = 3
        empty_zB = ZonoBundle.empty(n)
        
        # Should have proper attributes
        assert hasattr(empty_zB, 'Z')
        assert hasattr(empty_zB, 'parallelSets')
        
        # Dimension should be correct
        assert empty_zB.dim() == n 