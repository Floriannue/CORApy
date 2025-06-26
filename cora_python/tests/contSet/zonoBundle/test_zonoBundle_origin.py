"""
test_origin - unit test function for zonoBundle origin

Tests the origin bundle creation functionality for ZonoBundle objects.

Authors:       Matthias Althoff (MATLAB)
               Python translation by AI Assistant
Written:       02-August-2023
Last update:   ---
Last revision: ---
"""

import pytest
import numpy as np
from cora_python.contSet.zonoBundle.zonoBundle import ZonoBundle


class TestZonoBundleOrigin:
    """Test class for zonoBundle origin method"""

    def test_origin_basic(self):
        """Test basic origin creation"""
        n = 2
        origin_zB = ZonoBundle.origin(n)
        
        # Should create a valid ZonoBundle object
        assert isinstance(origin_zB, ZonoBundle)
        assert origin_zB.dim() == n

    def test_origin_various_dimensions(self):
        """Test origin creation for various dimensions"""
        for n in [1, 3, 5, 10]:
            origin_zB = ZonoBundle.origin(n)
            assert isinstance(origin_zB, ZonoBundle)
            assert origin_zB.dim() == n

    def test_origin_zero_dimension(self):
        """Test origin creation for zero dimension"""
        origin_zB = ZonoBundle.origin(0)
        assert isinstance(origin_zB, ZonoBundle)
        assert origin_zB.dim() == 0

    def test_origin_static_method(self):
        """Test that origin is a static method"""
        # Should be callable without instance
        origin_zB = ZonoBundle.origin(3)
        assert isinstance(origin_zB, ZonoBundle)

    def test_origin_invalid_dimension(self):
        """Test origin creation with invalid dimension"""
        # Negative dimension should raise error
        with pytest.raises((ValueError, Exception)):
            ZonoBundle.origin(-1)

    def test_origin_large_dimension(self):
        """Test origin creation for large dimension"""
        n = 100
        origin_zB = ZonoBundle.origin(n)
        assert isinstance(origin_zB, ZonoBundle)
        assert origin_zB.dim() == n

    def test_origin_consistency(self):
        """Test that origin creation is consistent"""
        n = 4
        origin1 = ZonoBundle.origin(n)
        origin2 = ZonoBundle.origin(n)
        
        # Both should be valid ZonoBundle objects with same dimension
        assert isinstance(origin1, ZonoBundle)
        assert isinstance(origin2, ZonoBundle)
        assert origin1.dim() == origin2.dim() == n

    def test_origin_not_empty(self):
        """Test that origin is not an empty object"""
        n = 3
        origin_zB = ZonoBundle.origin(n)
        assert not origin_zB.isemptyobject()

    def test_origin_properties(self):
        """Test properties of origin ZonoBundle"""
        n = 3
        origin_zB = ZonoBundle.origin(n)
        
        # Should have proper attributes
        assert hasattr(origin_zB, 'Z')
        assert hasattr(origin_zB, 'parallelSets')
        
        # Dimension should be correct
        assert origin_zB.dim() == n
        
        # Should have at least one zonotope
        assert origin_zB.parallelSets >= 1

    def test_origin_different_from_empty(self):
        """Test that origin is different from empty"""
        n = 2
        origin_zB = ZonoBundle.origin(n)
        empty_zB = ZonoBundle.empty(n)
        
        # Both should have same dimension but potentially different properties
        assert origin_zB.dim() == empty_zB.dim()
        # Origin should not be empty
        assert not origin_zB.isemptyobject() 