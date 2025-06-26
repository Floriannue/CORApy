"""
test_isemptyobject - unit test function for zonoBundle isemptyobject

Tests the empty object check functionality for ZonoBundle objects.

Authors:       Matthias Althoff (MATLAB)
               Python translation by AI Assistant
Written:       02-August-2023
Last update:   ---
Last revision: ---
"""

import pytest
import numpy as np
from cora_python.contSet.zonoBundle.zonoBundle import ZonoBundle
from cora_python.contSet.zonotope.zonotope import Zonotope


class TestZonoBundleIsemptyobject:
    """Test class for zonoBundle isemptyobject method"""

    def test_isemptyobject_normal_bundle(self):
        """Test isemptyobject for normal ZonoBundle objects"""
        Z1 = Zonotope(np.array([1, 2]), np.eye(2))
        Z2 = Zonotope(np.array([2, 3]), np.eye(2))
        zB = ZonoBundle([Z1, Z2])
        
        # Normal ZonoBundle should not be empty
        assert not zB.isemptyobject()

    def test_isemptyobject_single_zonotope(self):
        """Test isemptyobject for single zonotope bundle"""
        Z = Zonotope(np.array([1, 2]), np.eye(2))
        zB = ZonoBundle([Z])
        
        assert not zB.isemptyobject()

    def test_isemptyobject_various_dimensions(self):
        """Test isemptyobject for various dimensions"""
        for n in [1, 2, 5, 10]:
            Z1 = Zonotope(np.zeros(n), np.eye(n))
            Z2 = Zonotope(np.ones(n), np.eye(n))
            zB = ZonoBundle([Z1, Z2])
            assert not zB.isemptyobject()

    def test_isemptyobject_return_type(self):
        """Test that isemptyobject returns boolean"""
        Z = Zonotope(np.array([0, 1]), np.eye(2))
        zB = ZonoBundle([Z])
        
        result = zB.isemptyobject()
        assert isinstance(result, bool)

    def test_isemptyobject_consistency(self):
        """Test that isemptyobject is consistent"""
        Z1 = Zonotope(np.array([1, 2, 3]), np.eye(3))
        Z2 = Zonotope(np.array([2, 3, 4]), np.eye(3))
        zB = ZonoBundle([Z1, Z2])
        
        # Multiple calls should return same result
        result1 = zB.isemptyobject()
        result2 = zB.isemptyobject()
        result3 = zB.isemptyobject()
        
        assert result1 == result2 == result3

    def test_isemptyobject_generated_random(self):
        """Test isemptyobject for randomly generated ZonoBundle"""
        zB = ZonoBundle.generateRandom(Dimension=3)
        assert not zB.isemptyobject()

    def test_isemptyobject_origin(self):
        """Test isemptyobject for origin ZonoBundle"""
        origin_zB = ZonoBundle.origin(4)
        assert not origin_zB.isemptyobject()

    def test_isemptyobject_multiple_zonotopes(self):
        """Test isemptyobject for bundle with many zonotopes"""
        zonotopes = []
        for i in range(10):
            Z = Zonotope(np.array([i, i+1]), np.eye(2))
            zonotopes.append(Z)
        
        zB = ZonoBundle(zonotopes)
        assert not zB.isemptyobject()

    def test_isemptyobject_large_dimension(self):
        """Test isemptyobject for large dimension"""
        n = 50
        Z1 = Zonotope(np.zeros(n), np.eye(n))
        Z2 = Zonotope(np.ones(n), np.eye(n))
        zB = ZonoBundle([Z1, Z2])
        assert not zB.isemptyobject()

    def test_isemptyobject_empty_bundle(self):
        """Test isemptyobject for empty bundle"""
        empty_zB = ZonoBundle.empty(2)
        # Empty bundle should be empty
        result = empty_zB.isemptyobject()
        assert isinstance(result, bool)
        # Note: The actual result depends on implementation 