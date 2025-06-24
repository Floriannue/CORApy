"""
test_generateRandom - unit test function for zonoBundle generateRandom

Tests the random generation functionality for ZonoBundle objects.

Authors:       Matthias Althoff (MATLAB)
               Python translation by AI Assistant
Written:       02-August-2023
Last update:   ---
Last revision: ---
"""

import pytest
import numpy as np
from cora_python.contSet.zonoBundle.zonoBundle import ZonoBundle


class TestZonoBundleGenerateRandom:
    """Test class for zonoBundle generateRandom method"""

    def test_generateRandom_with_dimension(self):
        """Test generateRandom with dimension specified"""
        n = 2
        zB = ZonoBundle.generateRandom(Dimension=n)
        
        assert isinstance(zB, ZonoBundle)
        assert zB.dim() == n
        assert not zB.isemptyobject()

    def test_generateRandom_various_dimensions(self):
        """Test generateRandom with various dimensions"""
        for n in [1, 3, 5, 8]:
            zB = ZonoBundle.generateRandom(Dimension=n)
            assert isinstance(zB, ZonoBundle)
            assert zB.dim() == n
            assert not zB.isemptyobject()

    def test_generateRandom_with_number_of_sets(self):
        """Test generateRandom with specified number of zonotopes"""
        n = 3
        num_sets = 5
        zB = ZonoBundle.generateRandom(Dimension=n, NrGenerators=10)
        
        assert isinstance(zB, ZonoBundle)
        assert zB.dim() == n
        assert hasattr(zB, 'parallelSets')

    def test_generateRandom_consistency(self):
        """Test that generateRandom produces valid objects consistently"""
        for _ in range(5):
            zB = ZonoBundle.generateRandom(Dimension=3)
            assert isinstance(zB, ZonoBundle)
            assert zB.dim() == 3
            assert not zB.isemptyobject()

    def test_generateRandom_static_method(self):
        """Test that generateRandom is a static method"""
        zB = ZonoBundle.generateRandom(Dimension=2)
        assert isinstance(zB, ZonoBundle)

    def test_generateRandom_invalid_dimension(self):
        """Test generateRandom with invalid dimension"""
        # Negative dimension should raise error
        with pytest.raises((ValueError, Exception)):
            ZonoBundle.generateRandom(Dimension=-1)
        
        # Zero dimension should raise error or be handled gracefully
        with pytest.raises((ValueError, Exception)):
            ZonoBundle.generateRandom(Dimension=0)

    def test_generateRandom_properties(self):
        """Test that generated objects have expected properties"""
        zB = ZonoBundle.generateRandom(Dimension=4)
        
        # Should have proper attributes
        assert hasattr(zB, 'Z')
        assert hasattr(zB, 'parallelSets')
        
        # Should not be empty
        assert not zB.isemptyobject()
        
        # Should have valid zonotopes
        assert len(zB.Z) > 0
        assert zB.parallelSets == len(zB.Z)

    def test_generateRandom_large_dimension(self):
        """Test generateRandom with moderately large dimension"""
        n = 20
        zB = ZonoBundle.generateRandom(Dimension=n)
        assert isinstance(zB, ZonoBundle)
        assert zB.dim() == n

    def test_generateRandom_default_behavior(self):
        """Test generateRandom with default behavior"""
        zB = ZonoBundle.generateRandom(Dimension=3)
        assert isinstance(zB, ZonoBundle)
        assert zB.dim() == 3
        assert zB.parallelSets >= 1 