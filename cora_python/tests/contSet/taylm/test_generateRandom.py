"""
test_generateRandom - unit test function for taylm generateRandom

Tests the random generation functionality for Taylm objects.

Authors:       Tobias Ladner (MATLAB)
               Python translation by AI Assistant
Written:       02-August-2023
Last update:   ---
Last revision: ---
"""

import pytest
import numpy as np
from cora_python.contSet.taylm.taylm import Taylm


class TestTaylmGenerateRandom:
    """Test class for taylm generateRandom method"""

    def test_generateRandom_without_arguments(self):
        """Test generateRandom without input arguments"""
        # Without input arguments
        tay = Taylm.generateRandom()
        
        # Should create a valid Taylm object
        assert isinstance(tay, Taylm)
        assert tay.dim() > 0  # Should have positive dimension
        assert not tay.isemptyobject()

    def test_generateRandom_with_dimension(self):
        """Test generateRandom with dimension specified"""
        # Dimension given
        n = 2
        tay = Taylm.generateRandom(Dimension=n)
        assert tay.dim() == n

    def test_generateRandom_various_dimensions(self):
        """Test generateRandom with various dimensions"""
        for n in [1, 3, 5, 10]:
            tay = Taylm.generateRandom(Dimension=n)
            assert tay.dim() == n
            assert isinstance(tay, Taylm)
            assert not tay.isemptyobject()

    def test_generateRandom_large_dimension(self):
        """Test generateRandom with large dimension"""
        n = 50
        tay = Taylm.generateRandom(Dimension=n)
        assert tay.dim() == n
        assert isinstance(tay, Taylm)

    def test_generateRandom_consistency(self):
        """Test that generateRandom produces valid objects consistently"""
        # Generate multiple objects
        for _ in range(10):
            tay = Taylm.generateRandom(Dimension=3)
            assert isinstance(tay, Taylm)
            assert tay.dim() == 3
            assert not tay.isemptyobject()

    def test_generateRandom_static_method(self):
        """Test that generateRandom is a static method"""
        # Should be callable without instance
        tay = Taylm.generateRandom()
        assert isinstance(tay, Taylm)

    def test_generateRandom_invalid_dimension(self):
        """Test generateRandom with invalid dimension"""
        # Negative dimension should raise error
        with pytest.raises((ValueError, Exception)):
            Taylm.generateRandom(Dimension=-1)
        
        # Zero dimension should raise error or be handled gracefully
        with pytest.raises((ValueError, Exception)):
            Taylm.generateRandom(Dimension=0)

    def test_generateRandom_default_dimension(self):
        """Test generateRandom default behavior"""
        tay1 = Taylm.generateRandom()
        tay2 = Taylm.generateRandom()
        
        # Both should be valid Taylm objects
        assert isinstance(tay1, Taylm)
        assert isinstance(tay2, Taylm)
        
        # Should have consistent default dimension behavior
        dim1 = tay1.dim()
        dim2 = tay2.dim()
        assert dim1 > 0
        assert dim2 > 0 