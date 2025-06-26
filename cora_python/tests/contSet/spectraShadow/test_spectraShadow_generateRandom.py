"""
test_generateRandom - unit test function for SpectraShadow generateRandom

Tests the random generation functionality for SpectraShadow objects.

Authors:       Niklas Kochdumper (MATLAB)
               Python translation by AI Assistant
Written:       02-August-2023
Last update:   ---
Last revision: ---
"""

import pytest
import numpy as np
from cora_python.contSet.spectraShadow.spectraShadow import SpectraShadow


class TestSpectraShadowGenerateRandom:
    """Test class for SpectraShadow generateRandom method"""

    def test_generateRandom_with_dimension(self):
        """Test generateRandom with dimension specified"""
        n = 2
        spec = SpectraShadow.generateRandom(Dimension=n)
        
        assert isinstance(spec, SpectraShadow)
        assert spec.dim() == n
        assert not spec.isemptyobject()

    def test_generateRandom_various_dimensions(self):
        """Test generateRandom with various dimensions"""
        for n in [1, 3, 5, 8]:
            spec = SpectraShadow.generateRandom(Dimension=n)
            assert isinstance(spec, SpectraShadow)
            assert spec.dim() == n
            assert not spec.isemptyobject()

    def test_generateRandom_consistency(self):
        """Test that generateRandom produces valid objects consistently"""
        for _ in range(5):
            spec = SpectraShadow.generateRandom(Dimension=3)
            assert isinstance(spec, SpectraShadow)
            assert spec.dim() == 3
            assert not spec.isemptyobject()

    def test_generateRandom_static_method(self):
        """Test that generateRandom is a static method"""
        spec = SpectraShadow.generateRandom(Dimension=2)
        assert isinstance(spec, SpectraShadow)

    def test_generateRandom_invalid_dimension(self):
        """Test generateRandom with invalid dimension"""
        # Negative dimension should raise error
        with pytest.raises((ValueError, Exception)):
            SpectraShadow.generateRandom(Dimension=-1)
        
        # Zero dimension should raise error or be handled gracefully
        with pytest.raises((ValueError, Exception)):
            SpectraShadow.generateRandom(Dimension=0)

    def test_generateRandom_large_dimension(self):
        """Test generateRandom with moderately large dimension"""
        n = 15  # Reasonable for spectrahedral shadows
        spec = SpectraShadow.generateRandom(Dimension=n)
        assert isinstance(spec, SpectraShadow)
        assert spec.dim() == n

    def test_generateRandom_properties(self):
        """Test that generated objects have expected properties"""
        spec = SpectraShadow.generateRandom(Dimension=4)
        
        # Should have proper attributes
        assert hasattr(spec, 'A')
        assert hasattr(spec, 'c')
        assert hasattr(spec, 'G')
        
        # Should not be empty
        assert not spec.isemptyobject()

    def test_generateRandom_default_behavior(self):
        """Test generateRandom with default behavior"""
        # Should work with just dimension specified
        spec = SpectraShadow.generateRandom(Dimension=3)
        assert isinstance(spec, SpectraShadow)
        assert spec.dim() == 3 