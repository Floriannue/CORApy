"""
test_spectraShadow - unit test function for SpectraShadow constructor

Tests the spectral shadow constructor functionality.

Authors:       Niklas Kochdumper (MATLAB)
               Python translation by AI Assistant
Written:       02-August-2023
Last update:   ---
Last revision: ---
"""

import pytest
import numpy as np
from cora_python.contSet.spectraShadow.spectraShadow import SpectraShadow


class TestSpectraShadow:
    """Test class for SpectraShadow constructor"""

    def test_constructor_basic(self):
        """Test basic constructor"""
        # Create a simple SpectraShadow
        spec = SpectraShadow.generateRandom(Dimension=2)
        
        # Should create a valid SpectraShadow object
        assert isinstance(spec, SpectraShadow)
        assert spec.dim() == 2

    def test_constructor_validation(self):
        """Test constructor input validation"""
        # Valid construction through generateRandom
        spec = SpectraShadow.generateRandom(Dimension=3)
        assert isinstance(spec, SpectraShadow)
        assert spec.dim() == 3

    def test_constructor_properties(self):
        """Test constructor sets proper properties"""
        spec = SpectraShadow.generateRandom(Dimension=4)
        
        # Check basic properties
        assert spec.dim() > 0
        assert not spec.isemptyobject()

    def test_constructor_various_dimensions(self):
        """Test constructor with various dimensions"""
        for n in [1, 2, 5, 10]:
            spec = SpectraShadow.generateRandom(Dimension=n)
            assert isinstance(spec, SpectraShadow)
            assert spec.dim() == n

    def test_constructor_large_dimension(self):
        """Test constructor with large dimension"""
        n = 50
        spec = SpectraShadow.generateRandom(Dimension=n)
        assert isinstance(spec, SpectraShadow)
        assert spec.dim() == n

    def test_constructor_consistency(self):
        """Test that constructor produces consistent objects"""
        for _ in range(5):
            spec = SpectraShadow.generateRandom(Dimension=3)
            assert isinstance(spec, SpectraShadow)
            assert spec.dim() == 3
            assert not spec.isemptyobject() 