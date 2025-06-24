"""
test_empty - unit test function for spectraShadow empty

Tests the empty set creation functionality for SpectraShadow objects.

Authors:       Niklas Kochdumper (MATLAB)
               Python translation by AI Assistant
Written:       02-August-2023
Last update:   ---
Last revision: ---
"""

import pytest
import numpy as np
from cora_python.contSet.spectraShadow.spectraShadow import SpectraShadow


class TestSpectraShadowEmpty:
    """Test class for spectraShadow empty method"""

    def test_empty_basic(self):
        """Test basic empty creation"""
        n = 2
        empty_spec = SpectraShadow.empty(n)
        
        # Should create a valid SpectraShadow object
        assert isinstance(empty_spec, SpectraShadow)
        assert empty_spec.dim() == n

    def test_empty_various_dimensions(self):
        """Test empty creation for various dimensions"""
        for n in [1, 3, 5, 10]:
            empty_spec = SpectraShadow.empty(n)
            assert isinstance(empty_spec, SpectraShadow)
            assert empty_spec.dim() == n

    def test_empty_zero_dimension(self):
        """Test empty creation for zero dimension"""
        empty_spec = SpectraShadow.empty(0)
        assert isinstance(empty_spec, SpectraShadow)
        assert empty_spec.dim() == 0

    def test_empty_static_method(self):
        """Test that empty is a static method"""
        # Should be callable without instance
        empty_spec = SpectraShadow.empty(3)
        assert isinstance(empty_spec, SpectraShadow)

    def test_empty_invalid_dimension(self):
        """Test empty creation with invalid dimension"""
        # Negative dimension should raise error
        with pytest.raises((ValueError, Exception)):
            SpectraShadow.empty(-1)

    def test_empty_large_dimension(self):
        """Test empty creation for large dimension"""
        n = 50
        empty_spec = SpectraShadow.empty(n)
        assert isinstance(empty_spec, SpectraShadow)
        assert empty_spec.dim() == n

    def test_empty_consistency(self):
        """Test that empty creation is consistent"""
        n = 4
        empty1 = SpectraShadow.empty(n)
        empty2 = SpectraShadow.empty(n)
        
        # Both should be valid SpectraShadow objects with same dimension
        assert isinstance(empty1, SpectraShadow)
        assert isinstance(empty2, SpectraShadow)
        assert empty1.dim() == empty2.dim() == n

    def test_empty_properties(self):
        """Test properties of empty SpectraShadow"""
        n = 3
        empty_spec = SpectraShadow.empty(n)
        
        # Should have proper attributes
        assert hasattr(empty_spec, 'A')
        assert hasattr(empty_spec, 'c')
        assert hasattr(empty_spec, 'G')
        
        # Dimension should be correct
        assert empty_spec.dim() == n 