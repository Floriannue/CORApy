"""
test_dim - unit test function for SpectraShadow dim

Tests the dimension functionality for SpectraShadow objects.

Authors:       Niklas Kochdumper (MATLAB)
               Python translation by AI Assistant
Written:       02-August-2023
Last update:   ---
Last revision: ---
"""

import pytest
import numpy as np
from cora_python.contSet.spectraShadow.spectraShadow import SpectraShadow


class TestSpectraShadowDim:
    """Test class for SpectraShadow dim method"""

    def test_dim_basic(self):
        """Test basic dimension computation"""
        spec = SpectraShadow.generateRandom(Dimension=3)
        assert spec.dim() == 3

    def test_dim_various_dimensions(self):
        """Test dimension for various sizes"""
        for n in [1, 2, 4, 5, 10]:
            spec = SpectraShadow.generateRandom(Dimension=n)
            assert spec.dim() == n

    def test_dim_return_type(self):
        """Test that dim returns integer"""
        spec = SpectraShadow.generateRandom(Dimension=2)
        
        result = spec.dim()
        assert isinstance(result, int)
        assert result == 2

    def test_dim_consistency(self):
        """Test that dim is consistent across calls"""
        spec = SpectraShadow.generateRandom(Dimension=5)
        
        # Multiple calls should return same result
        dim1 = spec.dim()
        dim2 = spec.dim()
        dim3 = spec.dim()
        
        assert dim1 == dim2 == dim3 == 5

    def test_dim_large_dimension(self):
        """Test dimension for large dimensions"""
        n = 20  # Reasonable size for spectrahedral shadow
        spec = SpectraShadow.generateRandom(Dimension=n)
        assert spec.dim() == n

    def test_dim_small_dimension(self):
        """Test dimension for smallest valid dimension"""
        n = 1
        spec = SpectraShadow.generateRandom(Dimension=n)
        assert spec.dim() == n 