"""
test_zonotope_compact_ - unit test function of compact_

Tests the compact_ method for zonotope objects to check:
- Zero generator removal
- Generator alignment combination

Syntax:
    pytest cora_python/tests/contSet/zonotope/test_zonotope_compact_.py

Authors: Florian Lercher (translated from MATLAB)
Date: 2025-01-11
"""

import pytest
import numpy as np
from cora_python.contSet.zonotope.zonotope import Zonotope
from cora_python.g.functions.matlab.validate.check import compareMatrices


class TestZonotopeCompact:
    def test_compact_zeros_removal(self):
        """Test zero generator removal"""
        # 1D, zero removal
        Z = Zonotope(np.array([[4]]), np.array([[2, -3, 1, 0, 2]]))
        Z_compact = Z.compact_('zeros')
        G_true = np.array([[2, -3, 1, 2]])
        assert compareMatrices(Z_compact.G, G_true)
        
        # 2D, zero removal
        Z = Zonotope(np.array([[1], [5]]), np.array([[2, 0, 4], [6, 0, 0]]))
        Z_compact = Z.compact_('zeros')
        G_true = np.array([[2, 4], [6, 0]])
        assert compareMatrices(Z_compact.G, G_true)

    def test_compact_all_removal(self):
        """Test all generator combination"""
        # 1D, all
        Z = Zonotope(np.array([[4]]), np.array([[2, -3, 1, 0, 2]]))
        Z_compact = Z.compact_('all')
        G_true = np.array([[8]])
        assert compareMatrices(Z_compact.G, G_true)

    def test_compact_aligned_generators(self):
        """Test aligned generator combination"""
        # 2D, aligned generators differ by scaling, sign
        Z = Zonotope(np.zeros((2, 1)), 
                    np.array([[4, 2, 2, 3, 1, -4], 
                             [2, 3, 1, 0, 2, -2]]))
        Z_compact = Z.compact_('all')
        G_true = np.array([[10, 2, 3, 1], 
                          [5, 3, 0, 2]])
        assert compareMatrices(Z_compact.G, G_true)


if __name__ == "__main__":
    pytest.main([__file__]) 