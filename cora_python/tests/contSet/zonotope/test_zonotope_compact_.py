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
from cora_python.g.functions.matlab.validate.check.compareMatrices import compareMatrices


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

    def test_compact_no_zero_generators(self):
        """Test compact on zonotope without zero generators"""
        # Random test similar to MATLAB
        n = 3
        nrOfGens = 8
        
        # Center, generator matrix without any all-zero generators
        c = np.zeros((n, 1))
        G_nozeros = 1 + np.random.rand(n, nrOfGens)
        
        # Zonotope without all-zero generators
        Z_nozeros = Zonotope(c, G_nozeros)
        
        # Representation without zero-generators
        tol = 1e-12
        Z_compact = Z_nozeros.compact_('zeros', tol)
        
        # Since no zero generators, results has to be the same as before
        assert compareMatrices(Z_compact.c, c)
        assert compareMatrices(Z_compact.G, G_nozeros)

    def test_compact_with_zero_generators(self):
        """Test compact on zonotope with appended zero generators"""
        n = 3
        nrOfGens = 8
        
        # Center, generator matrix without any all-zero generators
        c = np.zeros((n, 1))
        G_nozeros = 1 + np.random.rand(n, nrOfGens)
        
        # Zonotope without all-zero generators
        Z_nozeros = Zonotope(c, G_nozeros)
        
        # Append zero generators
        zeros_cols = 5
        G_withzeros = np.hstack([G_nozeros, np.zeros((n, zeros_cols))])
        
        # Shuffle matrix
        perm = np.random.permutation(G_withzeros.shape[1])
        G_withzeros = G_withzeros[:, perm]
        Z_withzeros = Zonotope(c, G_withzeros)
        
        # Representation without zero-generators
        tol = 1e-12
        Z_compact = Z_withzeros.compact_('zeros', tol)
        
        # Result has to be the same as original zonotope
        # Instead of isequal, use direct comparison for now
        assert np.allclose(Z_compact.c, Z_nozeros.c, atol=tol)
        assert compareMatrices(Z_compact.G, Z_nozeros.G, tol)
        assert compareMatrices(Z_compact.c, c)
        assert compareMatrices(Z_compact.G, G_nozeros)

    def test_compact_tolerance(self):
        """Test compact with tolerance"""
        # Create zonotope with nearly zero generators
        tol = 1e-10
        Z = Zonotope(np.array([[1], [2]]), 
                    np.array([[1, tol/2, 2], 
                             [0, tol/2, 1]]))
        Z_compact = Z.compact_('zeros', tol)
        
        # Should remove the nearly zero generator
        G_true = np.array([[1, 2], [0, 1]])
        assert compareMatrices(Z_compact.G, G_true)

    def test_compact_empty_zonotope(self):
        """Test compact on empty zonotope"""
        Z_empty = Zonotope.empty(2)
        Z_compact = Z_empty.compact_('zeros')
        assert Z_compact.representsa_('emptySet')


if __name__ == "__main__":
    pytest.main([__file__]) 