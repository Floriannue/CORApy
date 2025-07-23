"""
Long test for Zonotope.supportFunc_ mirroring MATLAB's testLong_zonotope_supportFunc.m

Authors: Victor Gassmann (MATLAB)
         Python translation by AI Assistant
Written: 11-October-2019 (MATLAB)
Python translation: 2025
"""

import numpy as np
import pytest
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.zonotope.minnorm import minnorm
from cora_python.contSet.zonotope.norm_ import norm_ as zonotope_norm

TOL = 1e-8

@pytest.mark.slow
@pytest.mark.parametrize("dim", [2, 3, 4, 5])
def test_long_supportFunc_randomized(dim):
    dGen = 5
    steps = 3
    np.random.seed(42 + dim)  # For reproducibility per dimension
    for j in range(1, steps + 1):
        m = dim + j * dGen
        # Randomly generate zonotope
        c = np.random.randn(dim, 1)
        G = np.random.randn(dim, m)
        Z = Zonotope(c, G)
        # Compute min and max support vectors
        _, x_min = minnorm(Z)
        _, x_max = zonotope_norm(Z, 2, mode='exact', return_vertex=True)
        c_vec = Z.c.reshape(-1, 1)
        r_min = np.linalg.norm(x_min - c_vec)
        r_max = np.linalg.norm(x_max - c_vec)
        # Centered zonotope
        Z0 = Zonotope(np.zeros((dim, 1)), G)
        # Generate 2*dim random directions
        L = np.random.randn(dim, 2 * dim)
        L = L / np.linalg.norm(L, axis=0, keepdims=True)
        for k in range(L.shape[1]):
            dir_k = L[:, k].reshape(-1, 1)
            val, _, _ = Z0.supportFunc_(dir_k, 'upper')
            assert val >= r_min - TOL, f"Support function {val} < r_min {r_min} (dim={dim}, j={j}, k={k})"
            assert val <= r_max + TOL, f"Support function {val} > r_max {r_max} (dim={dim}, j={j}, k={k})" 