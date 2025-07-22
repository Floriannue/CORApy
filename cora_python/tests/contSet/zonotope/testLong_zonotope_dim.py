"""
testLong_zonotope_dim - long unit test function of dim

Performs randomized long tests for the dim method of zonotope objects.

Syntax:
    pytest cora_python/tests/contSet/zonotope/testLong_zonotope_dim.py

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Date:          2025-01-11
"""

import numpy as np
from cora_python.contSet.zonotope.zonotope import Zonotope

def test_long_zonotope_dim():
    nr_of_tests = 100
    for i in range(nr_of_tests):
        n = np.random.randint(2, 51)
        c = np.random.randn(n, 1)
        if np.random.rand() < 0.05:
            G = np.zeros((n, 0))
        else:
            G = 5 * np.random.randn(n, np.random.randint(1, 11))
        Z = Zonotope(c, G)
        Zdim = Z.dim()
        assert Zdim == n, f"Failed at iteration {i}: got {Zdim}, expected {n}" 