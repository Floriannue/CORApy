"""
test_zonotope_dim - unit test function of dim

Tests the dim method for zonotope objects to check dimension calculation.

Syntax:
    pytest cora_python/tests/contSet/zonotope/test_zonotope_dim.py

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Date:          2025-01-11
"""

import pytest
import numpy as np
from cora_python.contSet.zonotope.zonotope import Zonotope

def test_dim_empty_zonotope():
    Z = Zonotope.empty(2)
    assert Z.dim() == 2
    Z = Zonotope(np.zeros((3, 0)))
    assert Z.dim() == 3

def test_dim_regular_zonotope():
    c = np.array([[-2], [1]])
    G = np.array([[2, 4, 5, 3, 3], [0, 3, 5, 2, 3]])
    Z = Zonotope(c, G)
    assert Z.dim() == 2

def test_dim_no_generator():
    c = np.array([[-2], [1]])
    Z = Zonotope(c)
    assert Z.dim() == 2 