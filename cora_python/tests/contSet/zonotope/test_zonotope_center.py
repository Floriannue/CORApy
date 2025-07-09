"""
test_zonotope_center - unit test function of center

Tests the center method for zonotope objects.

Syntax:
    pytest cora_python/tests/contSet/zonotope/test_zonotope_center.py

Authors: Florian Lercher (translated from MATLAB)
Date: 2025-01-11
"""

import pytest
import numpy as np
from cora_python.contSet.zonotope.zonotope import Zonotope

def test_center_empty_zonotope():
    """Test center of empty zonotope (should return shape (2, 0) empty array)"""
    Z = Zonotope.empty(2)
    c = Z.center()
    assert isinstance(c, np.ndarray)
    assert c.shape == (2, 0)
    assert c.size == 0

def test_center_2d_zonotope():
    """Test center of 2D zonotope (should match input c, shape (2, 1))"""
    c = np.array([[1], [5]])
    G = np.array([[2, 3, 4], [6, 7, 8]])
    Z = Zonotope(c, G)
    center_result = Z.center()
    assert center_result.shape == (2, 1)
    assert np.allclose(center_result, c)