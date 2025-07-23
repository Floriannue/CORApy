"""
Test supportFunc_ method for zonotope class
Mirrors MATLAB test_zonotope_supportFunc.m

Authors: Mark Wetzlinger, Victor Gassmann (MATLAB)
         Python translation by AI Assistant
Written: 27-July-2021 (MATLAB)
Python translation: 2025
"""

import numpy as np
import pytest
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.interval import Interval

TOL = 1e-10

def test_supportFunc_matlab_equivalence():
    # Empty set
    Z_empty = Zonotope.empty(2)
    dir_ = np.array([1, 1])
    val_upper, _, _ = Z_empty.supportFunc_(dir_, 'upper')
    val_lower, _, _ = Z_empty.supportFunc_(dir_, 'lower')
    assert val_upper == -np.inf
    assert val_lower == np.inf

    # Instantiate zonotope (matches MATLAB test)
    c = np.array([[2], [1]])
    G = np.array([[4, 2, -2], [1, 3, 7]])
    Z = Zonotope(c, G)

    # Check a couple of evaluations
    val_1, _, _ = Z.supportFunc_(np.array([1, 0]), 'upper')
    assert np.isclose(val_1, 10, atol=TOL), f"Expected 10, got {val_1}"
    val_2, _, _ = Z.supportFunc_(np.array([-1, 0]), 'upper')
    assert np.isclose(val_2, 6, atol=TOL), f"Expected 6, got {val_2}"
    val_3, _, _ = Z.supportFunc_(np.array([0, 1]), 'upper')
    assert np.isclose(val_3, 12, atol=TOL), f"Expected 12, got {val_3}"
    val_4, _, _ = Z.supportFunc_(np.array([0, -1]), 'upper')
    assert np.isclose(val_4, 10, atol=TOL), f"Expected 10, got {val_4}"

    # Check 'range'
    val_lower, _, _ = Z.supportFunc_(np.array([1, 0]), 'lower')
    val_upper, _, _ = Z.supportFunc_(np.array([1, 0]), 'upper')
    val_range, _, _ = Z.supportFunc_(np.array([1, 0]), 'range')
    assert isinstance(val_range, Interval)
    assert np.isclose(val_range.inf, val_lower, atol=TOL)
    assert np.isclose(val_range.sup, val_upper, atol=TOL)

    # Check a couple of support vectors
    _, x1, _ = Z.supportFunc_(np.array([1, 1]), 'upper')
    _, x2, _ = Z.supportFunc_(np.array([-1, 1]), 'upper')
    _, x3, _ = Z.supportFunc_(np.array([-1, -1]), 'upper')
    expected = np.array([[6, -2, -2], [12, 10, -10]])
    actual = np.column_stack([x1, x2, x3])
    assert np.allclose(actual, expected, atol=TOL), f"Support vectors do not match.\nExpected:\n{expected}\nActual:\n{actual}" 