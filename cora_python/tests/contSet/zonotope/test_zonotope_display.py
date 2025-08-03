"""
test_zonotope_display - unit test function of display

Syntax:
    pytest cora_python/tests/contSet/zonotope/test_zonotope_display.py

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 28-April-2023 (MATLAB)
               2025 (Tiange Yang, Florian Nüssel, Python translation by AI Assistant)
"""

import pytest
import numpy as np
from cora_python.contSet.zonotope import Zonotope

def test_empty_zonotope_display():
    n = 2
    Z = Zonotope.empty(n)
    display_str = Z.display()
    assert isinstance(display_str, str)
    assert len(display_str) > 0

def test_2d_zonotope_display():
    c = np.array([-2, 1])
    G = np.array([[2, 4, 5, 3, 3], [0, 3, 5, 2, 3]])
    Z = Zonotope(c, G)
    display_str = Z.display()
    assert isinstance(display_str, str)
    assert len(display_str) > 0

def test_no_generator_matrix_display():
    c = np.array([-2, 1])
    Z = Zonotope(c)
    display_str = Z.display()
    assert isinstance(display_str, str)
    assert len(display_str) > 0

def test_many_generators_display():
    c = np.array([-2, 1])
    G = np.ones((2, 25))
    Z = Zonotope(c, G)
    display_str = Z.display()
    assert isinstance(display_str, str)
    assert len(display_str) > 0 