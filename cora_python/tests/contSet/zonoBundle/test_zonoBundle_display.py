"""
test_zonoBundle_display - unit test function of display

TRANSLATED FROM: cora_matlab/unitTests/contSet/zonoBundle/test_zonoBundle_display.m

Syntax:
    pytest cora_python/tests/contSet/zonoBundle/test_zonoBundle_display.py

Inputs:
    -

Outputs:
    res - true/false

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       23-April-2023 (MATLAB)
               2025 (Python translation)
Last update:   ---
Last revision: ---
"""

import pytest
import numpy as np
from cora_python.contSet.zonoBundle.zonoBundle import ZonoBundle
from cora_python.contSet.zonotope.zonotope import Zonotope
import io
import sys


def test_zonoBundle_display_empty():
    """
    TRANSLATED TEST - Fully-empty zonoBundle display test
    """
    # fully-empty zonoBundle
    zB = ZonoBundle.empty(2)
    
    # Should not raise an error
    display_str = zB.display_()
    assert isinstance(display_str, str)
    assert len(display_str) > 0
    
    # Test that display() prints the same
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    try:
        zB.display()
        printed_output = buffer.getvalue()
        assert printed_output == display_str
    finally:
        sys.stdout = old_stdout
    
    # Test that __str__ returns the same
    assert str(zB) == display_str


def test_zonoBundle_display_non_empty_intersection():
    """
    TRANSLATED TEST - Non-empty intersection
    """
    # non-empty intersection
    Z1 = Zonotope(np.array([[1], [1]]), np.array([[3, 0], [0, 2]]))
    Z2 = Zonotope(np.array([[0], [0]]), np.array([[2, 2], [2, -2]]))
    zB = ZonoBundle([Z1, Z2])
    
    # Should not raise an error
    display_str = zB.display_()
    assert isinstance(display_str, str)
    assert len(display_str) > 0
    
    # Test that display() prints the same
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    try:
        zB.display()
        printed_output = buffer.getvalue()
        assert printed_output == display_str
    finally:
        sys.stdout = old_stdout


def test_zonoBundle_display_empty_intersection():
    """
    TRANSLATED TEST - Empty intersection
    """
    # empty intersection
    Z1 = Zonotope(np.array([[1], [1]]), np.array([[3, 0], [0, 2]]))
    Z2 = Zonotope(np.array([[-4], [1]]), np.array([[0.5, 1], [1, -1]]))
    zB = ZonoBundle([Z1, Z2])
    
    # Should not raise an error
    display_str = zB.display_()
    assert isinstance(display_str, str)
    assert len(display_str) > 0


def test_zonoBundle_display_empty_generator_matrix():
    """
    TRANSLATED TEST - Empty generator matrix
    """
    # empty generator matrix
    Z1 = Zonotope(np.array([[1], [1]]), np.array([[3, 0], [0, 2]]))
    Z3 = Zonotope(np.array([[0], [1]]))
    zB = ZonoBundle([Z1, Z3])
    
    # Should not raise an error
    display_str = zB.display_()
    assert isinstance(display_str, str)
    assert len(display_str) > 0


def test_zonoBundle_display_pattern_consistency():
    """
    Test that display_(), display(), and __str__ produce identical output
    """
    Z1 = Zonotope(np.array([[0], [0]]), np.array([[1, 0], [0, 1]]))
    Z2 = Zonotope(np.array([[1], [1]]), np.array([[1, 0], [0, 1]]))
    zB = ZonoBundle([Z1, Z2])
    
    display_str = zB.display_()
    
    # Test that display() prints the same
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    try:
        zB.display()
        printed_output = buffer.getvalue()
        assert printed_output == display_str
    finally:
        sys.stdout = old_stdout
    
    # Test that __str__ returns the same
    assert str(zB) == display_str


if __name__ == "__main__":
    pytest.main([__file__])

