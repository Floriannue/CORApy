"""
test_conPolyZono_display - unit test function of display

TRANSLATED FROM: cora_matlab/unitTests/contSet/conPolyZono/test_conPolyZono_display.m

Syntax:
    pytest cora_python/tests/contSet/conPolyZono/test_conPolyZono_display.py

Inputs:
    -

Outputs:
    res - true/false

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Tobias Ladner (MATLAB)
               Python translation by AI Assistant
Written:       04-February-2024 (MATLAB)
               2025 (Python translation)
Last update:   ---
Last revision: ---
"""

import pytest
import numpy as np
from cora_python.contSet.conPolyZono.conPolyZono import ConPolyZono
import io
import sys


def test_conPolyZono_display_empty():
    """
    TRANSLATED TEST - Empty conPolyZono display test
    """
    # test empty set
    cPZ = ConPolyZono.empty(3)
    
    # Should not raise an error
    display_str = cPZ.display_()
    assert isinstance(display_str, str)
    assert len(display_str) > 0
    
    # Test that display() prints the same
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    try:
        cPZ.display()
        printed_output = buffer.getvalue()
        assert printed_output == display_str
    finally:
        sys.stdout = old_stdout
    
    # Test that __str__ returns the same
    assert str(cPZ) == display_str


def test_conPolyZono_display_example():
    """
    TRANSLATED TEST - Example from docstring
    """
    # test example in docstring
    c = np.array([[0], [0]])
    G = np.array([[1, 0, 1, -1], [0, 1, 1, 1]])
    E = np.array([[1, 0, 1, 2], [0, 1, 1, 0], [0, 0, 1, 1]])
    A = np.array([[1, -0.5, 0.5]])
    b = np.array([[0.5]])
    EC = np.array([[0, 1, 2], [1, 0, 0], [0, 1, 0]])
    
    cPZ = ConPolyZono(c, G, E, A, b, EC)
    
    # check display function
    display_str = cPZ.display_()
    assert isinstance(display_str, str)
    assert len(display_str) > 0
    
    # Test that display() prints the same
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    try:
        cPZ.display()
        printed_output = buffer.getvalue()
        assert printed_output == display_str
    finally:
        sys.stdout = old_stdout
    
    # Test that __str__ returns the same (check direct output)
    assert str(cPZ) == display_str


def test_conPolyZono_display_pattern_consistency():
    """
    Test that display_(), display(), and __str__ produce identical output
    """
    c = np.array([[1], [2]])
    G = np.array([[1, 0], [0, 1]])
    E = np.array([[1, 0], [0, 1]])
    
    cPZ = ConPolyZono(c, G, E)
    
    display_str = cPZ.display_()
    
    # Test that display() prints the same
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    try:
        cPZ.display()
        printed_output = buffer.getvalue()
        assert printed_output == display_str
    finally:
        sys.stdout = old_stdout
    
    # Test that __str__ returns the same
    assert str(cPZ) == display_str


if __name__ == "__main__":
    pytest.main([__file__])

