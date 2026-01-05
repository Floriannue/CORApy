"""
test_levelSet_display - unit test function of display

TRANSLATED FROM: cora_matlab/unitTests/contSet/levelSet/test_levelSet_display.m

Syntax:
    pytest cora_python/tests/contSet/levelSet/test_levelSet_display.py

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
Written:       28-April-2023 (MATLAB)
               2025 (Python translation)
Last update:   ---
Last revision: ---
"""

import pytest
import numpy as np
import sympy as sp
from cora_python.contSet.levelSet.levelSet import LevelSet
import io
import sys


def test_levelSet_display_empty():
    """
    TRANSLATED TEST - Empty levelSet display test
    """
    # empty set
    ls = LevelSet.empty(2)
    
    # Should not raise an error
    display_str = ls.display_()
    assert isinstance(display_str, str)
    assert len(display_str) > 0
    
    # Test that display() prints the same
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    try:
        ls.display()
        printed_output = buffer.getvalue()
        assert printed_output == display_str
    finally:
        sys.stdout = old_stdout
    
    # Test that __str__ returns the same
    assert str(ls) == display_str


def test_levelSet_display_single_equation():
    """
    TRANSLATED TEST - Single equation with different comparison operators
    """
    # init symbolic variables
    x, y = sp.symbols('x y')
    
    # single equation with different comparison operators
    eq = x**2 + y**2 - 4
    
    # Test == operator
    ls = LevelSet(eq, [x, y], '==')
    display_str = ls.display_()
    assert isinstance(display_str, str)
    assert len(display_str) > 0
    
    # Test < operator
    ls = LevelSet(eq, [x, y], '<')
    display_str = ls.display_()
    assert isinstance(display_str, str)
    assert len(display_str) > 0
    
    # Test <= operator
    ls = LevelSet(eq, [x, y], '<=')
    display_str = ls.display_()
    assert isinstance(display_str, str)
    assert len(display_str) > 0


def test_levelSet_display_multiple_equations():
    """
    TRANSLATED TEST - Multiple equations with different comparison operators
    """
    # init symbolic variables
    a, b = sp.symbols('a b')
    
    # multiple equations with different comparison operators
    eq1 = sp.sin(a) + sp.log(b)
    eq2 = sp.Abs(a) * b
    eq = [eq1, eq2]
    
    # Test with both <=
    ls = LevelSet(eq, [a, b], ['<=', '<='])
    display_str = ls.display_()
    assert isinstance(display_str, str)
    assert len(display_str) > 0
    
    # Test with <= and <
    ls = LevelSet(eq, [a, b], ['<=', '<'])
    display_str = ls.display_()
    assert isinstance(display_str, str)
    assert len(display_str) > 0


def test_levelSet_display_independent_variables():
    """
    TRANSLATED TEST - Independent variables
    """
    # init symbolic variables
    a, b, x, y = sp.symbols('a b x y')
    
    # independent variables
    eq1 = a
    eq2 = b
    eq3 = x
    eq4 = y
    eq = [eq1, eq2, eq3, eq4]
    
    ls = LevelSet(eq, [a, b, x, y], ['<', '<=', '<', '<='])
    display_str = ls.display_()
    assert isinstance(display_str, str)
    assert len(display_str) > 0


def test_levelSet_display_unused_variable():
    """
    TRANSLATED TEST - Unused variable in vars
    """
    # init symbolic variables
    a, b, y = sp.symbols('a b y')
    
    # unused variable in vars
    eq = a + y
    ls = LevelSet(eq, [a, b, y], '==')
    display_str = ls.display_()
    assert isinstance(display_str, str)
    assert len(display_str) > 0


def test_levelSet_display_pattern_consistency():
    """
    Test that display_(), display(), and __str__ produce identical output
    """
    x, y = sp.symbols('x y')
    eq = x**2 + y**2 - 4
    ls = LevelSet(eq, [x, y], '==')
    
    display_str = ls.display_()
    
    # Test that display() prints the same
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    try:
        ls.display()
        printed_output = buffer.getvalue()
        assert printed_output == display_str
    finally:
        sys.stdout = old_stdout
    
    # Test that __str__ returns the same
    assert str(ls) == display_str


if __name__ == "__main__":
    pytest.main([__file__])

