"""
test_emptySet_display - unit test function of display

Syntax:
    res = test_emptySet_display()

Inputs:
    -

Outputs:
    res - true/false
"""

import pytest
import numpy as np
from cora_python.contSet.emptySet import EmptySet


def test_display_0d():
    """Test display of 0-dimensional empty set."""
    # 0-dimensional empty set
    O = EmptySet(0)
    display_str = O.display()
    
    # Check that display returns a string
    assert isinstance(display_str, str)
    assert "emptySet" in display_str
    assert "dimension: 0" in display_str


def test_display_nd():
    """Test display of n-dimensional empty set."""
    # n-dimensional empty set
    n = 2
    O = EmptySet(n)
    display_str = O.display()
    
    # Check that display returns a string
    assert isinstance(display_str, str)
    assert "emptySet" in display_str
    assert f"dimension: {n}" in display_str


def test_str_method():
    """Test __str__ method uses display."""
    n = 3
    O = EmptySet(n)
    
    str_result = str(O)
    display_result = O.display()
    
    # Should be the same
    assert str_result == display_result


if __name__ == "__main__":
    test_display_0d()
    test_display_nd()
    test_str_method()
    print("All tests passed!") 