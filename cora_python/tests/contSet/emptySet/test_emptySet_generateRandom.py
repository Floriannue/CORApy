"""
test_emptySet_generateRandom - unit test function of generateRandom

Syntax:
    res = test_emptySet_generateRandom()

Inputs:
    -

Outputs:
    res - true/false
"""

import pytest
import numpy as np
from cora_python.contSet.emptySet import EmptySet


def test_generateRandom_no_args():
    """Test generateRandom without input arguments."""
    # without input arguments
    O = EmptySet.generateRandom()
    
    # Should be an EmptySet
    assert isinstance(O, EmptySet)
    # Default dimension should be reasonable (check implementation)
    assert O.dim() >= 1


def test_generateRandom_with_dimension():
    """Test generateRandom with dimension specified."""
    # dimension given
    n = 2
    O = EmptySet.generateRandom('Dimension', n)
    
    # Should be an EmptySet with correct dimension
    assert isinstance(O, EmptySet)
    assert O.dim() == n


def test_generateRandom_various_dimensions():
    """Test generateRandom with various dimensions."""
    for n in [1, 3, 5, 10]:
        O = EmptySet.generateRandom('Dimension', n)
        assert isinstance(O, EmptySet)
        assert O.dim() == n


def test_generateRandom_static_method():
    """Test that generateRandom is a static method."""
    # Should be callable on class, not just instance
    O = EmptySet.generateRandom('Dimension', 4)
    assert isinstance(O, EmptySet)
    assert O.dim() == 4


if __name__ == "__main__":
    test_generateRandom_no_args()
    test_generateRandom_with_dimension()
    test_generateRandom_various_dimensions()
    test_generateRandom_static_method()
    print("All tests passed!") 