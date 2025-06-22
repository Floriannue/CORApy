"""
test_emptySet_representsa - unit test function of representsa_

Syntax:
    res = test_emptySet_representsa()

Inputs:
    -

Outputs:
    res - true/false
"""

import pytest
import numpy as np
from cora_python.contSet.emptySet import EmptySet


def test_representsa_negative_cases():
    """Test representations that empty set cannot represent."""
    # init empty set
    n = 2
    O = EmptySet(n)
    
    # compare to other representations - should be False
    assert not O.representsa_('origin')
    assert not O.representsa_('point')


def test_representsa_positive_cases():
    """Test representations that empty set can represent."""
    # init empty set
    n = 2
    O = EmptySet(n)
    
    # compare to other representations - should be True
    assert O.representsa_('emptySet')
    assert O.representsa_('interval')
    assert O.representsa_('zonotope')


def test_representsa_various_dimensions():
    """Test representsa for various dimensions."""
    for n in [1, 3, 5]:
        O = EmptySet(n)
        
        # These should always be True for empty sets
        assert O.representsa_('emptySet')
        assert O.representsa_('interval')
        assert O.representsa_('zonotope')
        
        # These should always be False for empty sets
        assert not O.representsa_('origin')
        assert not O.representsa_('point')


def test_representsa_case_sensitivity():
    """Test case sensitivity of representsa."""
    O = EmptySet(2)
    
    # Test various cases
    assert O.representsa_('emptySet')
    assert O.representsa_('emptyset')  # Should handle case variations if implemented
    assert O.representsa_('interval')
    assert O.representsa_('zonotope')


if __name__ == "__main__":
    test_representsa_negative_cases()
    test_representsa_positive_cases()
    test_representsa_various_dimensions()
    test_representsa_case_sensitivity()
    print("All tests passed!") 