"""
Test cases for isIntersecting_ method of EmptySet class
"""

import pytest
import numpy as np
from cora_python.contSet.emptySet import EmptySet
from cora_python.contSet.zonotope import Zonotope


def test_isIntersecting_self():
    """Test intersection with itself"""
    O = EmptySet(2)
    result = O.isIntersecting_(O)
    
    # Should always return False
    assert result is False


def test_isIntersecting_zonotope():
    """Test intersection with zonotope"""
    O = EmptySet(2)
    Z = Zonotope(np.zeros((2, 1)))
    result = O.isIntersecting_(Z)
    
    # Should always return False
    assert result is False


def test_isIntersecting_point():
    """Test intersection with point"""
    O = EmptySet(2)
    point = np.array([1, 1])
    result = O.isIntersecting_(point)
    
    # Should always return False
    assert result is False


def test_isIntersecting_with_optional_args():
    """Test intersection with optional arguments"""
    O = EmptySet(2)
    Z = Zonotope(np.zeros((2, 1)))
    
    # Test with type and tolerance
    result = O.isIntersecting_(Z, "exact", 1e-6)
    assert result is False
    
    # Test with additional arguments
    result = O.isIntersecting_(Z, "exact", 1e-6, "extra_arg")
    assert result is False


def test_isIntersecting_different_dimensions():
    """Test intersection with different dimensions"""
    for dim in [1, 3, 5]:
        O = EmptySet(dim)
        Z = Zonotope(np.zeros((dim, 1)))
        result = O.isIntersecting_(Z)
        
        # Should always return False regardless of dimension
        assert result is False 