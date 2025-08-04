"""
Test cases for not_op method of EmptySet class
"""

import pytest
from cora_python.contSet.emptySet import EmptySet
from cora_python.contSet.fullspace import Fullspace


def test_not_basic():
    """Test basic not functionality"""
    O = EmptySet(2)
    fs = ~O
    
    # Should return fullspace of same dimension
    assert isinstance(fs, Fullspace)
    assert fs.dimension == 2


def test_not_different_syntax():
    """Test not with different syntax"""
    O = EmptySet(3)
    
    # Test with ~ operator
    fs1 = ~O
    # Test with not_op method
    fs2 = O.not_op()
    
    # Both should return the same result
    assert isinstance(fs1, Fullspace)
    assert isinstance(fs2, Fullspace)
    assert fs1.dimension == fs2.dimension == 3


def test_not_different_dimensions():
    """Test not with different dimensions"""
    for dim in [1, 2, 4, 5, 10]:
        O = EmptySet(dim)
        fs = ~O
        
        assert isinstance(fs, Fullspace)
        assert fs.dimension == dim


def test_not_equality():
    """Test that not returns correct fullspace"""
    O = EmptySet(2)
    fs = ~O
    
    # Should be equal to a fullspace of same dimension
    expected_fs = Fullspace(2)
    assert fs.dimension == expected_fs.dimension
    # Note: We can't directly compare objects as they might have different instances


def test_not_return_type():
    """Test that not returns correct type"""
    O = EmptySet(4)
    fs = ~O
    
    assert isinstance(fs, Fullspace)
    assert hasattr(fs, 'dimension') 