"""
Test cases for getPrintSetInfo method of EmptySet class
"""

import pytest
from cora_python.contSet.emptySet import EmptySet


def test_getPrintSetInfo_basic():
    """Test basic getPrintSetInfo functionality"""
    O = EmptySet(2)
    abbrev, propertyOrder = O.getPrintSetInfo()
    
    assert abbrev == 'O'
    assert propertyOrder == ['dimension']


def test_getPrintSetInfo_different_dimensions():
    """Test getPrintSetInfo with different dimensions"""
    O1 = EmptySet(1)
    O2 = EmptySet(5)
    O3 = EmptySet(10)
    
    # All should return the same values regardless of dimension
    for O in [O1, O2, O3]:
        abbrev, propertyOrder = O.getPrintSetInfo()
        assert abbrev == 'O'
        assert propertyOrder == ['dimension']


def test_getPrintSetInfo_return_type():
    """Test that getPrintSetInfo returns correct types"""
    O = EmptySet(3)
    result = O.getPrintSetInfo()
    
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], str)
    assert isinstance(result[1], list) 