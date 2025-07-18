"""
Test cases for interval method of EmptySet class
"""

import pytest
from cora_python.contSet.emptySet import EmptySet


def test_interval_2d():
    """Test interval conversion for 2D emptySet"""
    O = EmptySet(2)
    I = O.interval()
    
    # Check that result represents emptySet
    assert I.representsa_('emptySet')
    # Check that dimension is preserved
    assert I.dim() == 2


def test_interval_3d():
    """Test interval conversion for 3D emptySet"""
    O = EmptySet(3)
    I = O.interval()
    
    # Check that result represents emptySet
    assert I.representsa_('emptySet')
    # Check that dimension is preserved
    assert I.dim() == 3


def test_interval_different_dimensions():
    """Test interval conversion with different dimensions"""
    for dim in [1, 4, 5, 10]:
        O = EmptySet(dim)
        I = O.interval()
        
        assert I.representsa_('emptySet')
        assert I.dim() == dim


def test_interval_return_type():
    """Test that interval returns correct type"""
    O = EmptySet(2)
    I = O.interval()
    
    from cora_python.contSet.interval import Interval
    assert isinstance(I, Interval) 