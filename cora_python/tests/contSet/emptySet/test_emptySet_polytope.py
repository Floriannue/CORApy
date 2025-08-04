"""
Test cases for polytope method of EmptySet class
"""

import pytest
from cora_python.contSet.emptySet import EmptySet


def test_polytope_basic():
    """Test basic polytope conversion"""
    O = EmptySet(2)
    P = O.polytope()
    
    # Should return polytope that represents emptySet
    assert P.representsa_('emptySet')
    assert P.dim() == 2


def test_polytope_different_dimensions():
    """Test polytope conversion with different dimensions"""
    for n in range(1, 11):  # Test dimensions 1-10 as in MATLAB test
        O = EmptySet(n)
        P = O.polytope()
        
        # Should represent emptySet and have correct dimension
        assert P.representsa_('emptySet'), f"Failed for dimension {n}"
        assert P.dim() == n


def test_polytope_return_type():
    """Test that polytope returns correct type"""
    O = EmptySet(3)
    P = O.polytope()
    
    from cora_python.contSet.polytope import Polytope
    assert isinstance(P, Polytope)


def test_polytope_high_dimensions():
    """Test polytope conversion with higher dimensions"""
    for dim in [5, 10, 15, 20]:
        O = EmptySet(dim)
        P = O.polytope()
        
        assert P.representsa_('emptySet')
        assert P.dim() == dim 