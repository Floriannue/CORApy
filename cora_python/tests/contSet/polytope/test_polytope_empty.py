import numpy as np
import pytest
from cora_python.contSet.polytope import Polytope
from cora_python.contSet.polytope.empty import empty

def test_empty_constructor():
    """Tests the correct construction of empty polytopes."""
    # Test default constructor (n=0)
    p0 = empty()
    assert p0.dim() == 0
    # An empty set should not contain anything, including the origin of its own space
    assert not p0.contains(np.array([]))

    # Test 2D empty polytope
    p2 = empty(2)
    assert p2.dim() == 2
    # Check that it's defined by the correct infeasible constraint
    assert np.allclose(p2.A, np.zeros((1, 2)))
    assert np.allclose(p2.b, np.array([[-1]]))
    # An empty set should not contain the origin
    assert not p2.contains(np.zeros((2, 1)))

    # Test with a larger dimension
    p10 = empty(10)
    assert p10.dim() == 10
    assert not p10.contains(np.zeros((10, 1)))


def test_empty_cache_values():
    """Tests that empty polytopes have correct cache values set like MATLAB."""
    p = empty(3)
    
    # Test cache values are set correctly (comparing to MATLAB lines 40-46)
    assert p._emptySet_val == True        # P_out.emptySet.val = true;
    assert p._bounded_val == True         # P_out.bounded.val = true;
    assert p._fullDim_val == False        # P_out.fullDim.val = false;
    assert p._minHRep_val == True         # P_out.minHRep.val = true;
    assert p._minVRep_val == True         # P_out.minVRep.val = true;
    
    # Test representation flags
    assert p.isVRep == True               # P_out.isVRep.val = true;
    assert p.isHRep == True               # Should be true from constructor
    
    # Test V matrix
    assert p._V.shape == (3, 0)           # P_out.V_.val = zeros(n,0);
    
    # Test that property access works
    assert p.minVRep == True
    assert p.minHRep == True
    # According to MATLAB, isemptyobject returns false for polytope.empty() 
    # because it has constraints (even though they're infeasible)
    assert p.isemptyobject() == False
    # But it should represent an empty set
    assert p.representsa_('emptySet', 1e-12) == True


def test_empty_invalid_input():
    """Tests that the empty function handles invalid input correctly."""
    # Test invalid input (non-integer dimension)
    with pytest.raises(ValueError):
        empty(2.5)

    # Test invalid input (negative dimension)
    with pytest.raises(ValueError):
        empty(-1)

    # Test invalid input (non-numeric)
    with pytest.raises((ValueError, TypeError)):
        empty('a') 