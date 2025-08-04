"""
Test cases for convHull_ method of EmptySet class
"""

import pytest
import numpy as np
from cora_python.contSet.emptySet import EmptySet
from cora_python.contSet.interval import Interval


def test_convHull_single_argument():
    """Test convHull with single argument (empty set is convex)"""
    O = EmptySet(2)
    result = O.convHull_()
    # MATLAB returns nothing (None in Python) for single argument
    assert result is None


def test_convHull_with_interval():
    """Test convHull with interval set"""
    O = EmptySet(2)
    I = Interval([1, 2], [3, 4])
    result = O.convHull_(I)
    # Should return a copy of the interval
    assert isinstance(result, Interval)
    # Check that it's a copy by comparing the internal representation
    assert np.array_equal(result.inf, I.inf)
    assert np.array_equal(result.sup, I.sup)
    # The copy should be a different object (even if values are the same)
    assert id(result) != id(I)


def test_convHull_with_numeric():
    """Test convHull with numeric input"""
    O = EmptySet(2)
    point = np.array([1, 2])
    result = O.convHull_(point)
    # Should return the numeric value directly
    assert np.array_equal(result, point)


def test_convHull_dimension_mismatch():
    """Test convHull with dimension mismatch (should raise error)"""
    O = EmptySet(2)
    I = Interval([1], [2])  # 1D interval
    with pytest.raises(Exception):  # Should raise dimension mismatch error
        O.convHull_(I)


def test_convHull_precedence_handling():
    """Test convHull precedence handling"""
    O = EmptySet(2)
    # Create a mock object with lower precedence
    class MockSet:
        def __init__(self):
            self.precedence = -1
        
        def convHull(self, other, *args):
            return "called_mock_convHull"
    
    mock_set = MockSet()
    result = O.convHull_(mock_set)
    assert result == "called_mock_convHull"


def test_convHull_with_additional_args():
    """Test convHull with additional arguments"""
    O = EmptySet(2)
    I = Interval([1, 2], [3, 4])
    result = O.convHull_(I, "method1", "method2")
    # Should handle additional arguments gracefully
    assert isinstance(result, Interval) 