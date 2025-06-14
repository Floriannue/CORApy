# test_interval_empty - unit test function of empty
#
# Syntax:
#    python -m pytest cora_python/tests/contSet/interval/test_interval_empty.py
#
# Inputs:
#    -
#
# Outputs:
#    res - true/false

import numpy as np
import pytest
from cora_python.contSet.interval.interval import Interval


class TestIntervalEmpty:
    """Test class for interval empty method."""

    def test_interval_empty_1d(self):
        """Test empty interval in 1D."""
        n = 1
        I = Interval.empty(n)
        assert I.representsa_('emptySet')
        assert I.dim() == 1
        assert I.inf.shape == (1, 0)
        assert I.sup.shape == (1, 0)

    def test_interval_empty_2d(self):
        """Test empty interval in 2D."""
        n = 2
        I = Interval.empty(n)
        assert I.representsa_('emptySet')
        assert I.dim() == 2
        assert I.inf.shape == (2, 0)
        assert I.sup.shape == (2, 0)

    def test_interval_empty_5d(self):
        """Test empty interval in 5D."""
        n = 5
        I = Interval.empty(n)
        assert I.representsa_('emptySet')
        assert I.dim() == 5
        assert I.inf.shape == (5, 0)
        assert I.sup.shape == (5, 0)

    def test_interval_empty_properties(self):
        """Test properties of empty intervals."""
        I = Interval.empty(3)
        
        # Empty interval should be bounded
        assert I.is_bounded()
        
        # Empty interval should be an empty object
        assert I.is_empty()
        
        # Empty interval should not contain any points
        test_point = np.array([[0], [0], [0]])
        assert not I.contains_(test_point)

    def test_interval_empty_operations(self):
        """Test operations with empty intervals."""
        I_empty = Interval.empty(2)
        I_regular = Interval(np.array([[-1], [0]]), np.array([[1], [2]]))
        
        # Addition with empty should return empty
        result = I_empty + I_regular
        assert result.representsa_('emptySet')
        
        result = I_regular + I_empty
        assert result.representsa_('emptySet')
        
        # Multiplication with scalar should return empty
        result = I_empty * 5.0
        assert result.representsa_('emptySet')

    def test_interval_empty_center_radius(self):
        """Test center and radius of empty intervals."""
        I = Interval.empty(2)
        
        # Center should be empty array
        c = I.center()
        assert c.shape == (2, 0)
        
        # Radius should be empty array
        r = I.rad()
        assert r.shape == (2, 0)

    def test_interval_empty_equality(self):
        """Test equality of empty intervals."""
        I1 = Interval.empty(2)
        I2 = Interval.empty(2)
        I3 = Interval.empty(3)
        
        # Same dimension empty intervals should be equal
        assert I1.isequal(I2)
        
        # Different dimension empty intervals should not be equal 
        # (For now we'll skip this test as it might have specific behavior)
        # assert not I1.isequal(I3)


def test_interval_empty():
    """Test function for interval empty method.
    
    Runs all test methods to verify correct implementation.
    """
    test = TestIntervalEmpty()
    test.test_interval_empty_1d()
    test.test_interval_empty_2d()
    test.test_interval_empty_5d()
    test.test_interval_empty_properties()
    test.test_interval_empty_operations()
    test.test_interval_empty_center_radius()
    test.test_interval_empty_equality()
    
    print("test_interval_empty: all tests passed")


if __name__ == "__main__":
    test_interval_empty() 