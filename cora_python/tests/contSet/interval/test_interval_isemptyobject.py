# test_interval_isemptyobject - unit test function of isemptyobject
#
# Syntax:
#    python -m pytest cora_python/tests/contSet/interval/test_interval_isemptyobject.py
#
# Inputs:
#    -
#
# Outputs:
#    res - true/false

import numpy as np
import pytest
from cora_python.contSet.interval.interval import Interval


class TestIntervalIsemptyobject:
    """Test class for interval isemptyobject method."""

    def test_interval_isemptyobject_empty(self):
        """Test isemptyobject with empty interval."""
        I = Interval.empty(2)
        assert I.is_empty()

    def test_interval_isemptyobject_3d(self):
        """Test isemptyobject with 3D interval."""
        lb = np.array([[-2], [-1], [-3]])
        ub = np.array([[1], [1], [2]])
        I = Interval(lb, ub)
        assert not I.is_empty()

    def test_interval_isemptyobject_matrix(self):
        """Test isemptyobject with interval matrix."""
        lb = np.array([[-2, 0], [-1, 1], [-3, -2]])
        ub = np.array([[1, 1], [1, 2], [2, 0]])
        I = Interval(lb, ub)
        assert not I.is_empty()

    def test_interval_isemptyobject_point(self):
        """Test isemptyobject with point interval."""
        I = Interval(np.array([[1], [2], [3]]))
        assert not I.is_empty()

    def test_interval_isemptyobject_unbounded(self):
        """Test isemptyobject with unbounded interval."""
        I = Interval(
            np.array([[-np.inf], [-2]]),
            np.array([[3], [np.inf]])
        )
        assert not I.is_empty()

    def test_interval_isemptyobject_degenerate(self):
        """Test isemptyobject with degenerate interval."""
        # Zero-width interval (should not be empty object)
        I = Interval(np.array([[1], [2]]), np.array([[1], [2]]))
        assert not I.is_empty()

    def test_interval_isemptyobject_large_dimension(self):
        """Test isemptyobject with large dimension empty interval."""
        I = Interval.empty(10)
        assert I.is_empty()

    def test_interval_isemptyobject_large_dimension_non_empty(self):
        """Test isemptyobject with large dimension non-empty interval."""
        n = 10
        lb = -np.ones((n, 1))
        ub = np.ones((n, 1))
        I = Interval(lb, ub)
        assert not I.is_empty()


def test_interval_isemptyobject():
    """Test function for interval isemptyobject method.
    
    Runs all test methods to verify correct implementation.
    """
    test = TestIntervalIsemptyobject()
    test.test_interval_isemptyobject_empty()
    test.test_interval_isemptyobject_3d()
    test.test_interval_isemptyobject_matrix()
    test.test_interval_isemptyobject_point()
    test.test_interval_isemptyobject_unbounded()
    test.test_interval_isemptyobject_degenerate()
    test.test_interval_isemptyobject_large_dimension()
    test.test_interval_isemptyobject_large_dimension_non_empty()
    
    print("test_interval_isemptyobject: all tests passed")


if __name__ == "__main__":
    test_interval_isemptyobject() 