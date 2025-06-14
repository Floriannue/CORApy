# test_interval_is_bounded - unit test function of is_bounded
#
# Syntax:
#    python -m pytest cora_python/tests/contSet/interval/test_interval_is_bounded.py
#
# Inputs:
#    -
#
# Outputs:
#    res - true/false

import numpy as np
import pytest
from cora_python.contSet.interval.interval import Interval


class TestIntervalIsBounded:
    """Test class for interval is_bounded method."""

    def test_interval_is_bounded_empty(self):
        """Test is_bounded with empty interval."""
        I = Interval.empty(2)
        assert I.is_bounded()

    def test_interval_is_bounded_point(self):
        """Test is_bounded with point interval."""
        I = Interval(np.array([[1]]))
        assert I.is_bounded()

    def test_interval_is_bounded_3d(self):
        """Test is_bounded with 3D bounded interval."""
        I = Interval(
            np.array([[-2], [-1], [-3]]),
            np.array([[1], [1], [2]])
        )
        assert I.is_bounded()

    def test_interval_is_bounded_unbounded_inf(self):
        """Test is_bounded with unbounded interval (lower bound -inf)."""
        I = Interval(
            np.array([[-2], [-np.inf], [-3]]),
            np.array([[1], [1], [2]])
        )
        assert not I.is_bounded()

    def test_interval_is_bounded_unbounded_both(self):
        """Test is_bounded with unbounded interval (both bounds inf)."""
        I = Interval(
            np.array([[-2], [-np.inf], [-3]]),
            np.array([[1], [np.inf], [2]])
        )
        assert not I.is_bounded()

    def test_interval_is_bounded_unbounded_upper(self):
        """Test is_bounded with unbounded interval (upper bound inf)."""
        I = Interval(
            np.array([[-2], [-1], [-3]]),
            np.array([[1], [np.inf], [2]])
        )
        assert not I.is_bounded()

    def test_interval_is_bounded_3d_arrays(self):
        """Test is_bounded with 3D array intervals."""
        # Bounded case
        lb = np.array([
            [[1.000, 2.000], [3.000, 5.000]],
            [[-3.000, 2.000], [0.000, 1.000]]
        ])
        ub = np.array([
            [[1.500, 4.000], [4.000, 10.000]],
            [[-1.000, 3.000], [0.000, 2.000]]
        ])
        I = Interval(lb, ub)
        assert I.is_bounded()

        # Unbounded case
        lb_unbounded = lb.copy()
        lb_unbounded[0, 0, 0] = -np.inf
        I_unbounded = Interval(lb_unbounded, ub)
        assert not I_unbounded.is_bounded()

    def test_interval_is_bounded_matrix(self):
        """Test is_bounded with matrix intervals."""
        # Bounded matrix
        I_bounded = Interval(
            np.array([[-2, 0], [-1, 1], [-3, -2]]),
            np.array([[1, 1], [1, 2], [2, 0]])
        )
        assert I_bounded.is_bounded()

        # Unbounded matrix
        I_unbounded = Interval(
            np.array([[-2, 0], [-np.inf, 1], [-3, -2]]),
            np.array([[1, 1], [1, 2], [2, np.inf]])
        )
        assert not I_unbounded.is_bounded()

    def test_interval_is_bounded_fully_unbounded(self):
        """Test is_bounded with fully unbounded interval."""
        I = Interval(
            np.array([[-np.inf], [-np.inf]]),
            np.array([[np.inf], [np.inf]])
        )
        assert not I.is_bounded()

    def test_interval_is_bounded_large_values(self):
        """Test is_bounded with large but finite values."""
        large_val = 1e308  # Large but finite
        I = Interval(
            np.array([[-large_val], [-1]]),
            np.array([[large_val], [1]])
        )
        assert I.is_bounded()


def test_interval_is_bounded():
    """Test function for interval is_bounded method.
    
    Runs all test methods to verify correct implementation.
    """
    test = TestIntervalIsBounded()
    test.test_interval_is_bounded_empty()
    test.test_interval_is_bounded_point()
    test.test_interval_is_bounded_3d()
    test.test_interval_is_bounded_unbounded_inf()
    test.test_interval_is_bounded_unbounded_both()
    test.test_interval_is_bounded_unbounded_upper()
    test.test_interval_is_bounded_3d_arrays()
    test.test_interval_is_bounded_matrix()
    test.test_interval_is_bounded_fully_unbounded()
    test.test_interval_is_bounded_large_values()
    
    print("test_interval_is_bounded: all tests passed")


if __name__ == "__main__":
    test_interval_is_bounded() 