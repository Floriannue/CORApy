# test_interval_randPoint - unit test function of randPoint
#
# Syntax:
#    python -m pytest cora_python/tests/contSet/interval/test_interval_randPoint.py
#
# Inputs:
#    -
#
# Outputs:
#    res - true/false

import numpy as np
import pytest
from cora_python.contSet.interval.interval import Interval


class TestIntervalRandPoint:
    """Test class for interval randPoint method."""

    def test_interval_randPoint_empty(self):
        """Test randPoint of empty interval."""
        I = Interval.empty(2)
        p = I.randPoint(5)
        assert p.shape == (2, 0)

    def test_interval_randPoint_1d(self):
        """Test randPoint of 1D interval."""
        I = Interval(np.array([[-2]]), np.array([[3]]))
        p = I.randPoint(10)
        assert p.shape == (1, 10)
        # All points should be within bounds
        assert np.all(p >= -2) and np.all(p <= 3)

    def test_interval_randPoint_2d(self):
        """Test randPoint of 2D interval."""
        I = Interval(np.array([[-2], [1]]), np.array([[3], [4]]))
        p = I.randPoint(20)
        assert p.shape == (2, 20)
        # All points should be within bounds
        assert np.all(p[0, :] >= -2) and np.all(p[0, :] <= 3)
        assert np.all(p[1, :] >= 1) and np.all(p[1, :] <= 4)

    def test_interval_randPoint_single_point(self):
        """Test randPoint with single point request."""
        I = Interval(np.array([[-1], [0]]), np.array([[1], [2]]))
        p = I.randPoint(1)
        assert p.shape == (2, 1)
        assert np.all(p[0, :] >= -1) and np.all(p[0, :] <= 1)
        assert np.all(p[1, :] >= 0) and np.all(p[1, :] <= 2)

    def test_interval_randPoint_default_count(self):
        """Test randPoint with default point count."""
        I = Interval(np.array([[-1], [0]]), np.array([[1], [2]]))
        p = I.randPoint()
        # Default should be 1 point
        assert p.shape == (2, 1)
        assert np.all(p[0, :] >= -1) and np.all(p[0, :] <= 1)
        assert np.all(p[1, :] >= 0) and np.all(p[1, :] <= 2)

    def test_interval_randPoint_point_interval(self):
        """Test randPoint of point interval."""
        point = np.array([[1], [2], [3]])
        I = Interval(point)
        p = I.randPoint(5)
        assert p.shape == (3, 5)
        # All points should be exactly the point
        assert np.allclose(p, np.tile(point, (1, 5)))

    def test_interval_randPoint_degenerate(self):
        """Test randPoint of degenerate interval (some dimensions are points)."""
        I = Interval(
            np.array([[-1], [2], [-2]]),  # y is fixed at 2
            np.array([[1], [2], [1]])
        )
        p = I.randPoint(10)
        assert p.shape == (3, 10)
        # x and z should vary, y should be constant
        assert np.all(p[0, :] >= -1) and np.all(p[0, :] <= 1)
        assert np.all(p[1, :] == 2)  # y is fixed
        assert np.all(p[2, :] >= -2) and np.all(p[2, :] <= 1)

    def test_interval_randPoint_large_count(self):
        """Test randPoint with large number of points."""
        I = Interval(np.array([[0], [0]]), np.array([[1], [1]]))
        p = I.randPoint(1000)
        assert p.shape == (2, 1000)
        # All points should be within bounds
        assert np.all(p >= 0) and np.all(p <= 1)
        # Points should be distributed (check variance is not zero)
        assert np.var(p[0, :]) > 0
        assert np.var(p[1, :]) > 0

    def test_interval_randPoint_unbounded(self):
        """Test randPoint of unbounded interval should handle gracefully."""
        I = Interval(
            np.array([[-np.inf], [-1]]),
            np.array([[1], [1]])
        )
        # This should either work (with bounded fallback) or raise an appropriate error
        try:
            p = I.randPoint(5)
            # If it works, bounded dimensions should be respected
            assert np.all(p[1, :] >= -1) and np.all(p[1, :] <= 1)
        except (ValueError, RuntimeError):
            # It's acceptable for unbounded intervals to raise an error
            pass

    def test_interval_randPoint_zero_count(self):
        """Test randPoint with zero points."""
        I = Interval(np.array([[-1], [0]]), np.array([[1], [2]]))
        p = I.randPoint(0)
        assert p.shape == (2, 0)


def test_interval_randPoint():
    """Test function for interval randPoint method.
    
    Runs all test methods to verify correct implementation.
    """
    test = TestIntervalRandPoint()
    test.test_interval_randPoint_empty()
    test.test_interval_randPoint_1d()
    test.test_interval_randPoint_2d()
    test.test_interval_randPoint_single_point()
    test.test_interval_randPoint_default_count()
    test.test_interval_randPoint_point_interval()
    test.test_interval_randPoint_degenerate()
    test.test_interval_randPoint_large_count()
    test.test_interval_randPoint_unbounded()
    test.test_interval_randPoint_zero_count()
    
    print("test_interval_randPoint: all tests passed")


if __name__ == "__main__":
    test_interval_randPoint() 