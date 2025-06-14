# test_interval_isequal - unit test function of isequal
#
# Syntax:
#    python -m pytest cora_python/tests/contSet/interval/test_interval_isequal.py
#
# Inputs:
#    -
#
# Outputs:
#    res - true/false

import numpy as np
import pytest
from cora_python.contSet.interval.interval import Interval


class TestIntervalIsequal:
    """Test class for interval isequal method."""

    def test_interval_isequal_bounded_different_size(self):
        """Test isequal with bounded intervals of different sizes."""
        I1 = Interval(np.array([[3], [2], [1]]), np.array([[4], [5], [6]]))
        I2 = Interval(np.array([[3], [2]]), np.array([[4], [5]]))
        assert not I1.isequal(I2)

    def test_interval_isequal_bounded_same_size(self):
        """Test isequal with bounded intervals of same size."""
        I1 = Interval(
            np.array([[-3], [-9], [-4], [-7], [-1]]),
            np.array([[4], [2], [6], [3], [8]])
        )
        I2 = Interval(
            np.array([[-3], [-9], [-4], [-7], [-1]]),
            np.array([[4], [2], [6], [2], [8]])
        )
        assert I1.isequal(I1)
        assert not I1.isequal(I2)

    def test_interval_isequal_unbounded(self):
        """Test isequal with unbounded intervals."""
        I1 = Interval(
            np.array([[-np.inf], [-9], [-4], [-7], [-1]]),
            np.array([[4], [2], [6], [np.inf], [8]])
        )
        I2 = Interval(
            np.array([[-np.inf], [-9], [-4], [-7], [-1]]),
            np.array([[4], [2], [np.inf], [2], [8]])
        )
        assert I1.isequal(I1)
        assert not I1.isequal(I2)

    def test_interval_isequal_matrix(self):
        """Test isequal with matrix intervals."""
        I1 = Interval(
            np.array([[-2, -3, -np.inf], [-4, -np.inf, -1]]),
            np.array([[2, np.inf, 5], [np.inf, 3, 0]])
        )
        I2 = Interval(
            np.array([[-2, -3, -np.inf], [-4, -np.inf, -1]]),
            np.array([[2, np.inf, np.inf], [np.inf, 3, 0]])
        )
        I3 = I1[:, :2]  # Subset of I1
        
        assert I1.isequal(I1)
        assert not I1.isequal(I2)
        assert not I1.isequal(I3)

    def test_interval_isequal_empty(self):
        """Test isequal with empty intervals."""
        I1 = Interval.empty(2)
        I2 = Interval.empty(2)
        I3 = Interval.empty(3)
        
        assert I1.isequal(I2)
        assert not I1.isequal(I3)

    def test_interval_isequal_point(self):
        """Test isequal with point intervals."""
        I1 = Interval(np.array([[1], [2], [3]]))
        I2 = Interval(np.array([[1], [2], [3]]))
        I3 = Interval(np.array([[1], [2], [4]]))
        
        assert I1.isequal(I2)
        assert not I1.isequal(I3)

    def test_interval_isequal_tolerance(self):
        """Test isequal with tolerance."""
        tol = 1e-10
        I1 = Interval(np.array([[1.0], [2.0]]), np.array([[3.0], [4.0]]))
        I2 = Interval(
            np.array([[1.0 + tol/2], [2.0]]),
            np.array([[3.0], [4.0 + tol/2]])
        )
        I3 = Interval(
            np.array([[1.0 + tol*2], [2.0]]),
            np.array([[3.0], [4.0]])
        )
        
        # Should be equal within tolerance
        assert I1.isequal(I2, tol)
        # Should not be equal outside tolerance
        assert not I1.isequal(I3, tol)

    def test_interval_isequal_3d_arrays(self):
        """Test isequal with 3D arrays."""
        lb = np.array([
            [[1.000, 2.000], [3.000, 5.000]],
            [[-3.000, 2.000], [0.000, 1.000]]
        ])
        ub = np.array([
            [[1.500, 4.000], [4.000, 10.000]],
            [[-1.000, 3.000], [0.000, 2.000]]
        ])
        I1 = Interval(lb, ub)
        I2 = Interval(lb, ub)
        I3 = Interval(lb, ub + 1)
        
        assert I1.isequal(I2)
        assert not I1.isequal(I3)


def test_interval_isequal():
    """Test function for interval isequal method.
    
    Runs all test methods to verify correct implementation.
    """
    test = TestIntervalIsequal()
    test.test_interval_isequal_bounded_different_size()
    test.test_interval_isequal_bounded_same_size()
    test.test_interval_isequal_unbounded()
    test.test_interval_isequal_matrix()
    test.test_interval_isequal_empty()
    test.test_interval_isequal_point()
    test.test_interval_isequal_tolerance()
    test.test_interval_isequal_3d_arrays()
    
    print("test_interval_isequal: all tests passed")


if __name__ == "__main__":
    test_interval_isequal() 