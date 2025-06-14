# test_interval_dim - unit test function of dim
#
# Syntax:
#    python -m pytest cora_python/tests/contSet/interval/test_interval_dim.py
#
# Inputs:
#    -
#
# Outputs:
#    res - true/false

import numpy as np
import pytest
from cora_python.contSet.interval.interval import Interval


class TestIntervalDim:
    """Test class for interval dim method."""

    def test_interval_dim_empty(self):
        """Test dim of empty interval."""
        # empty case
        n = 2
        I = Interval.empty(n)
        assert I.dim() == n

    def test_interval_dim_bounded_vector(self):
        """Test dim of bounded vector interval."""
        I = Interval(np.array([[-3], [-2], [-5]]), np.array([[4], [2], [1]]))
        n = I.dim()
        n_true = 3
        assert n == n_true

    def test_interval_dim_bounded_matrix(self):
        """Test dim of bounded matrix interval."""
        I = Interval(
            np.array([[-3, 0], [-2, -1], [-5, -2]]),
            np.array([[4, 1], [2, 3], [1, 1]])
        )
        n = I.dim()
        n_true = [3, 2]
        assert np.array_equal(n, n_true)

    def test_interval_dim_unbounded_vector(self):
        """Test dim of unbounded vector interval."""
        I = Interval(
            np.array([[-np.inf], [-2], [3]]),
            np.array([[4], [np.inf], [np.inf]])
        )
        n = I.dim()
        n_true = 3
        assert n == n_true

    def test_interval_dim_3d_arrays(self):
        """Test dim of 3D array intervals."""
        # n-d arrays
        lb = np.array([
            [[1.000, 2.000], [3.000, 5.000]],
            [[-3.000, 2.000], [0.000, 1.000]]
        ])
        ub = np.array([
            [[1.500, 4.000], [4.000, 10.000]],
            [[-1.000, 3.000], [0.000, 2.000]]
        ])
        I = Interval(lb, ub)
        expected_dim = [2, 2, 2]
        assert np.array_equal(I.dim(), expected_dim)

    def test_interval_dim_1d(self):
        """Test dim of 1D interval."""
        I = Interval(np.array([-2]), np.array([3]))
        assert I.dim() == 1

    def test_interval_dim_point(self):
        """Test dim of point interval."""
        I = Interval(np.array([[1], [2], [3]]))
        assert I.dim() == 3


def test_interval_dim():
    """Test function for interval dim method.
    
    Runs all test methods to verify correct implementation.
    """
    test = TestIntervalDim()
    test.test_interval_dim_empty()
    test.test_interval_dim_bounded_vector()
    test.test_interval_dim_bounded_matrix()
    test.test_interval_dim_unbounded_vector()
    test.test_interval_dim_3d_arrays()
    test.test_interval_dim_1d()
    test.test_interval_dim_point()
    
    print("test_interval_dim: all tests passed")


if __name__ == "__main__":
    test_interval_dim() 