import pytest
import numpy as np
from cora_python.contSet.interval.interval import Interval

def test_interval_isFullDim():
    # empty case
    I_empty = Interval.empty(2)
    assert not I_empty.isFullDim()

    # bounded, full-dimensional
    I_full = Interval(np.array([-2, -4, -7, -1, -2]), np.array([4, 2, 6, 4, 8]))
    assert I_full.isFullDim()

    # bounded, degenerate
    I_degen = Interval(np.array([-2, -4, 0, -1, -2]), np.array([4, 2, 0, 4, 8]))
    assert not I_degen.isFullDim()

    # unbounded, full-dimensional
    I_unbound_full = Interval(np.array([-np.inf, -2]), np.array([1, 1]))
    assert I_unbound_full.isFullDim()

    # unbounded, degenerate
    I_unbound_degen = Interval(np.array([-np.inf, 0]), np.array([1, 0]))
    assert not I_unbound_degen.isFullDim()

    # n-d arrays
    lb = np.reshape(np.array([ 1.000, 3.000, 2.000, 5.000, -3.000, 0.000, 2.000, 1.000, 0.000, -2.000, -1.000, 3.000, 0.000, 0.000, 0.000, 0.000, 1.000, -1.000, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000 ]), (2,2,2,3))
    ub = np.reshape(np.array([ 1.500, 4.000, 4.000, 10.000, -1.000, 0.000, 3.000, 2.000, 1.000, 0.000, 2.000, 4.000, 0.000, 0.000, 0.000, 0.000, 2.000, -0.500, 3.000, 2.000, 0.000, 0.000, 0.000, 0.000 ]), (2,2,2,3))
    I_nd = Interval(lb, ub)
    assert not I_nd.isFullDim() 