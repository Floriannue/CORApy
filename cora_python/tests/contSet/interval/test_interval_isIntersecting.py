import pytest
import numpy as np
from cora_python.contSet.interval.interval import Interval
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def test_interval_isIntersecting():
    # empty case
    I1_empty = Interval.empty(1)
    I2_empty = Interval(np.array([-1]), np.array([1]))
    assert not I1_empty.isIntersecting(I2_empty)
    assert not I2_empty.isIntersecting(I1_empty)

    # bounded intersecting
    I1_b_int = Interval(np.array([-2, -1]), np.array([1, 2]))
    I2_b_int = Interval(np.array([-4, -2]), np.array([-1, 0]))
    assert I1_b_int.isIntersecting(I2_b_int)
    assert I2_b_int.isIntersecting(I1_b_int)

    # bounded non-intersecting
    I1_b_noint = Interval(np.array([-2, -1]), np.array([1, 2]))
    I2_b_noint = Interval(np.array([-4, -2]), np.array([-3, 0]))
    assert not I1_b_noint.isIntersecting(I2_b_noint)
    assert not I2_b_noint.isIntersecting(I1_b_noint)

    # bounded and unbounded
    I1_bu = Interval(np.array([-2, -1]), np.array([1, 2]))
    I2_bu = Interval(np.array([-4, 1]), np.array([-1, np.inf]))
    assert I1_bu.isIntersecting(I2_bu)
    assert I2_bu.isIntersecting(I1_bu)

    # unbounded and unbounded
    I1_uu = Interval(np.array([-np.inf]), np.array([0]))
    I2_uu = Interval(np.array([-1]), np.array([np.inf]))
    assert I1_uu.isIntersecting(I2_uu)
    assert I2_uu.isIntersecting(I1_uu)

    # check numeric
    I_numeric = Interval(np.array([2, 4]), np.array([3, 5]))
    points_inside = I_numeric.randPoint(10)
    points_outside = np.array([[1, 1], [1, 2]])
    assert np.all(I_numeric.isIntersecting(points_inside))
    assert not np.any(I_numeric.isIntersecting(points_outside))
    
    # dimension mismatch
    I1_dim = Interval(np.array([-1]), np.array([1]))
    I2_dim = Interval(np.array([-1, -2]), np.array([2, 1]))
    with pytest.raises(CORAerror) as e:
        I1_dim.isIntersecting(I2_dim)
    assert e.value.id == "CORA:dimensionMismatch"

    # n-d arrays
    lb = np.reshape(np.array([1., 3., 2., 5., -3., 0., 2., 1., 0., -2., -1., 3., 0., 0., 0., 0., 1., -1., 1., 0., 0., 0., 0., 0.]), (2, 2, 2, 3))
    ub = np.reshape(np.array([1.5, 4., 4., 10., -1., 0., 3., 2., 1., 0., 2., 4., 0., 0., 0., 0., 2., -0.5, 3., 2., 0., 0., 0., 0.]), (2, 2, 2, 3))
    I_nd = Interval(lb, ub)
    assert I_nd.isIntersecting(I_nd) 