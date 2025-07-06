import pytest
import numpy as np
from cora_python.contSet.interval.interval import Interval
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def test_interval_enlarge():
    # bounded
    I = Interval(np.array([[-2], [-4], [-3]]), np.array([[2], [3], [1]]))
    # ...bounded scaling factor
    factor = 2
    I_enlarge = I.enlarge(factor)
    I_true = Interval(np.array([[-4], [-7.5], [-5]]), np.array([[4], [6.5], [3]]))
    assert I_enlarge.isequal(I_true)

    # ...Inf as scaling factor
    factor = np.inf
    I_enlarge = I.enlarge(factor)
    I_true = Interval(-np.inf * np.ones((3, 1)), np.inf * np.ones((3, 1)))
    assert I_enlarge.isequal(I_true)

    # ...0 as scaling factor
    factor = 0
    I_enlarge = I.enlarge(factor)
    I_true = Interval(I.center())
    assert I_enlarge.isequal(I_true)

    # unbounded
    I = Interval(np.array([[-np.inf], [-2]]), np.array([[2], [np.inf]]))
    # ...bounded scaling factor
    factor = 2
    I_enlarge = I.enlarge(factor)
    I_true = Interval(-np.inf * np.ones((2, 1)), np.inf * np.ones((2, 1)))
    assert I_enlarge.isequal(I_true)

    # ...Inf as scaling factor
    factor = np.inf
    I_enlarge = I.enlarge(factor)
    I_true = Interval(-np.inf * np.ones((2, 1)), np.inf * np.ones((2, 1)))
    assert I_enlarge.isequal(I_true)

    # ...0 as scaling factor (unbounded case throws an error)
    factor = 0
    with pytest.raises(CORAerror) as e:
        I.enlarge(factor)
    assert e.value.id == 'CORA:notSupported'
    
    # n-d arrays
    lb = np.reshape(np.array([ 1.000, 3.000, 2.000, 5.000, -3.000, 0.000, 2.000, 1.000, 0.000, -2.000, -1.000, 3.000, 0.000, 0.000, 0.000, 0.000, 1.000, -1.000, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000 ]), (2,2,2,3))
    ub = np.reshape(np.array([ 1.500, 4.000, 4.000, 10.000, -1.000, 0.000, 3.000, 2.000, 1.000, 0.000, 2.000, 4.000, 0.000, 0.000, 0.000, 0.000, 2.000, -0.500, 3.000, 2.000, 0.000, 0.000, 0.000, 0.000 ]), (2,2,2,3))
    I = Interval(lb,ub)
    I_enlarge = I.enlarge(2)
    inf = np.reshape(np.array([ 0.750, 2.500, 1.000, 2.500, -4.000, 0.000, 1.500, 0.500, -0.500, -3.000, -2.500, 2.500, 0.000, 0.000, 0.000, 0.000, 0.500, -1.250, 0.000, -1.000, 0.000, 0.000, 0.000, 0.000 ]), (2,2,2,3))
    sup = np.reshape(np.array([ 1.750, 4.500, 5.000, 12.500, 0.000, 0.000, 3.500, 2.500, 1.500, 1.000, 3.500, 4.500, 0.000, 0.000, 0.000, 0.000, 2.500, -0.250, 4.000, 3.000, 0.000, 0.000, 0.000, 0.000 ]), (2,2,2,3))
    I_true = Interval(inf,sup)
    assert I_enlarge.isequal(I_true) 