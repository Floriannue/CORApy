import numpy as np
import pytest
from cora_python.contSet import Interval
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def test_boundaryPoint():
    # tolerance
    tol = 1e-14

    # empty set
    I = Interval.empty(2)
    dir_ = np.array([1, 1])
    x = I.boundaryPoint(dir_)
    assert x.shape == (2, 0)

    # bounded, non-degenerate
    I = Interval(np.array([-2, 1]), np.array([4, 2]))
    dir_ = np.array([1, 1])
    x = I.boundaryPoint(dir_)
    x_true = np.array([1.5, 2])
    assert np.allclose(x, x_true, tol)

    # bounded, non-degenerate, different start point
    startPoint = np.array([-1, 1])
    x = I.boundaryPoint(dir_, startPoint)
    x_true = np.array([0, 2])
    assert np.allclose(x, x_true, tol)

    # bounded, degenerate
    I = Interval(np.array([-1, 0]), np.array([-1, 4]))
    dir_ = np.array([1, 1])
    x = I.boundaryPoint(dir_)
    x_true = I.center()
    assert np.allclose(x, x_true, tol)

    # bounded, degenerate, different start point
    startPoint = np.array([-1, 0])
    x = I.boundaryPoint(dir_, startPoint)
    assert np.allclose(x, startPoint, tol)

    # unbounded
    I = Interval(np.array([-np.inf, 1]), np.array([0, 2]))
    dir_ = np.array([-1, 1])
    startPoint = np.array([0, 1])
    x = I.boundaryPoint(dir_, startPoint)
    assert x[0] == -np.inf and np.isclose(x[1], 2)

    # unbounded, but vector does reach a finite boundary
    dir_ = np.array([0, 1])
    x = I.boundaryPoint(dir_, startPoint)
    x_true = np.array([0, 2])
    assert np.allclose(x, x_true, tol)
    
    # wrong calls
    I = Interval(np.array([-2, 1]), np.array([4, 2]))
    # ...all-zero 'direction'
    with pytest.raises(CORAerror) as e:
        I.boundaryPoint(np.array([0, 0]))
    assert e.value.id == "wrongValue"
    # ...start point not in the set
    with pytest.raises(CORAerror) as e:
        I.boundaryPoint(np.array([1, 1]), np.array([-5, 10]))
    assert e.value.id == "wrongValue"
    # ...dimension mismatch
    with pytest.raises(CORAerror) as e:
        I.boundaryPoint(np.array([1, 1, 1]), np.array([-5, 10]))
    assert hasattr(e.value, 'id')
    assert e.value.id == 'CORA:dimensionMismatch'
    with pytest.raises(CORAerror) as e:
        I.boundaryPoint(np.array([1, 1]), np.array([0, 1, 0]))
    assert e.value.id == 'CORA:dimensionMismatch'

def test_boundaryPoint_nd():
    from cora_python.g.functions.matlab.validate.check.withinTol import withinTol

    lb = np.zeros((2, 2, 2, 3))
    lb[:,:,0,0] = np.array([[1, 2], [3, 5]])
    lb[:,:,0,1] = np.array([[0, -1], [-2, 3]])
    lb[:,:,0,2] = np.array([[1, 1], [-1, 0]])
    lb[:,:,1,0] = np.array([[-3, 2], [0, 1]])
    
    ub = np.zeros((2, 2, 2, 3))
    ub[:,:,0,0] = np.array([[1.5, 4], [4, 10]])
    ub[:,:,0,1] = np.array([[1, 2], [0, 4]])
    ub[:,:,0,2] = np.array([[2, 3], [-0.5, 2]])
    ub[:,:,1,0] = np.array([[-1, 3], [0, 2]])
    
    I = Interval(lb, ub)
    p = I.boundaryPoint(I.inf)
    
    p_true_flat = [ 1.416667, 4.000000, 3.333333, 8.333333, -2.500000, 0.000000, 2.833333, 1.666667, 0.500000, -1.333333, 0.333333, 4.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.666667, -0.916667, 2.166667, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000 ]
    p_true = np.reshape(p_true_flat, (2, 2, 2, 3), order='F')
    
    assert np.all(withinTol(p, p_true, 1e-6)) 