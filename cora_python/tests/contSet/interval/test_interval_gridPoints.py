import pytest
import numpy as np
from cora_python.contSet.interval.interval import Interval

def test_interval_gridPoints():
    tol = 1e-12

    # 1. empty case
    n = 2
    I = Interval.empty(n)
    vals = I.gridPoints(5)
    assert vals.size == 0 and vals.shape[0] == n

    # 2. interval
    # full-dimensional non-empty case: segments scalar, greater than 1
    I_fullDim = Interval(np.array([[-1], [-2]]), np.array([[1], [4]]))
    segments = 3
    vals = I_fullDim.gridPoints(segments)
    vals_true = np.array([
        [-1, -1, -1, 0, 0, 0, 1, 1, 1],
        [-2, 1, 4, -2, 1, 4, -2, 1, 4]
    ])
    assert np.allclose(vals, vals_true, atol=tol)

    # full-dimensional non-empty case: segments vector, some are 1
    segments = np.array([3, 1])
    vals = I_fullDim.gridPoints(segments)
    vals_true = np.array([
        [-1, 0, 1],
        [1, 1, 1]
    ])
    assert np.allclose(vals, vals_true, atol=tol)
         
    # full-dimensional non-empty case: segments vector, greater than 1
    segments = np.array([3, 2])
    vals = I_fullDim.gridPoints(segments)
    vals_true = np.array([
        [-1, -1, 0, 0, 1, 1],
        [-2, 4, -2, 4, -2, 4]
    ])
    assert np.allclose(vals, vals_true, atol=tol)

    # full-dimensional non-empty case: transposed, segments vector, greater than 1
    segments = np.array([3, 2])
    vals = I_fullDim.T.gridPoints(segments)
    vals_true = np.array([
        [-1, -1, 0, 0, 1, 1],
        [-2, 4, -2, 4, -2, 4]
    ]).T
    assert np.allclose(vals, vals_true, atol=tol)

    # non-empty case, partially radius 0: segments scalar, greater than 1
    I_rpart0 = Interval(np.array([[-1], [-2]]), np.array([[1], [-2]]))
    segments = 3
    vals = I_rpart0.gridPoints(segments)
    vals_true = np.array([
        [-1, 0, 1],
        [-2, -2, -2]
    ])
    assert np.allclose(vals, vals_true, atol=tol)

    # all radius 0: segments scalar, greater than 1
    I_r0 = Interval(np.array([[1], [-2]]), np.array([[1], [-2]]))
    segments = 3
    vals = I_r0.gridPoints(segments)
    vals_true = np.array([[1], [-2]])
    assert np.allclose(vals, vals_true, atol=tol)
    
    # 3. interval matrix
    I = Interval(np.array([[-5, -2], [3, -4]]), np.array([[1, 2], [5, 2]]))
    segments = 3
    vals = I.gridPoints(segments)
    assert isinstance(vals, list) and all(v.shape == I.inf.shape for v in vals)

    segments_mat = np.array([[3, 2], [6, 4]])
    vals_mat = I.gridPoints(segments_mat)
    assert isinstance(vals_mat, list) and all(v.shape == I.inf.shape for v in vals_mat) 