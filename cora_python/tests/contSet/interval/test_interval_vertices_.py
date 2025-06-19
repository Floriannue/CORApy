"""
Test file for interval vertices_ method - translated from MATLAB

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 28-August-2019 (MATLAB)
Last update: 28-April-2023 (MATLAB)
Python translation: 2025
"""

import pytest
import numpy as np
from cora_python.contSet.interval.interval import Interval
from cora_python.contSet.interval.vertices_ import vertices_
from cora_python.g.functions.matlab.validate.check import compareMatrices


def test_interval_vertices():
    """Test interval vertices_ method - translated from MATLAB"""
    
    tol = 1e-9
    
    # empty
    I = Interval.empty(2)
    V = vertices_(I)
    assert V.size == 0 and V.shape == (2, 0)
    
    # bounded
    lb = np.array([-2, -4])
    ub = np.array([3, 1])
    I = Interval(lb, ub)
    V = vertices_(I)
    V_true = np.array([[-2, 3, 3, -2],
                       [1, 1, -4, -4]])
    assert compareMatrices(V, V_true, tol)
    
    # unbounded
    lb = np.array([-2, -4])
    ub = np.array([3, np.inf])
    I = Interval(lb, ub)
    V = vertices_(I)
    V_true = np.array([[-2, -2, 3, 3],
                       [-4, np.inf, np.inf, -4]])
    # check result (compareMatrices cannot deal with Inf...)
    assert np.array_equal(V, V_true)
    
    # degenerate
    lb = np.array([-2, 0, 1])
    ub = np.array([5, 2, 1])
    I = Interval(lb, ub)
    V = vertices_(I)
    V_true = np.array([[-2, -2, 5, 5],
                       [0, 2, 0, 2],
                       [1, 1, 1, 1]])
    assert compareMatrices(V, V_true)
    
    # degenerate, point
    lb = np.array([1, 4, -2, 6])
    I = Interval(lb)  # Point interval
    V = vertices_(I)
    assert compareMatrices(V, lb.reshape(-1, 1))


if __name__ == '__main__':
    test_interval_vertices()
    print("All tests passed!") 