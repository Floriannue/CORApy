"""
Test file for interval volume_ method - translated from MATLAB

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 28-August-2019 (MATLAB)
Last update: 04-December-2023 (MW, add degenerate and unbounded cases) (MATLAB)
Python translation: 2025
"""

import pytest
import numpy as np
from cora_python.contSet.interval.interval import Interval
from cora_python.contSet.interval.volume_ import volume_
from cora_python.g.functions.matlab.validate.check import withinTol


def test_interval_volume_():
    """Test interval volume_ method - translated from MATLAB"""
    
    tol = 1e-9
    
    # empty
    I = Interval.empty(2)
    vol = volume_(I)
    vol_true = 0
    assert withinTol(vol, vol_true, tol)
    
    # bounded, full-dimensional
    I = Interval([-2, -4, -3], [3, 1, 2])
    vol = volume_(I)
    vol_true = 125
    assert withinTol(vol, vol_true, tol)
    
    # bounded, degenerate
    I = Interval([0, 2], [1, 2])
    vol = volume_(I)
    vol_true = 0
    assert withinTol(vol, vol_true, tol)
    
    # unbounded, full-dimensional
    I = Interval([-np.inf, -2], [1, 2])
    vol = volume_(I)
    vol_true = np.inf
    assert withinTol(vol, vol_true, tol)
    
    # unbounded, degenerate
    I = Interval([-np.inf, 2], [1, 2])
    vol = volume_(I)
    vol_true = 0
    assert withinTol(vol, vol_true, tol)
    
    # matrix
    lb = np.array([[-0.265, -0.520], [-0.713, -1.349]])
    ub = np.array([[2.496, 0.546], [1.357, 4.379]])
    I = Interval(lb, ub)
    vol = volume_(I)
    vol_true = 34.8977
    assert withinTol(vol, vol_true, 1e-3)
    
    # n-d arrays
    lb = np.reshape([-0.785, -2.226, -0.776, -1.997, -3.416, -2.200, 0.121, -0.638, -0.458, -3.367, -2.707, -3.100], [2, 2, 3])
    ub = np.reshape([2.633, -1.267, 0.718, -0.164, 2.115, 2.613, 1.828, -0.064, 0.519, 0.399, 0.426, 1.443], [2, 2, 3])
    I = Interval(lb, ub)
    vol = volume_(I)
    vol_true = 12261.587
    assert withinTol(vol, vol_true, 1e-2)
    
    # 1D interval
    I = Interval([-2], [3])
    vol = volume_(I)
    vol_true = 5
    assert withinTol(vol, vol_true, tol)


if __name__ == "__main__":
    test_interval_volume_()
    print("All volume_ tests passed!") 