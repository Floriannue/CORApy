"""
Test file for interval exp method - translated from MATLAB

Authors: Dmitry Grebenyuk, Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 14-January-2016 (MATLAB)
Last update: 04-December-2023 (MW, add unbounded cases) (MATLAB)
Python translation: 2025
"""

import pytest
import numpy as np
from cora_python.contSet.interval.interval import Interval
from cora_python.contSet.interval.exp import exp
from cora_python.contSet.interval.isequal import isequal


def test_interval_exp():
    """Test interval exp method - translated from MATLAB"""
    
    # tolerance
    tol = 1e-9
    
    # bounded
    I = Interval([-5, -4, -3, 0, 0, 5], [-2, 0, 2, 0, 5, 8])
    I_exp = exp(I)
    I_true = Interval([0.006737947, 0.0183156389, 0.0497870684, 1, 1, 148.4131591026],
                      [0.135335283, 1, 7.3890560989, 1, 148.4131591026, 2980.9579870418])
    assert isequal(I_exp, I_true, tol)
    
    # unbounded
    I = Interval(-np.inf, 0)
    I_exp = exp(I)
    I_true = Interval(0, 1)
    assert isequal(I_exp, I_true, tol)
    
    # unbounded
    I = Interval(1, np.inf)
    I_exp = exp(I)
    I_true = Interval(np.exp(1), np.inf)
    assert isequal(I_exp, I_true, tol)
    
    # n-d arrays
    lb = np.reshape([1.000, 3.000, 2.000, 5.000, -3.000, 0.000, 2.000, 1.000, 0.000, -2.000, -1.000, 3.000, 0.000, 0.000, 0.000, 0.000, 1.000, -1.000, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000], [2, 2, 2, 3])
    ub = np.reshape([1.500, 4.000, 4.000, 10.000, -1.000, 0.000, 3.000, 2.000, 1.000, 0.000, 2.000, 4.000, 0.000, 0.000, 0.000, 0.000, 2.000, -0.500, 3.000, 2.000, 0.000, 0.000, 0.000, 0.000], [2, 2, 2, 3])
    I = Interval(lb, ub)
    I_exp = exp(I)
    assert isequal(I_exp, Interval(np.exp(lb), np.exp(ub)))
    
    # empty interval
    I_empty = Interval.empty(2)
    I_exp_empty = exp(I_empty)
    assert I_exp_empty.is_empty()


if __name__ == "__main__":
    test_interval_exp()
    print("All exp tests passed!") 