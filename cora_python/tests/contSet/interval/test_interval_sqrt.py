"""
Test file for interval sqrt method - translated from MATLAB

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 29-August-2019 (MATLAB)
Last update: 04-December-2023 (MW, add empty and unbounded cases) (MATLAB)
Python translation: 2025
"""

import pytest
import numpy as np
from cora_python.contSet.interval.interval import Interval
from cora_python.contSet.interval.sqrt import sqrt
from cora_python.contSet.interval.isequal import isequal
from cora_python.contSet.interval.representsa_ import representsa_
from cora_python.contSet.interval.abs_op import abs_op


def test_interval_sqrt():
    """Test interval sqrt method - translated from MATLAB"""
    
    tol = 1e-9
    
    # empty
    I = Interval.empty(2)
    I_sqrt = sqrt(I)
    assert representsa_(I_sqrt, 'emptySet')
    
    # bounded (only perfect squares)
    I = Interval([4, 9, 4, 16, 1], [9, 25, 36, 100, 4])
    I_sqrt = sqrt(I)
    I_true = Interval([2, 3, 2, 4, 1], [3, 5, 6, 10, 2])
    assert isequal(I_sqrt, I_true, tol)
    
    # unbounded
    I = Interval([2, 4], [np.inf, 9])
    I_sqrt = sqrt(I)
    I_true = Interval([np.sqrt(2), 2], [np.inf, 3])
    assert isequal(I_sqrt, I_true, tol)
    
    # out of bounds
    I = Interval(-2, 1)
    with pytest.raises(ValueError, match="CORA:outOfDomain"):
        sqrt(I)
    
    # n-d arrays
    inf = np.reshape([1.000, 3.000, 2.000, 5.000, -3.000, 0.000, 2.000, 1.000, 0.000, -2.000, -1.000, 3.000, 0.000, 0.000, 0.000, 0.000, 1.000, -1.000, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000], [2, 2, 2, 3])
    sup = np.reshape([1.500, 4.000, 4.000, 10.000, -1.000, 0.000, 3.000, 2.000, 1.000, 0.000, 2.000, 4.000, 0.000, 0.000, 0.000, 0.000, 2.000, -0.500, 3.000, 2.000, 0.000, 0.000, 0.000, 0.000], [2, 2, 2, 3])
    I = abs_op(Interval(inf, sup))
    Isqrt = sqrt(I)
    inf_true = np.reshape([1.000, 1.732, 1.414, 2.236, 1.000, 0.000, 1.414, 1.000, 0.000, 0.000, 0.000, 1.732, 0.000, 0.000, 0.000, 0.000, 1.000, 0.707, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000], [2, 2, 2, 3])
    sup_true = np.reshape([1.225, 2.000, 2.000, 3.162, 1.732, 0.000, 1.732, 1.414, 1.000, 1.414, 1.414, 2.000, 0.000, 0.000, 0.000, 0.000, 1.414, 1.000, 1.732, 1.414, 0.000, 0.000, 0.000, 0.000], [2, 2, 2, 3])
    I_true = Interval(inf_true, sup_true)
    assert isequal(Isqrt, I_true, 1e-3)
    
    # test simple case
    I = Interval(4, 9)
    I_sqrt = sqrt(I)
    assert isequal(I_sqrt, Interval(2, 3), tol)


if __name__ == "__main__":
    test_interval_sqrt()
    print("All sqrt tests passed!") 