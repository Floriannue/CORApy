import pytest
import numpy as np
from cora_python.contSet.interval.interval import Interval

def test_interval_dirac():
    # Test case 1: 0 not in interval
    I1 = Interval(np.array([1, 2]), np.array([3, 4]))
    res1 = I1.dirac()
    true1 = Interval(np.array([0, 0]), np.array([0, 0]))
    assert res1.isequal(true1)

    # Test case 2: 0 is in interval
    I2 = Interval(np.array([-1, 2]), np.array([1, 4]))
    res2 = I2.dirac()
    true2 = Interval(np.array([0, 0]), np.array([np.inf, 0]))
    assert res2.isequal(true2)

    # Test case 3: 0 is at the boundary (max)
    I3 = Interval(np.array([-2, -3]), np.array([0, -1]))
    res3 = I3.dirac()
    true3 = Interval(np.array([0, 0]), np.array([np.inf, 0]))
    assert res3.isequal(true3)

    # Test case 4: 0 is at the boundary (min)
    I4 = Interval(np.array([0, 2]), np.array([1, 4]))
    res4 = I4.dirac()
    true4 = Interval(np.array([0, 0]), np.array([np.inf, 0]))
    assert res4.isequal(true4)

def test_interval_dirac_derivative():
    # Derivative n=1

    # Case 1: 0 not in interval
    I1 = Interval(np.array([1, 2]), np.array([3, 4]))
    res1 = I1.dirac(1)
    true1 = Interval(np.array([0, 0]), np.array([0, 0]))
    assert res1.isequal(true1)

    # Case 2: 0 is in interval (interior)
    I2 = Interval(np.array([-1, 2]), np.array([1, 4]))
    res2 = I2.dirac(1)
    true2 = Interval(np.array([-np.inf, 0]), np.array([np.inf, 0]))
    assert res2.isequal(true2)

    # Case 3: 0 is at the boundary (max)
    I3 = Interval(np.array([-2, -3]), np.array([0, -1]))
    res3 = I3.dirac(1)
    true3 = Interval(np.array([0, 0]), np.array([np.inf, 0]))
    assert res3.isequal(true3)

    # Case 4: 0 is at the boundary (min)
    I4 = Interval(np.array([0, 2]), np.array([1, 4]))
    res4 = I4.dirac(1)
    true4 = Interval(np.array([-np.inf, 0]), np.array([0, 0]))
    assert res4.isequal(true4)
    
    # Derivative n > 1
    I5 = Interval(np.array([-1, 2]), np.array([1, 4]))
    res5 = I5.dirac(2)
    true5 = Interval(np.array([-np.inf, 0]), np.array([np.inf, 0]))
    assert res5.isequal(true5) 