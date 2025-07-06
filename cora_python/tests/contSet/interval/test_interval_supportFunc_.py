import numpy as np
import pytest
from cora_python.contSet import Interval

def test_supportFunc_():
    # empty set
    I = Interval.empty(1)
    dir = np.array([1])
    val, x = I.supportFunc_(dir, 'upper')
    assert val == -np.inf and len(x) == 0
    val, x = I.supportFunc_(dir, 'lower')
    assert val == np.inf and len(x) == 0

    # 2D set
    I = Interval(np.array([-2, -1]), np.array([3, 4]))
    dir = np.array([2, -1])
    val, x = I.supportFunc_(dir, 'upper')
    assert val == 7 and np.all(x == np.array([3, -1]))
    val, x = I.supportFunc_(dir, 'lower')
    assert val == -8 and np.all(x == np.array([-2, 4]))
    val, x = I.supportFunc_(dir, 'range')
    assert val.isequal(Interval(-8, 7)) and np.all(x == np.array([[-2, 3], [4, -1]]))

    # unbounded set
    I = Interval(np.array([2, -np.inf, -1]), np.array([np.inf, 4, 2]))
    dir = np.array([-2, 1, -1])
    val, x = I.supportFunc_(dir, 'upper')
    assert val == 1 and np.all(x == np.array([2, 4, -1]))
    dir = np.array([1, -2, 1])
    val, x = I.supportFunc_(dir, 'upper')
    assert np.isinf(val) and np.all(x == np.array([np.inf, -np.inf, 2])) 