import pytest
import numpy as np
from cora_python.contSet import Interval

def test_subsref():
    # Test scalar access
    i = Interval([-1, 1], [1, 2])
    i_sub = i[0]
    assert isinstance(i_sub, Interval)
    assert np.array_equal(i_sub.inf, np.array([-1.]))
    assert np.array_equal(i_sub.sup, np.array([1.]))

    # Test slice access
    i = Interval(np.array([[-1, 0], [1, 2]]), np.array([[1, 2], [3, 4]]))
    i_sub = i[0, :]
    assert isinstance(i_sub, Interval)
    assert np.array_equal(i_sub.inf, np.array([-1., 0.]))
    assert np.array_equal(i_sub.sup, np.array([1., 2.]))
    
    # Test property access
    assert np.array_equal(i.inf, np.array([[-1, 0], [1, 2]]))
    assert np.array_equal(i.sup, np.array([[1, 2], [3, 4]])) 