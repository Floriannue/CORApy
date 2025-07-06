import pytest
import numpy as np
from cora_python.contSet import Interval

def test_subsasgn():
    # Test scalar assignment
    i = Interval([-1, 1], [1, 2])
    i[0] = Interval(5, 6)
    assert np.array_equal(i.inf, np.array([5., 1.]))
    assert np.array_equal(i.sup, np.array([6., 2.]))
    
    # Test scalar assignment with number
    i[0] = 10
    assert np.array_equal(i.inf, np.array([10., 1.]))
    assert np.array_equal(i.sup, np.array([10., 2.]))

    # Test slice assignment
    i = Interval(np.array([[-1, 0], [1, 2]]), np.array([[1, 2], [3, 4]]))
    i[0, :] = Interval([-5, -6], [5, 6])
    assert np.array_equal(i.inf, np.array([[-5., -6.], [1, 2]]))
    assert np.array_equal(i.sup, np.array([[5., 6.], [3, 4]]))

    # Test slice assignment with number
    i[0, :] = 100
    assert np.array_equal(i.inf, np.array([[100., 100.], [1, 2]]))
    assert np.array_equal(i.sup, np.array([[100., 100.], [3, 4]])) 