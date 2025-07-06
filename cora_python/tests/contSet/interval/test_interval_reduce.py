import pytest
import numpy as np

from cora_python.contSet import Interval

def test_reduce():
    i1 = Interval(np.array([-1, 1]), np.array([2, 3]))
    i2 = i1.reduce()
    
    # Check if it returns the same object
    assert i1 is i2
    
    # Check if the values are unchanged
    assert np.array_equal(i2.inf, i1.inf)
    assert np.array_equal(i2.sup, i1.sup) 