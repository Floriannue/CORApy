import pytest
import numpy as np
from cora_python.contSet.interval.interval import Interval

def test_interval_ellipsoid():
    
    I = Interval(np.array([[-1], [-2], [1]]), np.array([[3], [1], [4]]))
    
    # check outer approximation
    E_outer = I.ellipsoid('outer')
    assert E_outer.contains(I)

    # check inner approximation
    E_inner = I.ellipsoid('inner')
    assert I.contains(E_inner) 