import pytest
import numpy as np
from cora_python.contSet.interval.interval import Interval

def test_interval_projectHighDim():
    # check default case
    I = Interval(np.array([[2], [3]]), np.array([[4], [5]]))
    R = I.projectHighDim(5, [1, 4])
    Rtrue = Interval(np.array([[2], [0], [0], [3], [0]]), np.array([[4], [0], [0], [5], [0]]))
    assert R.isequal(Rtrue)

    # empty cannot be projected
    I_empty = Interval.empty(1)
    R_empty = I_empty.projectHighDim(5, [1])
    assert R_empty.is_empty() 