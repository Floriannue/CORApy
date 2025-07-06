import pytest
import numpy as np
from cora_python.contSet.interval.interval import Interval

def test_interval_intervalMatrix():
    # Example from MATLAB documentation
    I = Interval(np.array([[2, 1], [-3, 4]]), np.array([[3, 2], [1, 5]]))
    
    # This call will fail until the dependency is complete
    im = I.intervalMatrix()
    
    # Basic check to ensure it returns an object of the expected type
    from cora_python.matrixSet.intervalMatrix.intervalMatrix import IntervalMatrix
    assert isinstance(im, IntervalMatrix)
    
    # Check if the center and radius are correctly passed
    assert np.all(im.center == I.center())
    assert np.all(im.rad == I.rad()) 