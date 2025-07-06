import pytest
import numpy as np
from cora_python.contSet.interval.interval import Interval

def test_interval_conPolyZono():
    # Example from MATLAB documentation
    I = Interval(np.array([[-2], [1]]), np.array([[2], [4]]))
    
    # This call will fail until the dependency chain is complete
    cpz = I.conPolyZono()
    
    # Basic check to ensure it returns an object of the expected type
    # This part of the test will only run if the above call succeeds
    from cora_python.contSet.conPolyZono.conPolyZono import ConPolyZono
    assert isinstance(cpz, ConPolyZono) 