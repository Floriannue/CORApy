import pytest
import numpy as np
from cora_python.contSet.interval.interval import Interval

def test_interval_conZonotope():
    # Example from MATLAB documentation
    I = Interval(np.array([[1], [-1]]), np.array([[2], [1]]))
    
    # This call will fail until the dependency chain is complete
    cz = I.conZonotope()
    
    # Basic check to ensure it returns an object of the expected type
    from cora_python.contSet.conZonotope.conZonotope import ConZonotope
    assert isinstance(cz, ConZonotope) 