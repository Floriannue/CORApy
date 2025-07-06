import pytest
import numpy as np
from cora_python.contSet.interval.interval import Interval
from cora_python.contSet.polyZonotope.polyZonotope import PolyZonotope

def test_interval_polyZonotope():
    # Example from MATLAB documentation
    I = Interval(np.array([[1], [2]]), np.array([[3], [5]]))
    pZ = I.polyZonotope()

    assert isinstance(pZ, PolyZonotope)
    
    # Expected properties
    expected_c = np.array([[2], [3.5]])
    expected_G = np.array([[1, 0], [0, 1.5]])
    expected_E = np.array([[1, 0], [0, 1]])
    
    assert np.allclose(pZ.c, expected_c)
    assert np.allclose(pZ.G, expected_G)
    assert np.allclose(pZ.E, expected_E)
    # The dependent generator matrix should be empty
    assert pZ.Grest.size == 0 