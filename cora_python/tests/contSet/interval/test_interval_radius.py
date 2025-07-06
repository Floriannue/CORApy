import pytest
import numpy as np

from cora_python.contSet import Interval

def test_radius():
    # Example from MATLAB file
    # I = interval([-2;1],[4;3]);
    # r = radius(I);
    
    inf = np.array([[-2], [1]])
    sup = np.array([[4], [3]])
    i = Interval(inf, sup)
    
    r = i.radius()
    
    # MATLAB calculation:
    # rad = (sup - inf) / 2 = ([6; 2]) / 2 = [3; 1]
    # r = sqrt(sum(rad.^2)) = sqrt(3^2 + 1^2) = sqrt(10)
    
    assert np.isclose(r, np.sqrt(10))

def test_radius_scalar():
    i = Interval(-5, 5)
    r = i.radius()
    # rad = (5 - (-5))/2 = 5
    # r = sqrt(5^2) = 5
    assert np.isclose(r, 5.0)

def test_radius_zero_volume():
    i = Interval(np.array([1, 2]), np.array([1, 2]))
    r = i.radius()
    assert np.isclose(r, 0.0) 