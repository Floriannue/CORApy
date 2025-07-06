import pytest
import numpy as np
from cora_python.contSet.interval.interval import Interval
from cora_python.contSet.zonotope.zonotope import Zonotope

def test_interval_quadMap():
    # Example from MATLAB documentation
    Z = Zonotope(np.array([[0], [0]]), np.array([[1, 1], [1, 0]]))
    I = Interval(Z)
    
    Q = [
        np.array([[0.5, 0.5], [0, -0.5]]),
        np.array([[-1, 0], [1, 1]])
    ]

    # Call the interval quadMap
    res_interval = I.quadMap(Q)

    # Verify the result is an interval
    assert isinstance(res_interval, Interval)
    
    # Verify the result has the expected dimensions
    assert res_interval.dim() == 2
    
    # Verify the result is bounded (not infinite)
    assert np.all(np.isfinite(res_interval.inf))
    assert np.all(np.isfinite(res_interval.sup))
    
    # Verify inf <= sup
    assert np.all(res_interval.inf <= res_interval.sup)
    
    # Test with simple case: point interval
    I_point = Interval(np.array([1, 2]), np.array([1, 2]))
    res_point = I_point.quadMap(Q)
    
    # For a point interval, the result should be deterministic
    expected_point_result = np.array([
        [1, 2] @ Q[0] @ np.array([1, 2]),  # x^T Q[0] x
        [1, 2] @ Q[1] @ np.array([1, 2])   # x^T Q[1] x  
    ])
    
    # The result should contain this point
    assert np.all(res_point.inf <= expected_point_result)
    assert np.all(expected_point_result <= res_point.sup) 