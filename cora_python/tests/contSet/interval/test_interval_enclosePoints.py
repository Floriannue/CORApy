"""
test_interval_enclosePoints - unit test function for Interval.enclosePoints

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
"""

import numpy as np
import pytest
from cora_python.contSet.interval.interval import Interval


def test_interval_enclosePoints():
    """Test enclosePoints method for Interval."""
    
    # Point cloud test case
    pts = np.array([
        [1, 4, 2, 5, 3, 2, 4, 3, 2, 5],
        [6, 8, 9, 7, 6, 9, 8, 6, 8, 7]
    ])
    I = Interval.enclosePoints(pts)
    I_true = Interval([1, 6], [5, 9])
    assert I.isequal(I_true)
    
    # Unbounded case with inf values
    pts_unbounded = np.array([
        [-np.inf, -2, 1, 5],
        [2, 3, 4, -1],
        [1, 3, 1, np.inf]
    ])
    I_unbounded = Interval.enclosePoints(pts_unbounded)
    I_true_unbounded = Interval([-np.inf, -1, 1], [5, 4, np.inf])
    assert I_unbounded.isequal(I_true_unbounded)
    
    # Edge case: single point
    single_pt = np.array([[2], [3]])
    I_single = Interval.enclosePoints(single_pt)
    I_single_true = Interval([2, 3], [2, 3])
    assert I_single.isequal(I_single_true)
    
    # Edge case: 1D points (should be converted to column vector)
    pts_1d = np.array([1, 4, 2, 5, 3])
    I_1d = Interval.enclosePoints(pts_1d)
    I_1d_true = Interval([1], [5])
    assert I_1d.isequal(I_1d_true)


def test_interval_enclosePoints_errors():
    """Test error cases for enclosePoints method."""
    
    # Empty point cloud should raise error
    with pytest.raises(ValueError):
        Interval.enclosePoints(np.array([]))
        
    with pytest.raises(ValueError):
        Interval.enclosePoints(np.array([]).reshape(0, 0))


if __name__ == "__main__":
    test_interval_enclosePoints()
    test_interval_enclosePoints_errors()
    print("All tests passed!") 