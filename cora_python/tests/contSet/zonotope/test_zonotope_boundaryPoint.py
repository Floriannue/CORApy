"""
Test boundaryPoint method for zonotope class
"""

import numpy as np
import pytest
from cora_python.contSet.zonotope import Zonotope
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol


def test_boundaryPoint_1d():
    """Test boundaryPoint for 1D zonotopes - matches MATLAB test"""
    # 1D zonotope: matches MATLAB test case
    Z = Zonotope(np.array([2]), np.array([[1, -0.5, 1]]))
    
    # positive direction
    dir_pos = np.array([0.5])
    x_pos = Z.boundaryPoint(dir_pos)
    x_true = 4.5
    assert withinTol(x_pos, x_true), f"Expected {x_true}, got {x_pos}"
    
    # negative direction
    dir_neg = np.array([-5])
    x_neg = Z.boundaryPoint(dir_neg)
    x_true = -0.5
    assert withinTol(x_neg, x_true), f"Expected {x_true}, got {x_neg}"
    
    # different start point
    start_point = np.array([3])
    x_start = Z.boundaryPoint(dir_pos, start_point)
    x_true = 4.5
    assert withinTol(x_start, x_true), f"Expected {x_true}, got {x_start}"


def test_boundaryPoint_2d_degenerate():
    """Test boundaryPoint for 2D degenerate zonotopes - matches MATLAB test"""
    # 2D degenerate zonotope: matches MATLAB test case
    c = np.array([1, -1])
    G = np.array([[2, 4, -1], [1, 2, -0.5]])
    Z = Zonotope(c, G)
    
    # Test direction [2, 1]
    dir1 = np.array([2, 1])
    x1 = Z.boundaryPoint(dir1)
    x_true = np.array([8, 2.5])
    assert np.all(withinTol(x1, x_true)), f"Expected {x_true}, got {x1}"
    
    # Test direction [1, -2]
    dir2 = np.array([1, -2])
    x2 = Z.boundaryPoint(dir2)
    x_true = Z.c.flatten()  # Convert column vector to row vector
    assert np.all(withinTol(x2, x_true)), f"Expected {x_true}, got {x2}"
    
    # Test with different start point
    dir3 = np.array([1, -2])
    start_point = np.array([4.5, 0.75])
    x3 = Z.boundaryPoint(dir3, start_point)
    assert np.all(withinTol(x3, start_point)), f"Expected {start_point}, got {x3}"


def test_boundaryPoint_2d_non_degenerate():
    """Test boundaryPoint for 2D non-degenerate zonotopes - matches MATLAB test"""
    # 2D non-degenerate zonotope: matches MATLAB test case
    c = np.array([1, -1])
    G = np.array([[-3, 2, 1], [-1, 0, 3]])
    Z = Zonotope(c, G)
    
    # Test direction [1, 1]
    dir1 = np.array([1, 1])
    x1 = Z.boundaryPoint(dir1)
    x_true = np.array([5, 3])
    assert np.all(withinTol(x1, x_true)), f"Expected {x_true}, got {x1}"
    
    # Test with different start point
    dir2 = np.array([1, 0])
    start_point = np.array([-2, 0])
    x2 = Z.boundaryPoint(dir2, start_point)
    x_true = np.array([6, 0])
    assert np.all(withinTol(x2, x_true)), f"Expected {x_true}, got {x2}"


def test_boundaryPoint_empty_set():
    """Test boundaryPoint for empty zonotope - matches MATLAB test"""
    # Create empty zonotope
    Z = Zonotope.empty(2)
    dir = np.array([1, 0])
    
    x = Z.boundaryPoint(dir)
    assert x.shape == (2, 0), f"Expected empty array of shape (2, 0), got {x.shape}"


def test_boundaryPoint_error_cases():
    """Test boundaryPoint error cases - matches MATLAB test"""
    Z = Zonotope(np.array([1, -1]), np.array([[-3, 2, 1], [-1, 0, 3]]))
    
    # all-zero direction
    dir_zero = np.array([0, 0])
    with pytest.raises(CORAerror, match="Vector has to be non-zero"):
        Z.boundaryPoint(dir_zero)
    
    # start point not in the set
    dir = np.array([1, 1])
    start_point_outside = np.array([-500, 100])
    with pytest.raises(CORAerror, match="Start point must be contained in the set"):
        Z.boundaryPoint(dir, start_point_outside)
    
    # dimension mismatch for direction
    dir_wrong_dim = np.array([1, 1, 1])
    with pytest.raises(CORAerror, match="Dimension mismatch"):
        Z.boundaryPoint(dir_wrong_dim, np.array([-5, 10]))
    
    # dimension mismatch for start point
    dir = np.array([1, 1])
    start_point_wrong_dim = np.array([0, 1, 0])
    with pytest.raises(CORAerror, match="Dimension mismatch"):
        Z.boundaryPoint(dir, start_point_wrong_dim)


if __name__ == "__main__":
    test_boundaryPoint_1d()
    test_boundaryPoint_2d_degenerate()
    test_boundaryPoint_2d_non_degenerate()
    test_boundaryPoint_empty_set()
    test_boundaryPoint_error_cases()
    print("All boundaryPoint tests passed!") 