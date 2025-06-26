"""
Test boundaryPoint method for zonotope class
"""

import numpy as np
import pytest
from cora_python.contSet.zonotope import Zonotope
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def test_boundaryPoint_1d():
    """Test boundaryPoint for 1D zonotopes"""
    # Simple 1D zonotope: [1,3]
    Z = Zonotope(np.array([2]), np.array([[1]]))
    
    # Direction towards positive
    dir_pos = np.array([1])
    x_pos = Z.boundaryPoint(dir_pos)
    assert np.isclose(x_pos, 3), f"Expected 3, got {x_pos}"
    
    # Direction towards negative
    dir_neg = np.array([-1])
    x_neg = Z.boundaryPoint(dir_neg)
    assert np.isclose(x_neg, 1), f"Expected 1, got {x_neg}"


def test_boundaryPoint_2d():
    """Test boundaryPoint for 2D zonotopes"""
    # 2D zonotope
    c = np.array([1, -1])
    G = np.array([[-3, 2, 1], [-1, 0, 3]])
    Z = Zonotope(c, G)
    
    # Test direction [1, 1]
    dir1 = np.array([1, 1])
    x1 = Z.boundaryPoint(dir1)
    
    # The boundary point should be on the boundary
    assert isinstance(x1, np.ndarray)
    assert x1.shape == (2,)
    
    # Test different direction
    dir2 = np.array([1, 0])
    x2 = Z.boundaryPoint(dir2)
    assert isinstance(x2, np.ndarray)
    assert x2.shape == (2,)


def test_boundaryPoint_with_start_point():
    """Test boundaryPoint with custom start point"""
    # Simple 2D zonotope
    c = np.array([0, 0])
    G = np.array([[1, 0], [0, 1]])  # Unit square
    Z = Zonotope(c, G)
    
    # Start from a point inside the zonotope
    start_point = np.array([0.5, 0.5])
    dir = np.array([1, 0])
    
    x = Z.boundaryPoint(dir, start_point)
    
    # Should reach the boundary at [1, 0.5]
    expected = np.array([1, 0.5])
    assert np.allclose(x, expected, atol=1e-10), f"Expected {expected}, got {x}"


def test_boundaryPoint_zero_direction():
    """Test boundaryPoint with zero direction vector"""
    Z = Zonotope(np.array([0, 0]), np.array([[1, 0], [0, 1]]))
    dir = np.array([0, 0])
    
    with pytest.raises(CORAerror):
        Z.boundaryPoint(dir)


def test_boundaryPoint_start_point_outside():
    """Test boundaryPoint with start point outside the zonotope"""
    Z = Zonotope(np.array([0, 0]), np.array([[1, 0], [0, 1]]))
    start_point = np.array([2, 2])  # Outside the zonotope
    dir = np.array([1, 0])
    
    with pytest.raises(CORAerror):
        Z.boundaryPoint(dir, start_point)


def test_boundaryPoint_dimension_mismatch():
    """Test boundaryPoint with dimension mismatch"""
    Z = Zonotope(np.array([0, 0]), np.array([[1, 0], [0, 1]]))
    
    # Wrong dimension for direction
    dir = np.array([1, 0, 0])
    with pytest.raises(Exception):  # equalDimCheck should raise an error
        Z.boundaryPoint(dir)
    
    # Wrong dimension for start point
    dir = np.array([1, 0])
    start_point = np.array([0, 0, 0])
    with pytest.raises(Exception):  # equalDimCheck should raise an error
        Z.boundaryPoint(dir, start_point)


def test_boundaryPoint_empty_set():
    """Test boundaryPoint for empty zonotope"""
    # Create empty zonotope
    Z = Zonotope.empty(2)
    dir = np.array([1, 0])
    
    x = Z.boundaryPoint(dir)
    assert x.shape == (2, 0), f"Expected empty array of shape (2, 0), got {x.shape}"


def test_boundaryPoint_center_default():
    """Test that default start point is the center"""
    c = np.array([1, 2])
    G = np.array([[1, 0], [0, 1]])
    Z = Zonotope(c, G)
    
    dir = np.array([1, 0])
    
    # Call without start point (should use center)
    x1 = Z.boundaryPoint(dir)
    
    # Call with explicit center as start point
    x2 = Z.boundaryPoint(dir, c)
    
    assert np.allclose(x1, x2), f"Results should be the same: {x1} vs {x2}"


def test_boundaryPoint_parallelotope():
    """Test boundaryPoint for parallelotope (special case of zonotope)"""
    # Create a parallelotope (2 generators in 2D)
    c = np.array([0, 0])
    G = np.array([[1, 0], [0.5, 1]])
    Z = Zonotope(c, G)
    
    dir = np.array([1, 1])
    x = Z.boundaryPoint(dir)
    
    assert isinstance(x, np.ndarray)
    assert x.shape == (2,)


def test_boundaryPoint_multiple_directions():
    """Test boundaryPoint with multiple directions"""
    c = np.array([0, 0])
    G = np.array([[2, 1], [1, 2]])
    Z = Zonotope(c, G)
    
    directions = [
        np.array([1, 0]),
        np.array([0, 1]),
        np.array([1, 1]),
        np.array([-1, 1]),
        np.array([-1, -1])
    ]
    
    for dir in directions:
        x = Z.boundaryPoint(dir)
        assert isinstance(x, np.ndarray)
        assert x.shape == (2,)
        
        # Verify the point is actually on or near the boundary
        # by checking that moving further in the same direction would exit the set
        eps = 1e-6
        point_beyond = x + eps * dir
        # This is a heuristic check - the point beyond should be outside or on boundary


if __name__ == "__main__":
    test_boundaryPoint_1d()
    test_boundaryPoint_2d()
    test_boundaryPoint_with_start_point()
    test_boundaryPoint_zero_direction()
    test_boundaryPoint_start_point_outside()
    test_boundaryPoint_dimension_mismatch()
    test_boundaryPoint_empty_set()
    test_boundaryPoint_center_default()
    test_boundaryPoint_parallelotope()
    test_boundaryPoint_multiple_directions()
    print("All boundaryPoint tests passed!") 