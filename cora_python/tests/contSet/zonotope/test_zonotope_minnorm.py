"""
test_zonotope_minnorm - unit test function of minnorm

Authors:       Victor Gassmann (MATLAB)
               Python translation by AI Assistant
Written:       15-October-2019 (MATLAB)
Python translation: 2025
"""

import numpy as np
import pytest
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.zonotope.minnorm import minnorm
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol


def test_zonotope_minnorm():
    """Test basic minnorm functionality"""
    
    # Example from MATLAB documentation
    c = np.array([[2], [1]])
    G = np.array([[1, -1, 0], [1, 2, 3]])
    Z = Zonotope(c, G)
    
    val, x = minnorm(Z)
    
    # Check that val is a positive number
    assert isinstance(val, (float, np.floating))
    assert val >= 0
    
    # Check that x is a point
    assert isinstance(x, np.ndarray)
    assert x.shape == (2, 1)
    
    # Check that the distance from x to center equals val
    distance = np.linalg.norm(x - Z.center())
    assert withinTol(distance, val, 1e-10)


def test_zonotope_minnorm_simple_cases():
    """Test minnorm for simple cases"""
    
    # Test with unit box in 2D
    Z_box = Zonotope(np.zeros((2, 1)), np.eye(2))
    val, x = minnorm(Z_box)
    
    # For a unit box centered at origin, minimum norm should be 1
    # (distance from origin to any face)
    assert withinTol(val, 1.0, 1e-10)
    
    # Test with a line segment
    Z_line = Zonotope(np.array([[1], [0]]), np.array([[1], [0]]))
    val, x = minnorm(Z_line)
    
    # Should be distance from origin to closest point on line
    assert val >= 0
    assert isinstance(x, np.ndarray)


def test_zonotope_minnorm_properties():
    """Test mathematical properties of minnorm"""
    
    # Test that minnorm gives the minimum distance to the zonotope boundary
    np.random.seed(42)  # For reproducibility
    
    for dim in [2, 3]:
        for ngens in [dim, dim + 2]:
            # Create random zonotope
            c = np.random.randn(dim, 1)
            G = np.random.randn(dim, ngens)
            Z = Zonotope(c, G)
            
            val, x = minnorm(Z)
            
            # Check that val is the distance from center to x
            distance = np.linalg.norm(x - Z.center())
            assert withinTol(distance, val, 1e-10)
            
            # Check that x should be contained in the zonotope
            # (or at least on its boundary)
            assert Z.contains_(x) or withinTol(Z.contains_(x), True, 1e-10)


def test_zonotope_minnorm_vs_support_function():
    """Test that minnorm is consistent with support function"""
    
    # This is a simplified version of the MATLAB test
    # We test that the minimum norm is indeed minimal by checking
    # against random directions
    
    np.random.seed(42)  # For reproducibility
    
    for dim in [2, 3]:
        for ngens in [dim + 1, dim + 3]:
            # Create random zonotope
            c = np.zeros((dim, 1))  # Center at origin for simplicity
            G = np.random.randn(dim, ngens)
            Z = Zonotope(c, G)
            
            val, x = minnorm(Z)
            
            # Generate random unit directions
            num_directions = 10
            for _ in range(num_directions):
                direction = np.random.randn(dim, 1)
                direction = direction / np.linalg.norm(direction)
                
                # Compute support function (simplified approximation)
                # For a zonotope Z = c + G*[-1,1]^m, the support function
                # in direction l is l^T*c + ||G^T*l||_1
                sF = direction.T @ Z.c + np.sum(np.abs(Z.G.T @ direction))
                
                # The minimum norm should be <= support function value
                # (with some tolerance for numerical errors)
                assert val <= sF + 1e-10 or withinTol(val, sF, 1e-10)


def test_zonotope_minnorm_edge_cases():
    """Test minnorm for edge cases"""
    
    # Test with point zonotope (no generators)
    Z_point = Zonotope(np.array([[1], [2]]))
    val, x = minnorm(Z_point)
    
    # For a point, minnorm should be distance from origin to the point
    expected_val = np.linalg.norm(np.array([[1], [2]]))
    assert withinTol(val, expected_val, 1e-10)
    assert np.allclose(x, np.array([[1], [2]]))
    
    # Test with zonotope centered at origin
    Z_origin = Zonotope(np.zeros((2, 1)), np.array([[1, 0], [0, 1]]))
    val, x = minnorm(Z_origin)
    
    # Should give minimum distance to boundary
    assert val > 0
    assert isinstance(x, np.ndarray)


def test_zonotope_minnorm_dimensional_consistency():
    """Test that minnorm works correctly for different dimensions"""
    
    # Test 1D case
    Z_1d = Zonotope(np.array([[2]]), np.array([[1]]))
    val, x = minnorm(Z_1d)
    assert x.shape == (1, 1)
    assert val >= 0
    
    # Test higher dimensions
    for dim in [2, 3, 4]:
        Z = Zonotope(np.ones((dim, 1)), np.eye(dim))
        val, x = minnorm(Z)
        assert x.shape == (dim, 1)
        assert val >= 0


if __name__ == '__main__':
    test_zonotope_minnorm()
    test_zonotope_minnorm_simple_cases()
    test_zonotope_minnorm_properties()
    test_zonotope_minnorm_vs_support_function()
    test_zonotope_minnorm_edge_cases()
    test_zonotope_minnorm_dimensional_consistency()
    print("All minnorm tests passed!") 