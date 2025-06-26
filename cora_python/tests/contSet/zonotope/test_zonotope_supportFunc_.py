"""
Test supportFunc_ method for zonotope class
"""

import numpy as np
import pytest
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.interval import Interval


def test_supportFunc_upper():
    """Test supportFunc_ with upper bound"""
    # Simple 2D zonotope
    c = np.array([1, 2])
    G = np.array([[1, 0], [0, 1]])
    Z = Zonotope(c, G)
    
    # Direction [1, 0]
    dir = np.array([1, 0])
    val, x, fac = Z.supportFunc_(dir, 'upper')
    
    # Expected: c[0] + sum(abs(G[0,:])) = 1 + 1 = 2
    assert np.isclose(val, 2), f"Expected 2, got {val}"
    
    # Check support vector
    expected_x = np.array([2, 2])
    assert np.allclose(x, expected_x), f"Expected {expected_x}, got {x}"
    
    # Check factors
    expected_fac = np.array([1, 0])
    assert np.allclose(fac, expected_fac), f"Expected {expected_fac}, got {fac}"


def test_supportFunc_lower():
    """Test supportFunc_ with lower bound"""
    # Simple 2D zonotope
    c = np.array([1, 2])
    G = np.array([[1, 0], [0, 1]])
    Z = Zonotope(c, G)
    
    # Direction [1, 0]
    dir = np.array([1, 0])
    val, x, fac = Z.supportFunc_(dir, 'lower')
    
    # Expected: c[0] - sum(abs(G[0,:])) = 1 - 1 = 0
    assert np.isclose(val, 0), f"Expected 0, got {val}"
    
    # Check support vector
    expected_x = np.array([0, 2])
    assert np.allclose(x, expected_x), f"Expected {expected_x}, got {x}"
    
    # Check factors
    expected_fac = np.array([-1, 0])
    assert np.allclose(fac, expected_fac), f"Expected {expected_fac}, got {fac}"


def test_supportFunc_range():
    """Test supportFunc_ with range"""
    # Simple 2D zonotope
    c = np.array([1, 2])
    G = np.array([[1, 0], [0, 1]])
    Z = Zonotope(c, G)
    
    # Direction [1, 0]
    dir = np.array([1, 0])
    val, x, fac = Z.supportFunc_(dir, 'range')
    
    # Expected: interval [0, 2]
    assert isinstance(val, Interval), f"Expected Interval, got {type(val)}"
    assert np.isclose(val.inf, 0), f"Expected lower bound 0, got {val.inf}"
    assert np.isclose(val.sup, 2), f"Expected upper bound 2, got {val.sup}"
    
    # Check support vectors (should be 2 columns)
    assert x.shape == (2, 2), f"Expected shape (2, 2), got {x.shape}"
    expected_x_lower = np.array([0, 2])
    expected_x_upper = np.array([2, 2])
    assert np.allclose(x[:, 0], expected_x_lower), f"Expected lower support vector {expected_x_lower}, got {x[:, 0]}"
    assert np.allclose(x[:, 1], expected_x_upper), f"Expected upper support vector {expected_x_upper}, got {x[:, 1]}"


def test_supportFunc_1d():
    """Test supportFunc_ for 1D zonotope"""
    # 1D zonotope: [1, 3]
    c = np.array([2])
    G = np.array([[1]])
    Z = Zonotope(c, G)
    
    # Direction [1]
    dir = np.array([1])
    
    # Upper bound
    val_upper, x_upper, fac_upper = Z.supportFunc_(dir, 'upper')
    assert np.isclose(val_upper, 3), f"Expected 3, got {val_upper}"
    assert np.allclose(x_upper, [3]), f"Expected [3], got {x_upper}"
    
    # Lower bound
    val_lower, x_lower, fac_lower = Z.supportFunc_(dir, 'lower')
    assert np.isclose(val_lower, 1), f"Expected 1, got {val_lower}"
    assert np.allclose(x_lower, [1]), f"Expected [1], got {x_lower}"


def test_supportFunc_multiple_generators():
    """Test supportFunc_ with multiple generators"""
    # Zonotope with 3 generators
    c = np.array([0, 0])
    G = np.array([[1, 2, -1], [0, 1, 2]])
    Z = Zonotope(c, G)
    
    # Direction [1, 1]
    dir = np.array([1, 1])
    val, x, fac = Z.supportFunc_(dir, 'upper')
    
    # Project: dir.T @ c = 0, dir.T @ G = [1, 3, 1]
    # Upper bound: 0 + sum(abs([1, 3, 1])) = 5
    assert np.isclose(val, 5), f"Expected 5, got {val}"
    
    # Factors should be signs of [1, 3, 1] = [1, 1, 1]
    expected_fac = np.array([1, 1, 1])
    assert np.allclose(fac, expected_fac), f"Expected {expected_fac}, got {fac}"


def test_supportFunc_negative_direction():
    """Test supportFunc_ with negative direction"""
    c = np.array([1, 1])
    G = np.array([[1, 0], [0, 1]])
    Z = Zonotope(c, G)
    
    # Direction [-1, -1]
    dir = np.array([-1, -1])
    val, x, fac = Z.supportFunc_(dir, 'upper')
    
    # Project: dir.T @ c = -2, dir.T @ G = [-1, -1]
    # Upper bound: -2 + sum(abs([-1, -1])) = -2 + 2 = 0
    assert np.isclose(val, 0), f"Expected 0, got {val}"


def test_supportFunc_empty_zonotope():
    """Test supportFunc_ for empty zonotope"""
    Z = Zonotope.empty(2)
    dir = np.array([1, 0])
    
    # Upper bound for empty set
    val, x, fac = Z.supportFunc_(dir, 'upper')
    assert val == -np.inf, f"Expected -inf, got {val}"
    assert x.size == 0, f"Expected empty array, got {x}"
    assert fac.size == 0, f"Expected empty array, got {fac}"
    
    # Lower bound for empty set
    val, x, fac = Z.supportFunc_(dir, 'lower')
    assert val == np.inf, f"Expected inf, got {val}"
    
    # Range for empty set
    val, x, fac = Z.supportFunc_(dir, 'range')
    assert isinstance(val, Interval), f"Expected Interval, got {type(val)}"
    assert val.inf == -np.inf and val.sup == np.inf


def test_supportFunc_zero_generators():
    """Test supportFunc_ with zero generators"""
    c = np.array([1, 2])
    G = np.array([[], []])  # No generators
    Z = Zonotope(c, G)
    
    dir = np.array([1, 0])
    val, x, fac = Z.supportFunc_(dir, 'upper')
    
    # With no generators, the support function is just the projection of center
    assert np.isclose(val, 1), f"Expected 1, got {val}"
    assert np.allclose(x, c), f"Expected {c}, got {x}"


def test_supportFunc_invalid_type():
    """Test supportFunc_ with invalid type"""
    Z = Zonotope(np.array([0, 0]), np.array([[1, 0], [0, 1]]))
    dir = np.array([1, 0])
    
    with pytest.raises(ValueError):
        Z.supportFunc_(dir, 'invalid')


def test_supportFunc_different_directions():
    """Test supportFunc_ with various directions"""
    c = np.array([0, 0])
    G = np.array([[1, 0], [0, 1]])
    Z = Zonotope(c, G)
    
    directions = [
        np.array([1, 0]),
        np.array([0, 1]),
        np.array([1, 1]),
        np.array([-1, 1]),
        np.array([1, -1])
    ]
    
    for dir in directions:
        val_upper, x_upper, fac_upper = Z.supportFunc_(dir, 'upper')
        val_lower, x_lower, fac_lower = Z.supportFunc_(dir, 'lower')
        
        # Upper bound should be >= lower bound
        assert val_upper >= val_lower, f"Upper bound {val_upper} should be >= lower bound {val_lower}"
        
        # Support vectors should have correct dimension
        assert x_upper.shape == (2,), f"Expected shape (2,), got {x_upper.shape}"
        assert x_lower.shape == (2,), f"Expected shape (2,), got {x_lower.shape}"


if __name__ == "__main__":
    test_supportFunc_upper()
    test_supportFunc_lower()
    test_supportFunc_range()
    test_supportFunc_1d()
    test_supportFunc_multiple_generators()
    test_supportFunc_negative_direction()
    test_supportFunc_empty_zonotope()
    test_supportFunc_zero_generators()
    test_supportFunc_invalid_type()
    test_supportFunc_different_directions()
    print("All supportFunc_ tests passed!") 