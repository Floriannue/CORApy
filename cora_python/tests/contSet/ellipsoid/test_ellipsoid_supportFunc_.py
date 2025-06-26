"""
test_ellipsoid_supportFunc - unit test function of supportFunc

This module tests the ellipsoid supportFunc implementation exactly matching MATLAB.

Authors:       Victor Gassmann (MATLAB), Python translation by AI Assistant
Written:       27-July-2021 (MATLAB)
Python translation: 2025
"""

import numpy as np
import pytest
from cora_python.contSet.ellipsoid import Ellipsoid
from cora_python.contSet.interval import Interval
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol


def test_ellipsoid_supportFunc():
    """Main supportFunc test matching MATLAB test_ellipsoid_supportFunc"""
    
    # Init cases (exact MATLAB values)
    E1 = Ellipsoid(
        np.array([[5.4387811500952807, 12.4977183618314545], 
                  [12.4977183618314545, 29.6662117284481646]]), 
        np.array([[-0.7445068341257537], [3.5800647524843665]]),
        0.000001
    )
    Ed1 = Ellipsoid(
        np.array([[4.2533342807136076, 0.6346400221575308], 
                  [0.6346400221575309, 0.0946946398147988]]), 
        np.array([[-2.4653656883489115], [0.2717868749873985]]),
        0.000001
    )
    E0 = Ellipsoid(
        np.array([[0.0000000000000000, 0.0000000000000000], 
                  [0.0000000000000000, 0.0000000000000000]]), 
        np.array([[1.0986933635979599], [-1.9884387759871638]]),
        0.000001
    )
    
    # Empty set
    E_empty = Ellipsoid.empty(2)
    val_upper_empty = E_empty.supportFunc_(np.array([[1], [1]]), 'upper')[0]
    val_lower_empty = E_empty.supportFunc_(np.array([[1], [1]]), 'lower')[0]
    
    assert val_upper_empty == -np.inf, f"Empty set upper support should be -inf, got {val_upper_empty}"
    assert val_lower_empty == np.inf, f"Empty set lower support should be inf, got {val_lower_empty}"
    
    # Check support function for all test ellipsoids
    assert aux_checkSuppFunc(E1), "E1 support function check failed"
    assert aux_checkSuppFunc(Ed1), "Ed1 support function check failed"
    assert aux_checkSuppFunc(E0), "E0 support function check failed"
    
    # Check type = 'range'
    E = Ellipsoid(np.array([[5, 7], [7, 13]]), np.array([[1], [2]]))
    direction = np.array([[1], [1]])
    
    val_upper, x_upper = E.supportFunc_(direction, 'upper')
    val_lower, x_lower = E.supportFunc_(direction, 'lower')
    val_int, x_both = E.supportFunc_(direction, 'range')
    
    # Check interval result
    expected_interval = Interval(val_lower, val_upper)
    assert np.isclose(val_int.infimum, expected_interval.infimum) and \
           np.isclose(val_int.supremum, expected_interval.supremum), \
           f"Range result should match interval({val_lower}, {val_upper})"
    
    # Check support vectors
    assert np.allclose(np.hstack([x_upper, x_lower]), x_both), \
           "Range support vectors should match individual vectors"


def aux_checkSuppFunc(E):
    """
    Auxiliary function to check support function implementation
    
    This matches the MATLAB aux_checkSuppFunc exactly
    """
    n = E.dim()
    T, S, _ = np.linalg.svd(E.Q)
    s = np.sqrt(np.diag(S))
    
    # Loop over all directions
    for i in range(n):
        # Direction (principal axis)
        l = T[:, i:i+1]  # Keep as column vector
        
        # Evaluate support function and compute support vector
        val, x = E.supportFunc_(l, 'upper')
        ri = abs(val - (l.T @ E.q)[0, 0])
        
        # Check results
        assert withinTol(s[i], ri, E.TOL), \
            f"Support function value mismatch for direction {i}: expected {s[i]}, got {ri}"
        assert withinTol(np.linalg.norm(x - E.q), s[i], E.TOL), \
            f"Support vector distance mismatch for direction {i}: expected {s[i]}, got {np.linalg.norm(x - E.q)}"
    
    return True


def test_supportFunc_point_ellipsoid():
    """Test support function for point ellipsoid (degenerate case)"""
    
    # Point ellipsoid
    E_point = Ellipsoid(np.zeros((2, 2)), np.array([[1], [2]]))
    direction = np.array([[1], [0]])
    
    val, x = E_point.supportFunc_(direction, 'upper')
    
    # For point ellipsoid, support vector should be the center
    assert np.allclose(x, E_point.q), "Support vector for point ellipsoid should be the center"
    assert np.isclose(val, direction.T @ E_point.q), "Support value should be direction dot center"


def test_supportFunc_various_directions():
    """Test support function in various directions"""
    
    # Unit ellipsoid
    E = Ellipsoid(np.eye(2))
    
    # Test cardinal directions
    directions = [
        np.array([[1], [0]]),   # +x direction
        np.array([[-1], [0]]),  # -x direction  
        np.array([[0], [1]]),   # +y direction
        np.array([[0], [-1]]),  # -y direction
    ]
    
    expected_vals = [1, -1, 1, -1]  # For unit ellipsoid
    
    for i, direction in enumerate(directions):
        val, x = E.supportFunc_(direction, 'upper')
        assert np.isclose(val, expected_vals[i]), \
            f"Support value in direction {direction.flatten()} should be {expected_vals[i]}, got {val}"


def test_supportFunc_consistency():
    """Test consistency between upper and lower support functions"""
    
    E = Ellipsoid(np.array([[2, 1], [1, 3]]), np.array([[1], [-1]]))
    direction = np.array([[1], [1]])
    
    val_upper, x_upper = E.supportFunc_(direction, 'upper')
    val_lower, x_lower = E.supportFunc_(-direction, 'upper')  # Flip direction for lower
    
    # Check that -direction upper equals direction lower
    val_lower_direct, x_lower_direct = E.supportFunc_(direction, 'lower')
    
    assert np.isclose(val_lower_direct, -val_lower), \
        "Lower support should equal negative of flipped direction upper support"


def test_supportFunc_edge_cases():
    """Test edge cases for support function"""
    
    # Very small ellipsoid
    E_small = Ellipsoid(1e-10 * np.eye(2))
    direction = np.array([[1], [0]])
    
    val, x = E_small.supportFunc_(direction, 'upper')
    assert np.isclose(val, np.sqrt(1e-10)), "Small ellipsoid support function failed"
    
    # Very elongated ellipsoid  
    E_elongated = Ellipsoid(np.diag([100, 0.01]))
    direction = np.array([[1], [0]])
    
    val, x = E_elongated.supportFunc_(direction, 'upper')
    assert np.isclose(val, 10), "Elongated ellipsoid support function failed"


@pytest.mark.parametrize("direction_type", ['upper', 'lower', 'range'])
def test_supportFunc_return_types(direction_type):
    """Test that support function returns correct types for different modes"""
    
    E = Ellipsoid(np.eye(2))
    direction = np.array([[1], [1]])
    
    result = E.supportFunc_(direction, direction_type)
    
    if direction_type == 'range':
        val, x = result
        assert isinstance(val, Interval), "Range mode should return Interval"
        assert x.shape[1] == 2, "Range mode should return 2 support vectors"
    else:
        val, x = result
        assert isinstance(val, (int, float, np.number)), "Upper/lower mode should return scalar"
        assert x.shape[1] == 1, "Upper/lower mode should return 1 support vector"


if __name__ == "__main__":
    test_ellipsoid_supportFunc()
    test_supportFunc_point_ellipsoid() 
    test_supportFunc_various_directions()
    test_supportFunc_consistency()
    test_supportFunc_edge_cases()
    print("All supportFunc tests passed!") 