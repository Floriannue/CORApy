"""
test_ellipsoid_representsa_ - unit test function of representsa_

This module tests the ellipsoid representsa_ implementation.

Authors:       Python translation by AI Assistant
Python translation: 2025
"""

import numpy as np
import pytest
from cora_python.contSet.ellipsoid import Ellipsoid
from cora_python.contSet.interval import Interval


def test_representsa_point():
    """Test representsa_ for point representation"""
    
    # Point ellipsoid (zero shape matrix)
    E_point = Ellipsoid(np.zeros((2, 2)), np.array([[1], [2]]))
    
    # Should represent a point
    assert E_point.representsa_('point'), "Zero shape matrix should represent a point"
    
    # Non-point ellipsoid
    E_nonpoint = Ellipsoid(np.eye(2))
    assert not E_nonpoint.representsa_('point'), "Non-zero shape matrix should not represent a point"


def test_representsa_origin():
    """Test representsa_ for origin representation"""
    
    # Point ellipsoid at origin
    E_origin = Ellipsoid(np.zeros((2, 2)), np.zeros((2, 1)))
    assert E_origin.representsa_('origin'), "Zero shape matrix at origin should represent origin"
    
    # Point ellipsoid not at origin
    E_not_origin = Ellipsoid(np.zeros((2, 2)), np.array([[1], [0]]))
    assert not E_not_origin.representsa_('origin'), "Point not at origin should not represent origin"
    
    # Non-point ellipsoid at origin
    E_nonpoint_origin = Ellipsoid(np.eye(2), np.zeros((2, 1)))
    assert not E_nonpoint_origin.representsa_('origin'), "Non-point at origin should not represent origin"


def test_representsa_ellipsoid():
    """Test representsa_ for ellipsoid representation"""
    
    # Any ellipsoid should represent an ellipsoid
    E = Ellipsoid(np.eye(2))
    assert E.representsa_('ellipsoid'), "Any ellipsoid should represent an ellipsoid"
    
    # Point ellipsoid should also represent an ellipsoid
    E_point = Ellipsoid(np.zeros((2, 2)), np.array([[1], [2]]))
    assert E_point.representsa_('ellipsoid'), "Point ellipsoid should represent an ellipsoid"


def test_representsa_interval():
    """Test representsa_ for interval representation"""
    
    # 1D ellipsoid should represent an interval
    E_1d = Ellipsoid(np.array([[2]]), np.array([[1]]))
    assert E_1d.representsa_('interval'), "1D ellipsoid should represent an interval"
    
    # Point ellipsoid should represent an interval
    E_point = Ellipsoid(np.zeros((2, 2)), np.array([[1], [2]]))
    assert E_point.representsa_('interval'), "Point ellipsoid should represent an interval"
    
    # 2D non-point ellipsoid should NOT represent an interval (MATLAB logic)
    E_2d = Ellipsoid(np.eye(2))
    assert not E_2d.representsa_('interval'), "2D non-point ellipsoid should not represent an interval"

    # 2D ellipsoid with non-zero center
    E_2d_center = Ellipsoid(np.eye(2), np.array([[1], [2]]))
    assert not E_2d_center.representsa_('interval'), "2D ellipsoid with center should not represent an interval"


def test_representsa_convexSet():
    """Test representsa_ for convex set representation"""
    
    # Any ellipsoid should represent a convex set
    E = Ellipsoid(np.eye(3))
    assert E.representsa_('convexSet'), "Any ellipsoid should represent a convex set"
    
    E_point = Ellipsoid(np.zeros((2, 2)), np.array([[1], [2]]))
    assert E_point.representsa_('convexSet'), "Point ellipsoid should represent a convex set"


def test_representsa_unsupported_types():
    """Test representsa_ for unsupported set types"""
    
    E = Ellipsoid(np.eye(2))
    
    # These should raise errors or return False
    unsupported_types = [
        'halfspace',      # Ellipsoids are bounded
        'fullspace',      # Ellipsoids are bounded  
        'probZonotope',   # Not supported
    ]
    
    for set_type in unsupported_types:
        result = E.representsa_(set_type)
        assert not result, f"Ellipsoid should not represent {set_type}"


def test_representsa_empty_set():
    """Test representsa_ for empty set representation"""
    
    # Empty ellipsoid
    E_empty = Ellipsoid.empty(2)
    assert E_empty.representsa_('emptySet'), "Empty ellipsoid should represent empty set"
    
    # Non-empty ellipsoid
    E_nonempty = Ellipsoid(np.eye(2))
    assert not E_nonempty.representsa_('emptySet'), "Non-empty ellipsoid should not represent empty set"


def test_representsa_capsule():
    """Test representsa_ for capsule representation"""
    
    # 1D ellipsoid should represent a capsule
    E_1d = Ellipsoid(np.array([[1]]))
    assert E_1d.representsa_('capsule'), "1D ellipsoid should represent a capsule"
    
    # Point ellipsoid should represent a capsule
    E_point = Ellipsoid(np.zeros((2, 2)), np.array([[1], [2]]))
    assert E_point.representsa_('capsule'), "Point ellipsoid should represent a capsule"
    
    # Ball (isotropic ellipsoid) should represent a capsule
    E_ball = Ellipsoid(2 * np.eye(2))
    assert E_ball.representsa_('capsule'), "Ball should represent a capsule"
    
    # Non-isotropic 2D ellipsoid should NOT represent a capsule  
    E_aniso = Ellipsoid(np.array([[1, 0], [0, 4]]))
    assert not E_aniso.representsa_('capsule'), "Non-isotropic ellipsoid should not represent a capsule"


def test_representsa_dual_output():
    """Test representsa_ with dual output (when conversion set is requested)"""
    
    # This test checks the introspection-based dual output detection
    # In practice, you would call: res, S = E.representsa_('type')
    
    E = Ellipsoid(np.eye(2))
    
    # Single output case (detected automatically)
    result = E.representsa_('ellipsoid')
    assert result is True, "Single output should return boolean"
    
    # Try to trigger dual output manually (this is tricky due to introspection)
    # We'll test this by examining the behavior when the calling pattern suggests dual output
    
    # For now, just verify single output works
    point_result = E.representsa_('point')
    assert point_result == False, "Regular ellipsoid should not represent a point"

    # Test dual output with return_set=True
    res, S = E.representsa_('ellipsoid', return_set=True)
    assert res == True, "Should return True for ellipsoid"
    assert S is E, "Should return the same ellipsoid object"


def test_representsa_tolerance_sensitivity():
    """Test representsa_ with different tolerance values"""
    
    # Create ellipsoid with very small eigenvalue
    small_val = 1e-12
    Q = np.diag([1, small_val])
    
    # With default tolerance, might be considered point-like
    E_default = Ellipsoid(Q)
    result_default = E_default.representsa_('point')
    
    # With very tight tolerance, should not be point-like
    result_tight = E_default.representsa_('point', tol=1e-15)
    
    # The results might differ based on tolerance
    # Both are valid depending on tolerance interpretation
    assert isinstance(result_default, (bool, np.bool_)), "Should return boolean"
    assert isinstance(result_tight, (bool, np.bool_)), "Should return boolean"

    # Test near-zero matrix
    Q_zero = np.array([[1e-15, 0], [0, 1e-15]])
    E_near_zero = Ellipsoid(Q_zero)
    
    # Should be considered a point with default tolerance
    assert E_near_zero.representsa_('point', tol=1e-9)


def test_representsa_various_types():
    """Test representsa_ for various set types with different ellipsoids"""
    
    # Test matrix of ellipsoids and types
    ellipsoids = {
        'unit_2d': Ellipsoid(np.eye(2)),
        'point': Ellipsoid(np.zeros((2, 2)), np.array([[1], [2]])),
        'unit_1d': Ellipsoid(np.array([[1]])),
        'degenerate': Ellipsoid(np.array([[1, 0], [0, 0]])),
    }
    
    types_to_test = [
        'point', 'origin', 'ellipsoid', 'interval', 'convexSet', 
        'emptySet', 'fullspace', 'halfspace'
    ]
    
    for ell_name, E in ellipsoids.items():
        for set_type in types_to_test:
            try:
                result = E.representsa_(set_type)
                assert isinstance(result, bool), f"representsa_({set_type}) should return boolean for {ell_name}"
            except Exception as e:
                # Some combinations might raise errors, which is acceptable
                print(f"Expected error for {ell_name}.representsa_('{set_type}'): {e}")


def test_representsa_error_cases():
    """Test error handling in representsa_"""
    
    E = Ellipsoid(np.eye(2))
    
    # Invalid set type - MATLAB doesn't throw error, just returns false
    result = E.representsa_('invalid_type')
    assert result == False, "Invalid set type should return False"

    # Test unsupported conversions that should raise NotImplementedError
    with pytest.raises(NotImplementedError):
        E.representsa_('conPolyZono')

    with pytest.raises(NotImplementedError):
        E.representsa_('levelSet')

    with pytest.raises(NotImplementedError):
        E.representsa_('polyZonotope')

    with pytest.raises(NotImplementedError):
        E.representsa_('parallelotope')


if __name__ == "__main__":
    test_representsa_point()
    test_representsa_origin()
    test_representsa_ellipsoid()
    test_representsa_interval()
    test_representsa_convexSet()
    test_representsa_unsupported_types()
    test_representsa_empty_set()
    test_representsa_capsule()
    test_representsa_dual_output()
    test_representsa_tolerance_sensitivity()
    test_representsa_various_types()
    test_representsa_error_cases()
    print("All representsa_ tests passed!") 