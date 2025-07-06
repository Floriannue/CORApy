"""
Test cases for ellipsoid contains method.

This test mirrors the MATLAB test_ellipsoid_contains.m exactly and adds comprehensive edge cases.
"""

import numpy as np
import pytest

from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid
from cora_python.contSet.emptySet.emptySet import EmptySet
from cora_python.contSet.fullspace.fullspace import Fullspace
from cora_python.contSet.zonotope.zonotope import Zonotope
from cora_python.contSet.polytope.polytope import Polytope
from cora_python.contSet.interval.interval import Interval
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def test_ellipsoid_contains():
    """Test ellipsoid contains method - mirrors MATLAB test_ellipsoid_contains.m"""
    
    # Test 1: all-zero shape matrix (point ellipsoid)
    Q = np.zeros((3, 3))
    q = np.array([[3], [2], [1]])
    E = Ellipsoid(Q, q)
    
    # empty set
    O = EmptySet(3)
    assert E.contains(O)
    
    # fullspace
    fs = Fullspace(3)
    assert not E.contains(fs)
    
    # points
    assert E.contains(q)
    assert not E.contains(q + q)
    
    # zonotope
    Z = Zonotope(q)
    assert E.contains(Z)
    
    # polytope
    P = Polytope(q)
    assert E.contains(P)
    
    # Test 2: non-degenerate ellipsoid
    Q = np.array([[2, 0], [0, 1]])
    q = np.array([[2], [-1]])
    M = np.array([[2, -1], [1, 1]])
    E = M @ Ellipsoid(Q, q)
    
    # points
    p = np.array([[4, 3, 5, 6, 8], [0, 1, 1, 2, 2]])
    assert np.all(E.contains(p))
    
    # zonotope
    c = np.array([[5], [1]])
    G_inside = np.array([[0.2, 0.3, 0.5], [0, -0.3, 0.2]])
    G_outside = np.array([[1, 1, 0.5], [0, -1, 1]])
    Z_inside = Zonotope(c, G_inside)
    assert E.contains(Z_inside)
    Z_outside = Zonotope(c, G_outside)
    assert not E.contains(Z_outside)
    
    # interval
    I_inside = Interval(np.array([[3], [0.5]]), np.array([[6], [1]]))
    assert E.contains(I_inside)
    I_outside = Interval(np.array([[3], [0]]), np.array([[7], [2]]))
    assert not E.contains(I_outside)
    
    # polytope
    A_inside = np.array([[1, 0], [-1, 1], [-1, -1]])
    b_inside = np.array([[4], [-2.2], [-4]])
    P_inside = Polytope(A_inside, b_inside)
    assert E.contains(P_inside)
    
    A_outside = np.array([[1, 0], [-1, 1], [-1, -1]]) 
    b_outside = np.array([[7], [-2], [-4]])
    P_outside = Polytope(A_outside, b_outside)
    assert not E.contains(P_outside)
    
    # Test 3: degenerate ellipsoid
    Q = np.array([[2, 0, 0], [0, 3, 0], [0, 0, 1]])
    q = np.array([[2], [-1], [4]])
    M = np.array([[2, -1, 1], [1, 1, 2], [-1, 3, 2]])
    E = M @ Ellipsoid(Q, q)
    
    # points
    p_offsets = np.array([[0.5, 0.2, -0.5], [-0.8, -1.2, 0.2], [1.2, 1.5, -0.5]]).T  # Transpose to match MATLAB
    p = M @ (q + p_offsets)
    assert np.all(E.contains(p))
    
    # zonotope
    G = np.array([[0.1, 0.2, 0.3], [-0.2, 0.2, 0.1], [0.0, 0.1, -0.1]])
    Z = M @ Zonotope(q, G)
    assert E.contains(Z)


def test_ellipsoid_contains_point_ellipsoid_edge_cases():
    """Test edge cases with point ellipsoids (zero shape matrix)"""
    
    # Point ellipsoid
    Q = np.zeros((2, 2))
    q = np.array([[1], [2]])
    E_point = Ellipsoid(Q, q)
    
    # Test 1: Point ellipsoid contains same point
    assert E_point.contains(q)
    
    # Test 2: Point ellipsoid does not contain different point
    q_diff = np.array([[2], [3]])
    assert not E_point.contains(q_diff)
    
    # Test 3: Point ellipsoid contains empty set
    empty = EmptySet(2)
    assert E_point.contains(empty)
    
    # Test 4: Point ellipsoid does not contain regular ellipsoid
    Q_reg = np.eye(2)
    q_reg = np.array([[1], [2]])
    E_reg = Ellipsoid(Q_reg, q_reg)
    assert not E_point.contains(E_reg)
    
    # Test 5: Point ellipsoid contains another point ellipsoid at same location
    Q2 = np.zeros((2, 2))
    E_point2 = Ellipsoid(Q2, q)
    assert E_point.contains(E_point2)
    
    # Test 6: Point ellipsoid does not contain point ellipsoid at different location
    E_point3 = Ellipsoid(Q2, q_diff)
    assert not E_point.contains(E_point3)
    
    # Test 7: Point ellipsoid contains point zonotope at same location
    Z_point = Zonotope(q)
    assert E_point.contains(Z_point)
    
    # Test 8: Point ellipsoid does not contain point zonotope at different location
    Z_point_diff = Zonotope(q_diff)
    assert not E_point.contains(Z_point_diff)


def test_ellipsoid_contains_empty_set_edge_cases():
    """Test edge cases with empty ellipsoids"""
    
    # Empty ellipsoid
    E_empty = EmptySet(2)
    
    # Test 1: Empty ellipsoid contains empty set
    empty2 = EmptySet(2)
    assert E_empty.contains(empty2)
    
    # Test 2: Empty ellipsoid does not contain non-empty point
    p = np.array([[1], [2]])
    assert not E_empty.contains(p)
    
    # Test 3: Empty ellipsoid does not contain regular ellipsoid
    Q = np.eye(2)
    q = np.zeros((2, 1))
    E_reg = Ellipsoid(Q, q)
    assert not E_empty.contains(E_reg)
    
    # Test 4: Regular ellipsoid contains empty set
    assert E_reg.contains(E_empty)
    
    # Test 5: Empty ellipsoid contains empty numeric array (mathematically correct)
    empty_array = np.array([]).reshape(2, 0)
    assert E_empty.contains(empty_array)


def test_ellipsoid_contains_numeric_points_edge_cases():
    """Test edge cases with numeric points"""
    
    Q = np.eye(2)
    q = np.zeros((2, 1))
    E = Ellipsoid(Q, q)
    
    # Test 1: Single point inside
    p_inside = np.array([[0.5], [0.5]])
    assert E.contains(p_inside)
    
    # Test 2: Single point outside
    p_outside = np.array([[2], [2]])
    assert not E.contains(p_outside)
    
    # Test 3: Multiple points - some inside, some outside
    p_mixed = np.array([[0.5, 2, 0.8], [0.5, 2, 0.6]])
    result = E.contains(p_mixed)
    # Should return array of results for multiple points
    if isinstance(result, np.ndarray):
        expected = np.array([True, False, True])
        assert np.array_equal(result, expected)
    else:
        # If it returns a single boolean, it should be False (not all contained)
        assert not result
    
    # Test 4: Point exactly on boundary
    p_boundary = np.array([[1], [0]])
    assert E.contains(p_boundary)
    
    # Test 5: Empty numeric array
    empty_array = np.array([]).reshape(2, 0)
    assert E.contains(empty_array)
    
    # Test 6: 1D array (should be reshaped to column vector)
    p_1d = np.array([0.5, 0.5])
    assert E.contains(p_1d)


def test_ellipsoid_contains_tolerance_edge_cases():
    """Test tolerance effects on containment"""
    
    Q = np.eye(2)
    q = np.zeros((2, 1))
    E = Ellipsoid(Q, q)
    
    # Test 1: Point slightly outside boundary with different tolerances
    p_close = np.array([[1.001], [0]])
    
    # Strict tolerance should reject
    assert not E.contains(p_close, tol=1e-6)
    
    # Loose tolerance should accept
    assert E.contains(p_close, tol=1e-2)
    
    # Test 2: Point exactly on boundary with different tolerances
    p_exact = np.array([[1], [0]])
    assert E.contains(p_exact, tol=1e-12)
    assert E.contains(p_exact, tol=1e-6)
    
    # Test 3: Point slightly inside boundary
    p_inside = np.array([[0.999], [0]])
    assert E.contains(p_inside, tol=1e-12)
    assert E.contains(p_inside, tol=1e-6)


def test_ellipsoid_contains_degenerate_ellipsoid_edge_cases():
    """Test edge cases with degenerate ellipsoids"""
    
    # Test 1: Rank-deficient ellipsoid (2D subspace in 3D)
    Q = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
    q = np.zeros((3, 1))
    E_degenerate = Ellipsoid(Q, q)
    
    # Point in the same subspace should be contained
    p_in_subspace = np.array([[0.5], [0.5], [0]])
    assert E_degenerate.contains(p_in_subspace)
    
    # Point outside the subspace should not be contained
    p_outside_subspace = np.array([[0], [0], [1]])
    assert not E_degenerate.contains(p_outside_subspace)
    
    # Test 2: Line ellipsoid (1D subspace in 2D)
    Q_line = np.array([[1, 0], [0, 0]])
    q_line = np.zeros((2, 1))
    E_line = Ellipsoid(Q_line, q_line)
    
    # Point on the line should be contained
    p_on_line = np.array([[0.8], [0]])
    assert E_line.contains(p_on_line)
    
    # Point off the line should not be contained
    p_off_line = np.array([[0.5], [0.5]])
    assert not E_line.contains(p_off_line)


def test_ellipsoid_contains_ellipsoid_containment():
    """Test ellipsoid-in-ellipsoid containment edge cases"""
    
    # Test 1: Larger ellipsoid contains smaller ellipsoid
    Q1 = 4 * np.eye(2)
    q1 = np.zeros((2, 1))
    E1 = Ellipsoid(Q1, q1)
    
    Q2 = np.eye(2)
    q2 = np.zeros((2, 1))
    E2 = Ellipsoid(Q2, q2)
    
    assert E1.contains(E2)
    assert not E2.contains(E1)
    
    # Test 2: Same ellipsoid contains itself
    assert E1.contains(E1)
    assert E2.contains(E2)
    
    # Test 3: Translated ellipsoid containment
    Q3 = 4 * np.eye(2)
    q3 = np.array([[1], [1]])
    E3 = Ellipsoid(Q3, q3)
    
    Q4 = np.eye(2)
    q4 = np.array([[1], [1]])
    E4 = Ellipsoid(Q4, q4)
    
    assert E3.contains(E4)
    assert not E4.contains(E3)
    
    # Test 4: Overlapping but not contained ellipsoids
    Q5 = np.eye(2)
    q5 = np.array([[0.5], [0]])
    E5 = Ellipsoid(Q5, q5)
    
    # E2 and E5 overlap but neither contains the other
    assert not E2.contains(E5)
    assert not E5.contains(E2)


def test_ellipsoid_contains_method_parameters():
    """Test different method parameters"""
    
    Q = np.eye(2)
    q = np.zeros((2, 1))
    E = Ellipsoid(Q, q)
    
    # Test with zonotope for both exact and approx methods
    # Use a smaller zonotope that is actually contained in the unit ellipsoid
    Z = Zonotope(np.array([[0.3], [0.3]]), np.array([[0.1, 0.05], [0.05, 0.1]]))
    
    # Test 1: Exact method
    assert E.contains(Z, method='exact')
    
    # Test 2: Approx method
    assert E.contains(Z, method='approx')
    
    # Test 3: Point with different methods
    p = np.array([[0.5], [0.5]])
    assert E.contains(p, method='exact')
    assert E.contains(p, method='approx')


def test_ellipsoid_contains_return_formats():
    """Test different return formats with optional parameters"""
    
    Q = np.eye(2)
    q = np.zeros((2, 1))
    E = Ellipsoid(Q, q)
    
    p = np.array([[0.5], [0.5]])
    
    # Test 1: Simple boolean return
    result = E.contains(p)
    assert isinstance(result, (bool, np.bool_))
    assert result
    
    # Test 2: With certificate
    result, cert = E.contains(p, return_cert=True)
    assert isinstance(result, (bool, np.bool_))
    assert isinstance(cert, (bool, np.bool_))
    assert result and cert
    
    # Test 3: With scaling
    result, cert, scaling = E.contains(p, return_scaling=True)
    assert isinstance(result, (bool, np.bool_))
    assert isinstance(cert, (bool, np.bool_))
    assert isinstance(scaling, (int, float, np.number))
    assert result and cert and 0 <= scaling <= 1


def test_ellipsoid_contains_high_dimensional():
    """Test containment in higher dimensions"""
    
    # Test 1: 4D ellipsoid
    Q = np.diag([1, 4, 9, 16])
    q = np.array([[1], [2], [3], [4]])
    E = Ellipsoid(Q, q)
    
    # Point inside
    p_inside = np.array([[1.5], [2.5], [3.2], [4.1]])
    assert E.contains(p_inside)
    
    # Point outside
    p_outside = np.array([[5], [5], [5], [5]])
    assert not E.contains(p_outside)
    
    # Test 2: High-dimensional point ellipsoid
    Q_point = np.zeros((5, 5))
    q_point = np.array([[1], [2], [3], [4], [5]])
    E_point = Ellipsoid(Q_point, q_point)
    
    assert E_point.contains(q_point)
    
    q_diff = q_point + 0.1
    assert not E_point.contains(q_diff)


def test_ellipsoid_contains_error_handling():
    """Test that unsupported set types raise appropriate errors"""
    Q = np.eye(2)
    c = np.zeros((2, 1))
    E = Ellipsoid(Q, c)
    
    # Test with mock object that doesn't have required methods
    class MockSet:
        def __init__(self):
            pass
    
    mock_set = MockSet()
    
    # Should raise CORAerror for unsupported types (matches MATLAB behavior)
    with pytest.raises(CORAerror):
        E.contains(mock_set)


def test_ellipsoid_contains_unsupported_method():
    """Test unsupported method parameter."""
    Q = np.eye(2)
    c = np.zeros((2, 1))
    E = Ellipsoid(Q, c)
    
    p = np.array([[0.5], [0]])
    
    with pytest.raises(CORAerror):
        E.contains(p, method='unknown')


def test_ellipsoid_contains_boundary_precision():
    """Test precision at ellipsoid boundary"""
    
    Q = np.eye(2)
    q = np.zeros((2, 1))
    E = Ellipsoid(Q, q)
    
    # Test points at various positions on the boundary
    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    for angle in angles:
        p_boundary = np.array([[np.cos(angle)], [np.sin(angle)]])
        assert E.contains(p_boundary), f"Boundary point at angle {angle} should be contained"
        
        # Point slightly outside
        p_outside = 1.01 * p_boundary
        assert not E.contains(p_outside, tol=1e-6), f"Point outside at angle {angle} should not be contained"
        
        # Point slightly inside
        p_inside = 0.99 * p_boundary
        assert E.contains(p_inside), f"Point inside at angle {angle} should be contained"


def test_ellipsoid_contains_zonotope_dimensions():
    """Test zonotope containment with different dimensions"""
    
    # Test 1: 2D case (should use vertices)
    Q = 4 * np.eye(2)
    q = np.zeros((2, 1))
    E = Ellipsoid(Q, q)
    
    Z_2d = Zonotope(np.array([[0.5], [0.5]]), np.array([[0.3, 0.2], [0.2, 0.3]]))
    assert E.contains(Z_2d)
    
    # Test 2: 3D case (should use special vertex enumeration)
    Q_3d = 4 * np.eye(3)
    q_3d = np.zeros((3, 1))
    E_3d = Ellipsoid(Q_3d, q_3d)
    
    Z_3d = Zonotope(np.array([[0.5], [0.5], [0.5]]), 
                    np.array([[0.3, 0.2, 0.1], [0.2, 0.3, 0.1], [0.1, 0.2, 0.3]]))
    assert E_3d.contains(Z_3d)
    
    # Test 3: Higher dimensional case (should use priv_venumZonotope)
    Q_4d = 4 * np.eye(4)
    q_4d = np.zeros((4, 1))
    E_4d = Ellipsoid(Q_4d, q_4d)
    
    Z_4d = Zonotope(np.array([[0.5], [0.5], [0.5], [0.5]]), 
                    np.array([[0.2, 0.1, 0.1], [0.1, 0.2, 0.1], 
                             [0.1, 0.1, 0.2], [0.1, 0.1, 0.1]]))
    assert E_4d.contains(Z_4d) 