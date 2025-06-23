"""
Test file for polytope isIntersecting_ method.

Based on test_polytope_isIntersecting.m from MATLAB CORA.
"""

import numpy as np
import pytest
from cora_python.contSet.polytope.polytope import Polytope
from cora_python.contSet.conZonotope.conZonotope import ConZonotope


def test_polytope_isIntersecting_1d_unbounded():
    """Test 1D unbounded & unbounded intersection"""
    # 1D, unbounded & unbounded
    A = np.array([[1]])
    b = np.array([1])
    P1 = Polytope(A, b)
    
    A = np.array([[-1]])
    b = np.array([5])
    P2 = Polytope(A, b)
    
    assert P1.isIntersecting_(P2, 'exact')


def test_polytope_isIntersecting_1d_unbounded_degenerate():
    """Test 1D unbounded & degenerate intersection"""
    # 1D, unbounded & degenerate
    A = np.array([[1]])
    b = np.array([1])
    P1 = Polytope(A, b)
    
    Ae = np.array([[1]])
    be = np.array([-5])
    P2 = Polytope(np.empty((0, 1)), np.empty((0,)), Ae, be)
    
    assert P1.isIntersecting_(P2, 'exact')


def test_polytope_isIntersecting_1d_bounded_unbounded():
    """Test 1D bounded & unbounded intersection in single point"""
    # 1D, bounded & unbounded, intersection in a single point
    A = np.array([[1], [-1]])
    b = np.array([1, 1])
    P1 = Polytope(A, b)
    
    Ae = np.array([[-1]])
    be = np.array([1])
    P2 = Polytope(np.empty((0, 1)), np.empty((0,)), Ae, be)
    
    assert P1.isIntersecting_(P2, 'exact')


def test_polytope_isIntersecting_1d_unbounded_point():
    """Test 1D unbounded & point"""
    # 1D, unbounded & point
    A = np.array([[1]])
    b = np.array([1])
    P1 = Polytope(A, b)
    
    p = np.array([[5]])
    assert not P1.isIntersecting_(p, 'exact')


def test_polytope_isIntersecting_1d_empty_point():
    """Test 1D fully empty & point"""
    # 1D, fully empty & point
    A = np.empty((0, 1))
    b = np.empty((0,))
    P1 = Polytope(A, b)
    
    p = np.array([[1]])
    assert P1.isIntersecting_(p, 'exact')


def test_polytope_isIntersecting_2d_empty_unbounded():
    """Test 2D fully empty & unbounded"""
    # 2D, fully empty & unbounded
    A = np.empty((0, 2))
    b = np.empty((0,))
    P1 = Polytope(A, b)
    
    A = np.array([[1, 0]])
    b = np.array([1])
    P2 = Polytope(A, b)
    
    assert P1.isIntersecting_(P2, 'exact')
    assert P2.isIntersecting_(P1, 'exact')


def test_polytope_isIntersecting_2d_bounded_quadrants():
    """Test 2D bounded polytopes in all quadrants (non-intersecting)"""
    # 2D, bounded polytopes in all quadrants
    V1 = np.array([[1, 1], [4, 0], [3, 3], [2, 4]]).T
    P1 = Polytope(V1)
    
    V2 = np.array([[-2, 1], [-4, 2], [-3, 4], [-1, 2]]).T
    P2 = Polytope(V2)
    
    V3 = np.array([[-1, -2], [-4, -1], [-3, -4], [-2, -3]]).T
    P3 = Polytope(V3)
    
    V4 = np.array([[1, -1], [2, -4], [6, -5], [5, -2]]).T
    P4 = Polytope(V4)
    
    # No combination should intersect
    assert not P1.isIntersecting_(P2, 'exact')
    assert not P1.isIntersecting_(P3, 'exact')
    assert not P1.isIntersecting_(P4, 'exact')
    assert not P2.isIntersecting_(P3, 'exact')
    assert not P2.isIntersecting_(P4, 'exact')
    assert not P3.isIntersecting_(P4, 'exact')


def test_polytope_isIntersecting_2d_bounded_point_cloud():
    """Test 2D bounded & point cloud (some contained, some not)"""
    # 2D, bounded & point cloud (some contained, some not)
    A = np.array([[1, 0], [-1, 1], [-1, -1]])
    b = np.array([1, 1, 1])
    P = Polytope(A, b)
    
    V = np.array([[0.5, 0], [0, 1.5], [-0.5, -1], [0, -1.5], [1, -1]]).T
    
    expected = [True, False, False, False, True]
    for i in range(V.shape[1]):
        point = V[:, i:i+1]
        result = P.isIntersecting_(point, 'exact')
        assert result == expected[i], f"Point {i} intersection result mismatch"


def test_polytope_isIntersecting_2d_bounded_contain_origin():
    """Test 2D bounded & bounded (both contain the origin)"""
    # 2D, bounded & bounded (both contain the origin)
    V = np.array([[2, 2], [3, -1], [-1, 0], [0, 3], [1, 3]]).T
    P = Polytope(V)
    
    # Multiply by -1 -> resulting polytope also contains the origin
    V_ = -1 * V
    P_ = Polytope(V_)
    
    assert P.isIntersecting_(P_, 'exact')


def test_polytope_isIntersecting_2d_unbounded_unbounded():
    """Test 2D unbounded & unbounded"""
    # 2D, unbounded & unbounded
    A1 = np.array([[1, 0], [-1, 0], [0, 1]])
    b1 = np.array([1, 1, 5])
    P1 = Polytope(A1, b1)
    
    A2 = np.array([[1, 0], [0, 1], [0, -1]])
    b2 = np.array([5, 1, 1])
    P2 = Polytope(A2, b2)
    
    assert P1.isIntersecting_(P2, 'exact')


def test_polytope_isIntersecting_2d_unbounded_single_point():
    """Test 2D unbounded & unbounded, intersection in single point"""
    # 2D, unbounded & unbounded, intersection in single point
    A1 = np.array([[1, 0], [-1, 0], [0, 1]])
    b1 = np.array([1, -1, 5])
    P1 = Polytope(A1, b1)
    
    A2 = np.array([[1, 0], [0, 1], [0, -1]])
    b2 = np.array([5, 1, -1])
    P2 = Polytope(A2, b2)
    
    assert P1.isIntersecting_(P2, 'exact')


def test_polytope_isIntersecting_2d_unbounded_no_intersection():
    """Test 2D unbounded & unbounded with no intersection"""
    # 2D, unbounded & unbounded
    A1 = np.array([[1, 0], [-1, 0], [0, 1]])
    b1 = np.array([1, -1, 5])
    P1 = Polytope(A1, b1)
    
    A2 = np.array([[1, 0], [0, 1], [0, -1]])
    b2 = np.array([-5, 1, -1])
    P2 = Polytope(A2, b2)
    
    assert not P1.isIntersecting_(P2, 'exact')


def test_polytope_isIntersecting_2d_degenerate_contained():
    """Test 2D degenerate contained in unbounded"""
    # 2D, degenerate contained in unbounded
    A1 = np.array([[1, 0], [-1, 0], [0, 1]])
    b1 = np.array([1, 1, 1])
    P1 = Polytope(A1, b1)
    
    A2 = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    b2 = np.array([0.5, 0.5, 0.5, -0.5])
    P2 = Polytope(A2, b2)
    
    assert P1.isIntersecting_(P2, 'exact')


def test_polytope_isIntersecting_constrained_zonotope():
    """Test 2D intersections of polytope and constrained zonotope"""
    # 2D, intersections of polytope and constrained zonotope
    c = np.array([[0.0], [0.0]])
    G = np.array([[1.5, -1.5, 0.5], [1.0, 0.5, -1.0]])
    A = np.array([[1.0, 1.0, 1.0]])
    b = np.array([1.0])
    cZ = ConZonotope(c, G, A, b)
    
    # Unbounded polytope, no intersection
    A_poly = np.array([[1, 0]])
    b_poly = np.array([-3])
    P = Polytope(A_poly, b_poly)
    assert not P.isIntersecting_(cZ, 'exact')
    
    # Unbounded polytope, intersection
    A_poly = np.array([[1, 0]])
    b_poly = np.array([0])
    P = Polytope(A_poly, b_poly)
    assert P.isIntersecting_(cZ, 'exact')
    
    # Unbounded polytope, intersection in a single point
    A_poly = np.array([[0, 1]])
    b_poly = np.array([-1.5])
    P = Polytope(A_poly, b_poly)
    assert P.isIntersecting_(cZ, 'exact', 1e-6)
    
    # Bounded polytope, no intersection
    A_poly = np.array([[-1, -1], [1, 0], [0, 1]])
    b_poly = np.array([-4, 5, 5])
    P = Polytope(A_poly, b_poly)
    assert not P.isIntersecting_(cZ, 'exact')
    
    # Bounded polytope, intersection
    A_poly = np.array([[-1, -1], [1, 0], [0, 1]])
    b_poly = np.array([-2, 5, 5])
    P = Polytope(A_poly, b_poly)
    assert P.isIntersecting_(cZ, 'exact')
    
    # Bounded polytope, intersection in a single point
    A_poly = np.array([[-1, -1], [1, 0], [0, 1]])
    b_poly = np.array([-3, 5, 5])
    P = Polytope(A_poly, b_poly)
    assert P.isIntersecting_(cZ, 'exact')
    
    # Intersection with empty polytope
    A_poly = np.array([[1, 0], [-1, 0]])
    b_poly = np.array([-1, -1])
    P = Polytope(A_poly, b_poly)
    assert not P.isIntersecting_(cZ, 'exact')


if __name__ == "__main__":
    pytest.main([__file__]) 