import numpy as np
import pytest
from cora_python.contSet.polytope.polytope import Polytope
from cora_python.contSet.polytope.project import project

def test_project_V_rep():
    # 3D polytope
    V = np.array([[0., 1., 0., 0.], 
                  [0., 0., 1., 0.], 
                  [0., 0., 0., 1.]])
    P = Polytope(V)
    
    # Project onto [1, 2]
    P_proj = project(P, [1, 2])
    
    V_expected = np.array([[0., 1., 0., 0.], 
                           [0., 0., 1., 0.]])
    
    assert P_proj.dim() == 2
    # Vertices can be in a different order, so we sort them before comparing
    P_proj_v_sorted = sorted([tuple(v) for v in P_proj.V.T])
    V_expected_sorted = sorted([tuple(v) for v in V_expected.T])
    assert np.allclose(P_proj_v_sorted, V_expected_sorted)

def _check_point_in_polytope(A, b, point, tol=1e-9):
    return np.all(A @ point - b.flatten() <= tol)

def test_project_fourier_method():
    """Test Fourier-Motzkin elimination method"""
    # 3D cube from -1 to 1
    A = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=float)
    b = np.ones(6)
    P = Polytope(A, b)

    # Project onto [1, 2]
    P_proj = project(P, [1, 2], method='fourier')

    assert P_proj.dim() == 2

    # Expected vertices of the projected square
    expected_vertices = [np.array([1,1]), np.array([1,-1]), np.array([-1,1]), np.array([-1,-1])]
    
    # Check if expected vertices are in the projected polytope
    A_proj, b_proj = P_proj.A, P_proj.b

    for v in expected_vertices:
        assert _check_point_in_polytope(A_proj, b_proj, v)


@pytest.mark.skip(reason="pycddlib causes fatal errors, skipping for now")
def test_project_fourier_jones_method():
    """Test Fourier-Jones (pycddlib) method"""
    # pytest.importorskip("cdd") # Temporarily remove to avoid fatal errors or comment out
    
    # Use a different polytope to avoid duplication with the consistency test
    # 3D tetrahedron
    A = np.array([[1, 1, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)
    b = np.array([1, 0, 0, 0])
    P = Polytope(A, b)

    # Project onto [1, 2]
    P_proj = project(P, [1, 2], method='fourier_jones')

    assert P_proj.dim() == 2
    
    # Check that the origin is in the projected polytope
    assert _check_point_in_polytope(P_proj.A, P_proj.b, np.array([0, 0]))


def test_project_1d_to_itself():
    """Test projecting a 1D polytope to itself (MATLAB test case)"""
    A = np.array([[1], [-1]], dtype=float)
    b = np.array([1, 0])
    P = Polytope(A, b)
    P_proj = project(P, [1])
    
    # Should be identical
    assert P_proj.dim() == 1
    # Check that the polytope represents the same region [0, 1]
    assert _check_point_in_polytope(P_proj.A, P_proj.b, np.array([0.5]))
    assert _check_point_in_polytope(P_proj.A, P_proj.b, np.array([0]))
    assert _check_point_in_polytope(P_proj.A, P_proj.b, np.array([1]))
    assert not _check_point_in_polytope(P_proj.A, P_proj.b, np.array([1.1]))


def test_project_2d_bounded():
    """Test bounded 2D polytope projection (MATLAB test case)"""
    A = np.array([[-1, 0], [2, 4], [1, -2]], dtype=float)
    b = np.array([-1, 14, -1])
    P = Polytope(A, b)
    
    # Project to dimension 1
    P_proj = project(P, [1])
    vertices = P_proj.vertices_()
    # Should be [1, 3] (approximately)
    expected_range = [1, 3]
    vertices_1d = vertices.flatten()
    assert np.allclose(sorted(vertices_1d), expected_range, rtol=1e-6)
    
    # Project to dimension 2  
    P_proj = project(P, [2])
    vertices = P_proj.vertices_()
    vertices_1d = vertices.flatten()
    assert np.allclose(sorted(vertices_1d), expected_range, rtol=1e-6)


def test_project_2d_unbounded():
    """Test unbounded 2D polytope projection (MATLAB test case)"""
    P = Polytope(np.array([[1, 0], [-1, 0], [0, 1]], dtype=float), np.array([1, 1, 1]))
    
    # Project to dimension 1
    P_proj = project(P, [1])
    
    # Should still be bounded in the projected dimension
    assert P_proj.isBounded()


def test_project_2d_degenerate():
    """Test degenerate 2D polytope projection (MATLAB test case)"""
    P = Polytope(np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=float), 
                 np.array([1, 1, 1, -1]))
    P_proj = project(P, [1])
    
    # Should be the interval [-1, 1]
    assert P_proj.dim() == 1
    assert _check_point_in_polytope(P_proj.A, P_proj.b, np.array([-1]))
    assert _check_point_in_polytope(P_proj.A, P_proj.b, np.array([1]))
    assert _check_point_in_polytope(P_proj.A, P_proj.b, np.array([0]))


def test_project_3d_vertex_representation():
    """Test 3D polytope with vertex representation (MATLAB test case)"""
    V = np.array([[1, 1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, -1]], dtype=float).T
    P = Polytope(V)
    P_proj = project(P, [1, 3])
    
    # Expected vertices in projection
    V_proj = P_proj.vertices_()
    V_true = np.array([[-1, 1], [1, 1], [1, -1]], dtype=float).T
    
    # Sort vertices for comparison
    V_proj_sorted = V_proj[:, np.lexsort(V_proj)]
    V_true_sorted = V_true[:, np.lexsort(V_true)]
    
    assert np.allclose(V_proj_sorted, V_true_sorted, atol=1e-8)


def test_project_empty_polytope():
    """Test projecting an empty polytope"""
    # Create an empty polytope
    A = np.zeros((0, 3))
    b = np.zeros(0)
    P = Polytope(A, b)
    P_proj = project(P, [1])
    
    # Should remain empty but in lower dimension
    assert P_proj.dim() == 1
    # Empty polytope should have no constraints or represent the full space
    # In MATLAB CORA, empty polytope is full space


def test_project_with_equality_constraints():
    """Test projection with equality constraints"""
    # Create a polytope with equality constraints
    A = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=float)
    b = np.array([1, 1, 1, 1])
    Ae = np.array([[0, 0, 1]], dtype=float)  # z = 0.5
    be = np.array([0.5])
    
    P = Polytope(A, b, Ae, be)
    P_proj = project(P, [1, 2])
    
    # Should project to a 2D square
    assert P_proj.dim() == 2
    # Check corners of the square
    corners = [np.array([1, 1]), np.array([1, -1]), np.array([-1, 1]), np.array([-1, -1])]
    for corner in corners:
        assert _check_point_in_polytope(P_proj.A, P_proj.b, corner)


@pytest.mark.skip(reason="pycddlib causes fatal errors, skipping for now")
def test_project_both_methods_consistency():
    """Test that both fourier and fourier_jones methods give consistent results"""
    # pytest.importorskip("cdd") # Temporarily remove to avoid fatal errors or comment out
    
    # 3D cube
    A = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=float)
    b = np.ones(6)
    P = Polytope(A, b)
    
    # Project using both methods
    P_proj_fourier = project(P, [1, 2], method='fourier')
    P_proj_fourier_jones = project(P, [1, 2], method='fourier_jones')
    
    # Both should have same dimension
    assert P_proj_fourier.dim() == P_proj_fourier_jones.dim() == 2
    
    # Test that the same test points are accepted/rejected by both
    test_points = [
        np.array([0, 0]),
        np.array([1, 1]),
        np.array([1, -1]),
        np.array([-1, 1]),
        np.array([-1, -1]),
        np.array([1.1, 0]),  # Should be outside
        np.array([0, 1.1])   # Should be outside
    ]
    
    for point in test_points:
        in_fourier = _check_point_in_polytope(P_proj_fourier.A, P_proj_fourier.b, point)
        in_fourier_jones = _check_point_in_polytope(P_proj_fourier_jones.A, P_proj_fourier_jones.b, point)
        assert in_fourier == in_fourier_jones, f"Methods disagree on point {point}"


def test_project_error_cases():
    """Test error cases and edge conditions"""
    # Test projection to higher dimension than exists
    A = np.array([[1, 0], [-1, 0]], dtype=float)
    b = np.array([1, 1])
    P = Polytope(A, b)
    
    with pytest.raises(Exception):  # Should raise CORAerror
        project(P, [1, 2, 3])  # Asking for 3D projection from 2D polytope
    
    # Test invalid method
    with pytest.raises(Exception):  # Should raise CORAerror
        project(P, [1], method='invalid_method')


def test_project_high_dimensional():
    """Test projection from higher dimensions"""
    # Create a 5D hypercube
    dim = 5
    # Generate constraints for [-1, 1]^5
    A_list = []
    b_list = []
    for i in range(dim):
        # x_i <= 1
        constraint = np.zeros(dim)
        constraint[i] = 1
        A_list.append(constraint)
        b_list.append(1)
        
        # x_i >= -1  =>  -x_i <= 1
        constraint = np.zeros(dim)
        constraint[i] = -1
        A_list.append(constraint)
        b_list.append(1)
    
    A = np.array(A_list, dtype=float)
    b = np.array(b_list)
    P = Polytope(A, b)
    
    # Project to first 2 dimensions
    P_proj = project(P, [1, 2])
    
    assert P_proj.dim() == 2
    # Should be the unit square
    corners = [np.array([1, 1]), np.array([1, -1]), np.array([-1, 1]), np.array([-1, -1])]
    for corner in corners:
        assert _check_point_in_polytope(P_proj.A, P_proj.b, corner) 