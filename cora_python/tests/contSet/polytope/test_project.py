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

def test_project_H_rep_fourier():
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

def test_project_H_rep_fourier_jones():
    pytest.importorskip("pypoman")
    
    # 3D cube from -1 to 1
    A = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=float)
    b = np.ones(6)
    P = Polytope(A, b)

    # Project onto [1, 2]
    P_proj = project(P, [1, 2], method='fourier_jones')

    assert P_proj.dim() == 2

    # Expected vertices of the projected square
    expected_vertices = [np.array([1,1]), np.array([1,-1]), np.array([-1,1]), np.array([-1,-1])]
    
    # Check if expected vertices are in the projected polytope
    # Since pypoman returns a different representation, we check containment
    A_proj, b_proj = P_proj.A, P_proj.b
    
    for v in expected_vertices:
        assert _check_point_in_polytope(A_proj, b_proj, v) 