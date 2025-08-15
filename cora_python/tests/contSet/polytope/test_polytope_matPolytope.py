import numpy as np
import pytest
from cora_python.contSet.polytope.polytope import Polytope
from cora_python.matrixSet.matPolytope.matPolytope import MatPolytope

def test_matpolytope_basic_conversion():
    """Test basic conversion from Polytope to MatPolytope."""
    # Define a simple 2D polytope (a square)
    A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    b = np.array([[1], [1], [1], [1]])
    P = Polytope(A, b)

    matP = P.matPolytope()

    assert isinstance(matP, MatPolytope)
    assert matP.V.ndim == 3
    # For a square [-1,1]x[-1,1], vertices are [[1,1], [1,-1], [-1,1], [-1,-1]]
    # Number of vertices should be 4
    # Each vertex is 2x1 matrix (dim x 1)
    assert matP.V.shape[0] == P.dim()
    assert matP.V.shape[1] == 1
    assert matP.V.shape[2] == P.V.shape[1] # Number of vertices

    # Define expected vertices for the square [-1,1]x[-1,1]
    expected_vertices = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]]).T # (dim x num_vertices)

    # Verify some vertices by sorting both sets of vertices to ensure order-independent comparison
    # Convert matP.V (n, 1, N) to (n, N) and then transpose to (N, n) for row sorting
    actual_vertices_2d = matP.V.reshape(matP.V.shape[0], matP.V.shape[2]) # (dim, num_vertices)
    # Transpose to (num_vertices, dim) for sorting rows
    actual_vertices_sorted = actual_vertices_2d.T[actual_vertices_2d.T[:,0].argsort()] # Sort by first column for consistency

    expected_vertices_sorted = expected_vertices.T[expected_vertices.T[:,0].argsort()] # Sort by first column

    assert np.allclose(actual_vertices_sorted, expected_vertices_sorted)

def test_matpolytope_empty_polytope():
    """Test conversion of an empty Polytope to MatPolytope."""
    P = Polytope.empty(3)

    matP = P.matPolytope()

    assert isinstance(matP, MatPolytope)
    # For an empty polytope, V should be (dim, 1, 0) or (0,0,0) if dim is 0
    assert matP.V.shape == (3, 1, 0)

def test_matpolytope_point_polytope():
    """Test conversion of a point Polytope to MatPolytope."""
    point = np.array([[0.5], [1.0]])
    P = Polytope(V=point)

    matP = P.matPolytope()

    assert isinstance(matP, MatPolytope)
    assert matP.V.shape == (2, 1, 1) # 2D, 1x1 matrix vertex, 1 vertex
    assert np.allclose(matP.V[:,:,0], point)
