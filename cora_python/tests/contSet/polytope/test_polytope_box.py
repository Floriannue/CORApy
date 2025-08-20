import pytest
import numpy as np
from cora_python.contSet.polytope.polytope import Polytope
from cora_python.contSet.polytope.box import box
from cora_python.contSet.polytope.dim import dim

def test_polytope_box_basic():
    """Test basic box enclosure functionality for a 2D polytope."""
    # Simple diamond shape test case from MATLAB test_polytope_box.m
    # This is a diamond with vertices at (±2,0) and (0,±2)
    # The box enclosure should be [-2,2] × [-2,2]
    A_matlab = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
    b_matlab = np.array([[2], [2], [2], [2]])
    P = Polytope(A=A_matlab, b=b_matlab)

    B_enclosing = box(P)

    # Expected bounding box values (manually verified)
    # For the diamond shape, the box should be [-2,2] × [-2,2]
    expected_min_x = -2.0
    expected_max_x = 2.0
    expected_min_y = -2.0
    expected_max_y = 2.0

    tol = 1e-9  # Tolerance for numerical comparisons

    assert isinstance(B_enclosing, Polytope)
    assert B_enclosing.isHRep is True

    # Extract actual bounds from the resulting polytope's A and b
    actual_A = B_enclosing.A
    actual_b = B_enclosing.b

    idx_x_plus = np.where((actual_A == np.array([1, 0])).all(axis=1))[0]
    idx_x_minus = np.where((actual_A == np.array([-1, 0])).all(axis=1))[0]
    idx_y_plus = np.where((actual_A == np.array([0, 1])).all(axis=1))[0]
    idx_y_minus = np.where((actual_A == np.array([0, -1])).all(axis=1))[0]

    assert len(idx_x_plus) == 1 and len(idx_x_minus) == 1
    assert len(idx_y_plus) == 1 and len(idx_y_minus) == 1

    actual_max_x = actual_b[idx_x_plus[0]][0]
    actual_min_x = -actual_b[idx_x_minus[0]][0]
    actual_max_y = actual_b[idx_y_plus[0]][0]
    actual_min_y = -actual_b[idx_y_minus[0]][0]

    assert np.isclose(actual_max_x, expected_max_x, atol=tol)
    assert np.isclose(actual_min_x, expected_min_x, atol=tol)
    assert np.isclose(actual_max_y, expected_max_y, atol=tol)
    assert np.isclose(actual_min_y, expected_min_y, atol=tol)

def test_polytope_box_empty_polytope():
    """Test box enclosure for an empty polytope."""
    # Create an empty 2D polytope (e.g., x >= 1, x <= 0)
    P_empty = Polytope(A=np.array([[1.0, 0.0], [-1.0, 0.0]]), b=np.array([[-1.0], [-1.0]]))
    B_enclosing = box(P_empty)
    assert B_enclosing.isemptyobject() is True

def test_polytope_box_fullspace():
    """Test box enclosure for a fullspace polytope."""
    # Create a 2D fullspace polytope (no constraints)
    P_fullspace = Polytope(A=np.zeros((0, 2)), b=np.zeros((0, 1)))
    B_enclosing = box(P_fullspace)
    # A fullspace box should also be a fullspace
    assert B_enclosing.isFullDim() is True
    assert B_enclosing.isBounded() is False

def test_polytope_box_1d_polytope():
    """Test box enclosure for a 1D polytope."""
    A_1d = np.array([[1.0], [-1.0]])
    b_1d = np.array([[1.0], [0.5]])  # x <= 1, x >= -0.5
    P_1d = Polytope(A=A_1d, b=b_1d)
    B_enclosing = box(P_1d)

    expected_min_x_1d = -0.5
    expected_max_x_1d = 1.0
    tol = 1e-9

    assert dim(B_enclosing) == 1
    
    actual_A = B_enclosing.A
    actual_b = B_enclosing.b

    idx_x_plus_1d = np.where((actual_A == np.array([[1.0]])).all(axis=1))[0]
    idx_x_minus_1d = np.where((actual_A == np.array([[-1.0]])).all(axis=1))[0]

    assert len(idx_x_plus_1d) == 1 and len(idx_x_minus_1d) == 1

    actual_max_x_1d = actual_b[idx_x_plus_1d[0]][0]
    actual_min_x_1d = -actual_b[idx_x_minus_1d[0]][0]

    assert np.isclose(actual_max_x_1d, expected_max_x_1d, atol=tol)
    assert np.isclose(actual_min_x_1d, expected_min_x_1d, atol=tol)

def test_polytope_box_already_box():
    """Test box enclosure for a polytope that is already an axis-aligned box."""
    A_box = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    b_box = np.array([[1.0], [1.0], [1.0], [1.0]])  # x in [-1,1], y in [-1,1]
    P_box = Polytope(A=A_box, b=b_box)
    B_enclosing = box(P_box)
    
    tol = 1e-9

    assert dim(B_enclosing) == 2

    # For a perfect box, the A and b should be very close to original
    assert np.allclose(B_enclosing.A, A_box, atol=tol)
    assert np.allclose(B_enclosing.b, b_box, atol=tol)

def test_polytope_box_single_point():
    """Test box enclosure for a single-point polytope."""
    Ae_point = np.array([[1.0, 0.0], [0.0, 1.0]])
    be_point = np.array([[0.5], [0.5]])  # Point (0.5, 0.5)
    P_point = Polytope(A=np.zeros((0,2)), b=np.zeros((0,1)), Ae=Ae_point, be=be_point)
    B_enclosing = box(P_point)

    expected_x = 0.5
    expected_y = 0.5
    tol = 1e-9

    actual_A = B_enclosing.A
    actual_b = B_enclosing.b

    idx_x_plus = np.where((actual_A == np.array([1.0, 0.0])).all(axis=1))[0]
    idx_x_minus = np.where((actual_A == np.array([-1.0, 0.0])).all(axis=1))[0]
    idx_y_plus = np.where((actual_A == np.array([0.0, 1.0])).all(axis=1))[0]
    idx_y_minus = np.where((actual_A == np.array([0.0, -1.0])).all(axis=1))[0]

    assert len(idx_x_plus) == 1 and len(idx_x_minus) == 1
    assert len(idx_y_plus) == 1 and len(idx_y_minus) == 1

    actual_max_x = actual_b[idx_x_plus[0]][0]
    actual_min_x = -actual_b[idx_x_minus[0]][0]
    actual_max_y = actual_b[idx_y_plus[0]][0]
    actual_min_y = -actual_b[idx_y_minus[0]][0]

    assert np.isclose(actual_max_x, expected_x, atol=tol)
    assert np.isclose(actual_min_x, expected_x, atol=tol)
    assert np.isclose(actual_max_y, expected_y, atol=tol)
    assert np.isclose(actual_min_y, expected_y, atol=tol)
