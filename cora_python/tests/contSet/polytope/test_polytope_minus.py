import numpy as np
from cora_python.contSet.polytope import Polytope

def test_minus_v_rep():
    """Test subtracting a vector from a polytope in V-representation."""
    V = np.array([[0, 2], [0, 2]]).T  # A square
    p = Polytope(V)
    v = np.array([1, 1])
    
    p_minus_v = p - v
    
    V_expected = np.array([[-1, 1], [-1, 1]]).T
    
    assert np.allclose(p_minus_v.V, V_expected)
    # Original polytope should be unchanged
    assert np.allclose(p.V, V)

def test_minus_h_rep():
    """Test subtracting a vector from a polytope in H-representation."""
    # A square centered at (0,0) with side length 2
    A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    b = np.ones((4, 1))
    p = Polytope(A, b)
    v = np.array([2, -3])
    
    p_minus_v = p - v
    
    # New b should be b_old + A @ v
    b_expected = b + A @ v.reshape(-1, 1)

    # Check the H-representation
    assert np.allclose(p_minus_v.A, A)
    assert np.allclose(p_minus_v.b, b_expected)
    assert p_minus_v.Ae is None
    assert p_minus_v.be is None
    
    # Original polytope should be unchanged
    assert np.allclose(p.b, b)

def test_minus_type_error():
    """Test that subtracting a non-vector raises a TypeError."""
    p1 = Polytope(np.eye(2))
    p2 = Polytope(np.eye(2) * 0.5)
    
    try:
        p1 - p2
        assert False, "Should have raised a TypeError for Polytope-Polytope subtraction"
    except TypeError as e:
        assert "Minkowski difference" in str(e)

    try:
        p1 - "not a vector"
        assert False, "Should have raised a TypeError for invalid type"
    except TypeError:
        pass # Expected 