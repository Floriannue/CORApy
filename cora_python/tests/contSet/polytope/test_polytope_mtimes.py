import numpy as np
import pytest
from cora_python.contSet.polytope import Polytope
from cora_python.contSet.polytope.mtimes import mtimes

def test_mtimes_numeric():
    # H-rep
    A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    b = np.array([1, 1, 1, 1])
    P = Polytope(A, b)
    M = np.array([[2, 1], [-1, 2]])
    
    # Left multiplication
    P_res = mtimes(M, P)
    
    A_exp = np.array([[ 0.4, -0.2], [-0.4,  0.2], [ 0.2,  0.4], [-0.2, -0.4]])
    b_exp = np.array([1, 1, 1, 1])

    assert np.allclose(P_res.A, A_exp)
    assert np.allclose(P_res.b, b_exp)

    # Scalar multiplication
    P_scaled = mtimes(P, 2)
    assert np.allclose(P_scaled.A, A)
    assert np.allclose(P_scaled.b, b * 2)
    
    # V-rep - use correct d × n_vertices format (2 × 4)
    V = np.array([[1,1,-1,-1], [1,-1,1,-1]])  # 2D space, 4 vertices
    P_v = Polytope(V)
    P_v_res = mtimes(M, P_v)
    # Expected result: M @ V = (2×2) @ (2×4) = (2×4)
    V_exp = M @ V
    assert np.allclose(P_v_res.V, V_exp)

def test_mtimes_interval():
    # This test will be more complex and requires IntervalMatrix
    # For now, we can skip it or create a mock
    pass 

def test_mtimes_v_rep():
    """Test matrix multiplication of a polytope in V-representation."""
    # Use correct d × n_vertices format (2 × 4)
    V = np.array([[0, 1, 1, 0], [0, 0, 1, 1]])  # Unit square: 2D space, 4 vertices
    p = Polytope(V)
    m = np.array([[2, 0], [0, 0.5]])  # Scaling matrix
    
    p_res = m @ p
    
    V_expected = m @ V  # (2×2) @ (2×4) = (2×4)
    
    assert isinstance(p_res, Polytope)
    assert np.allclose(p_res.V, V_expected)

def test_mtimes_h_rep_invertible():
    """Test matrix multiplication of a polytope in H-representation with an invertible matrix."""
    # Unit square centered at origin
    A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    b = np.ones(4) * 0.5
    p = Polytope(A, b)
    m = np.array([[2, 1], [-1, 2]])  # Invertible matrix
    
    p_res = m @ p
    
    # Expected new A is A_old @ inv(M)
    m_inv = np.linalg.inv(m)
    A_expected = A @ m_inv
    
    assert isinstance(p_res, Polytope)
    assert np.allclose(p_res.A, A_expected)
    assert np.allclose(p_res.b, b.reshape(-1, 1)) # b should be unchanged

def test_mtimes_h_rep_non_invertible_error():
    """Test that H-rep multiplication with a non-invertible matrix raises an error."""
    A = np.eye(2)
    b = np.ones(2)
    p = Polytope(A, b)
    m_non_invertible = np.array([[1, 1], [1, 1]])
    
    with pytest.raises(NotImplementedError):
        m_non_invertible @ p 