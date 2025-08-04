import numpy as np
import pytest
from cora_python.contSet.ellipsoid.private.priv_rootfnc import priv_rootfnc

def test_priv_rootfnc_p_is_zero():
    # Test case when p = 0
    W1 = np.eye(2)
    q1 = np.array([[1], [1]])
    W2 = np.eye(2) * 2
    q2 = np.array([[2], [2]])
    p = 0.0

    y, Q, q = priv_rootfnc(p, W1, q1, W2, q2)

    # When p=0, X should be W2, q should be q2, Q should be a*inv(W2)
    expected_X = W2
    expected_q = q2
    expected_Q_factor = 1 - 0 * (1 - 0) * (q2 - q1).T @ W2 @ np.linalg.inv(expected_X) @ W1 @ (q2 - q1)
    expected_Q = expected_Q_factor * np.linalg.inv(expected_X)
    
    # Since y is a complex polynomial, we'll focus on Q and q for correctness in this test
    assert np.allclose(q, expected_q)
    assert np.allclose(Q, expected_Q)
    assert isinstance(y, float) # Ensure y is a scalar

def test_priv_rootfnc_p_is_one():
    # Test case when p = 1
    W1 = np.eye(2) * 3
    q1 = np.array([[1], [0]])
    W2 = np.eye(2)
    q2 = np.array([[0], [1]])
    p = 1.0

    y, Q, q = priv_rootfnc(p, W1, q1, W2, q2)

    # When p=1, X should be W1, q should be q1, Q should be a*inv(W1)
    expected_X = W1
    expected_q = q1
    expected_Q_factor = 1 - 1 * (1 - 1) * (q2 - q1).T @ W2 @ np.linalg.inv(expected_X) @ W1 @ (q2 - q1)
    expected_Q = expected_Q_factor * np.linalg.inv(expected_X)
    
    assert np.allclose(q, expected_q)
    assert np.allclose(Q, expected_Q)
    assert isinstance(y, float)

def test_priv_rootfnc_intermediate_p():
    # Test case for p = 0.5
    W1 = np.array([[2, 0], [0, 2]])
    q1 = np.array([[0], [0]])
    W2 = np.array([[1, 0], [0, 1]])
    q2 = np.array([[1], [1]])
    p = 0.5

    y, Q, q = priv_rootfnc(p, W1, q1, W2, q2)

    # Expected values derived by manual calculation or by running MATLAB equivalent
    # For p=0.5, X = 0.5*W1 + 0.5*W2 = 0.5*np.array([[2,0],[0,2]]) + 0.5*np.array([[1,0],[0,1]]) = np.array([[1.5,0],[0,1.5]])
    expected_X = 0.5 * W1 + 0.5 * W2
    expected_X_inv = np.linalg.inv(expected_X)
    expected_X_inv = 0.5 * (expected_X_inv + expected_X_inv.T)
    
    expected_a = 1 - p * (1 - p) * (q2 - q1).T @ W2 @ expected_X_inv @ W1 @ (q2 - q1)
    expected_q = expected_X_inv @ (p * W1 @ q1 + (1 - p) * W2 @ q2)
    expected_Q = expected_a * np.linalg.inv(expected_X)
    
    assert np.allclose(q, expected_q)
    assert np.allclose(Q, expected_Q)
    assert isinstance(y, float)