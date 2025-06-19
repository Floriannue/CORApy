import numpy as np
from cora_python.contSet.polytope.private.priv_normalize_constraints import priv_normalize_constraints

def test_priv_normalize_constraints():
    # Test case 1: Normalize by 'b'/'be'
    A = np.array([[1., 2.], [3., 4.], [5., 6.]])
    b = np.array([2., -4., 0.])
    Ae = np.array([[7., 8.], [9., 10.]])
    be = np.array([5., 0.])
    
    A_res, b_res, Ae_res, be_res = priv_normalize_constraints(np.copy(A), np.copy(b), np.copy(Ae), np.copy(be), 'b')
    
    A_exp = np.array([[0.5, 1.], [-0.75, -1.], [5., 6.]])
    b_exp = np.array([1., -1., 0.])
    Ae_exp = np.array([[1.4, 1.6], [9., 10.]])
    be_exp = np.array([1., 0.])

    assert np.allclose(A_res, A_exp)
    assert np.allclose(b_res, b_exp)
    assert np.allclose(Ae_res, Ae_exp)
    assert np.allclose(be_res, be_exp)

    # Test case 2: Normalize by 'A'/'Ae'
    A = np.array([[3., 4.], [5., 12.]])
    b = np.array([5., 26.])
    Ae = np.array([[8., 6.], [0., 0.]])
    be = np.array([10., 5.])

    A_res, b_res, Ae_res, be_res = priv_normalize_constraints(np.copy(A), np.copy(b), np.copy(Ae), np.copy(be), 'A')
    
    A_exp = np.array([[0.6, 0.8], [5./13., 12./13.]])
    b_exp = np.array([1., 2.])
    Ae_exp = np.array([[0.8, 0.6], [0., 0.]])
    be_exp = np.array([1., 5.])

    assert np.allclose(A_res, A_exp)
    assert np.allclose(b_res, b_exp)
    assert np.allclose(Ae_res, Ae_exp)
    assert np.allclose(be_res, be_exp)

    # Test case 3: Empty matrices
    A_res, b_res, Ae_res, be_res = priv_normalize_constraints(np.array([]), np.array([]), np.array([]), np.array([]), 'b')
    assert A_res.size == 0
    assert b_res.size == 0
    assert Ae_res.size == 0
    assert be_res.size == 0

    A_res, b_res, Ae_res, be_res = priv_normalize_constraints(np.array([]), np.array([]), np.array([]), np.array([]), 'A')
    assert A_res.size == 0
    assert b_res.size == 0
    assert Ae_res.size == 0
    assert be_res.size == 0

    # Test case 4: All zero offsets for 'b' normalization
    A = np.array([[1., 2.], [3., 4.]])
    b = np.array([0., 0.])
    A_orig = np.copy(A)
    b_orig = np.copy(b)
    
    A_res, b_res, _, _ = priv_normalize_constraints(A, b, None, None, 'b')
    assert np.allclose(A_res, A_orig)
    assert np.allclose(b_res, b_orig)

    # Test case 5: All zero norm constraints for 'A' normalization
    A = np.array([[0., 0.], [0., 0.]])
    b = np.array([1., 2.])
    A_orig = np.copy(A)
    b_orig = np.copy(b)

    A_res, b_res, _, _ = priv_normalize_constraints(A, b, None, None, 'A')
    assert np.allclose(A_res, A_orig)
    assert np.allclose(b_res, b_orig) 