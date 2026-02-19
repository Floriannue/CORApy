import numpy as np

from cora_python.contSet.conZonotope import ConZonotope


def test_conZonotope_isFullDim():
    # check empty conZonotope object
    cZ = ConZonotope.empty(2)
    res_subspace, subspace = cZ.isFullDim()
    assert not cZ.isFullDim()[0]
    assert not res_subspace
    assert subspace.size == 0

    # constrained zonotope
    Z = np.array([[0, 3, 0, 1], [0, 0, 2, 1]], dtype=float)
    A = np.array([[1, 0, 1]], dtype=float)
    b = np.array([[1]], dtype=float)
    cZ = ConZonotope(Z, A, b)
    res_subspace, subspace = cZ.isFullDim()
    assert cZ.isFullDim()[0]
    assert res_subspace
    assert subspace.shape[1] == 2

    # degenerate constrained zonotope
    Z = np.array([[0, 3, 0, 1], [1, 0, 0, 0]], dtype=float)
    A = np.array([[1, 0, 1]], dtype=float)
    b = np.array([[1]], dtype=float)
    cZ = ConZonotope(Z, A, b)
    res_subspace, subspace = cZ.isFullDim()
    true_subspace = np.array([[1.0], [0.0]])
    same_subspace = (np.linalg.matrix_rank(np.hstack([subspace, true_subspace]), tol=1e-6) ==
                     subspace.shape[1])
    assert not cZ.isFullDim()[0]
    assert not res_subspace
    assert same_subspace

