import numpy as np

from cora_python.contSet.conZonotope import ConZonotope


def test_conZonotope_and_conZonotope_generated():
    """Generated test: verify intersection construction formula."""
    c1 = np.array([[0.0], [0.0]])
    G1 = np.array([[1.0, 0.0], [0.0, 1.0]])
    A1 = np.zeros((0, 2))
    b1 = np.zeros((0, 1))
    cZ1 = ConZonotope(c1, G1, A1, b1)

    c2 = np.array([[1.0], [2.0]])
    G2 = np.array([[2.0, 0.0], [0.0, 3.0]])
    A2 = np.zeros((0, 2))
    b2 = np.zeros((0, 1))
    cZ2 = ConZonotope(c2, G2, A2, b2)

    res = cZ1.and_(cZ2)

    Z_expected = np.hstack([c1, G1, np.zeros_like(G2)])
    A_expected = np.hstack([G1, -G2])
    b_expected = c2 - c1

    assert np.allclose(res.c, Z_expected[:, 0:1])
    assert np.allclose(res.G, Z_expected[:, 1:])
    assert np.allclose(res.A, A_expected)
    assert np.allclose(res.b, b_expected)
