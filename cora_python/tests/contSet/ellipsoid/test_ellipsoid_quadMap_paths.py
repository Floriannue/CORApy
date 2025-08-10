import numpy as np
from cora_python.contSet.zonotope import Zonotope

#i dont why this test exists, looks like a zonotope test
def test_quadMap_single_center_matches_no_generators():
    # If no generators, center is x^T Q x for x=c
    c = np.array([[1.0], [2.0]])
    Z = Zonotope(c, np.zeros((2, 0)))
    Q = [np.eye(2), np.array([[0.0, 1.0], [1.0, 0.0]])]
    Zs = Z.quadMap(Q)
    # Expected: [c^T I c, c^T offdiag c] = [1^2+2^2, 2*1*2] = [5, 4]
    assert np.allclose(Zs.c.flatten(), np.array([5.0, 4.0]))


def test_quadMap_mixed_center_matches_no_generators():
    c1 = np.array([[1.0], [2.0]])
    c2 = np.array([[3.0], [4.0]])
    Z1 = Zonotope(c1, np.zeros((2, 0)))
    Z2 = Zonotope(c2, np.zeros((2, 0)))
    Q = [np.eye(2), np.array([[0.0, 1.0], [1.0, 0.0]])]
    Zm = Z1.quadMap(Z2, Q)
    # Expected: [c1^T I c2, c1^T offdiag c2] = [1*3+2*4, 1*4+2*3] = [11, 10]
    assert np.allclose(Zm.c.flatten(), np.array([11.0, 10.0]))


