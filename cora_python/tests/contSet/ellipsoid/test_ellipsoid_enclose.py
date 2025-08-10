import numpy as np
from cora_python.contSet.ellipsoid import Ellipsoid


def test_enclose_pair():
    E1 = Ellipsoid(np.eye(2), np.array([[0.0],[0.0]]))
    E2 = Ellipsoid(2*np.eye(2), np.array([[1.0],[1.0]]))
    E = E1.enclose(E2)
    assert isinstance(E, Ellipsoid)
    # center should be somewhere between q1 and q2 due to convex hull
    assert E.q.shape == (2,1)


def test_enclose_affine():
    E1 = Ellipsoid(np.eye(2), np.zeros((2,1)))
    M = np.array([[2.0,0.0],[0.0,3.0]])
    Eplus = Ellipsoid(np.eye(2), np.array([[1.0],[0.0]]))
    E = E1.enclose(M, Eplus)
    assert isinstance(E, Ellipsoid)
    assert E.Q.shape == (2,2)

