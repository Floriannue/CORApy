import numpy as np
from cora_python.contSet.ellipsoid import Ellipsoid


def test_enclosePoints_basic():
    pts = np.array([[0.0, 1.0, -1.0], [0.0, 1.0, -1.0]])
    E = Ellipsoid(np.eye(2), np.zeros((2,1)))
    E2 = E.enclosePoints(pts)
    assert isinstance(E2, Ellipsoid)
    assert E2.Q.shape == (2,2)

