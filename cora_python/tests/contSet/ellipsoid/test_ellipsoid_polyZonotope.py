import numpy as np
from cora_python.contSet.ellipsoid import Ellipsoid
from cora_python.contSet.polyZonotope import PolyZonotope


def test_polyZonotope_basic():
    Q = np.array([[2.0, 0.5],[0.5, 1.0]])
    q = np.array([[0.1],[0.2]])
    E = Ellipsoid(Q, q)
    PZ = E.polyZonotope()
    assert isinstance(PZ, PolyZonotope)

