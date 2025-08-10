import numpy as np

from cora_python.contSet.ellipsoid import Ellipsoid
from cora_python.contSet.conPolyZono import ConPolyZono


def test_conPolyZono_basic():
    Q = np.array([[4.0, 2.0], [2.0, 4.0]])
    q = np.array([[1.0], [1.0]])
    E = Ellipsoid(Q, q)
    cPZ = E.conPolyZono()
    assert isinstance(cPZ, ConPolyZono)
    # structural properties
    assert cPZ.c.shape[0] == 2
    assert cPZ.G.shape[0] == 2
    assert cPZ.A.shape[0] >= 0 and cPZ.b.shape[0] == cPZ.A.shape[0]

