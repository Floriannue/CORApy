import numpy as np
from cora_python.contSet.ellipsoid import Ellipsoid
from cora_python.contSet.ellipsoid.private.priv_plusEllipsoidOA import priv_plusEllipsoidOA


def test_priv_plusEllipsoidOA_basic():
    E1 = Ellipsoid(np.eye(2), np.zeros((2,1)))
    E2 = Ellipsoid(2*np.eye(2), np.zeros((2,1)))
    E = priv_plusEllipsoidOA([E1,E2])
    assert isinstance(E, Ellipsoid)


