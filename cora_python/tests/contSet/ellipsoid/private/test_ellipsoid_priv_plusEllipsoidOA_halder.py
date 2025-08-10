import numpy as np
from cora_python.contSet.ellipsoid import Ellipsoid
from cora_python.contSet.ellipsoid.private.priv_plusEllipsoidOA_halder import priv_plusEllipsoidOA_halder


def test_priv_plusEllipsoidOA_halder_basic():
    E1 = Ellipsoid(np.eye(2), np.zeros((2,1)))
    E2 = Ellipsoid(2*np.eye(2), np.zeros((2,1)))
    E = priv_plusEllipsoidOA_halder([E1,E2])
    assert isinstance(E, Ellipsoid)


