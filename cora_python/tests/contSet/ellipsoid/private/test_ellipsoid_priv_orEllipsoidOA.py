import numpy as np
from cora_python.contSet.ellipsoid import Ellipsoid
from cora_python.contSet.ellipsoid.private.priv_orEllipsoidOA import priv_orEllipsoidOA


def test_priv_orEllipsoidOA_basic():
    E1 = Ellipsoid(np.eye(2), np.zeros((2,1)))
    E2 = Ellipsoid(0.5*np.eye(2), np.zeros((2,1)))
    E = priv_orEllipsoidOA([E1, E2])
    assert isinstance(E, Ellipsoid)


