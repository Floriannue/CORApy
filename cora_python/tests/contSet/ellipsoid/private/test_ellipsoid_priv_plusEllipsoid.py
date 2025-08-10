import numpy as np
from cora_python.contSet.ellipsoid import Ellipsoid
from cora_python.contSet.ellipsoid.private.priv_plusEllipsoid import priv_plusEllipsoid


def test_priv_plusEllipsoid_outer():
    E1 = Ellipsoid(np.eye(2), np.zeros((2,1)))
    E2 = Ellipsoid(2*np.eye(2), np.zeros((2,1)))
    E = priv_plusEllipsoid([E1,E2], np.zeros((2,0)), 'outer')
    assert isinstance(E, Ellipsoid)


