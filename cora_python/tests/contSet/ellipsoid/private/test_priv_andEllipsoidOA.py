import numpy as np
from cora_python.contSet.ellipsoid import Ellipsoid
from cora_python.contSet.ellipsoid.private.priv_andEllipsoidOA import priv_andEllipsoidOA


def test_priv_andEllipsoidOA_basic():
    E1 = Ellipsoid(np.eye(2), np.zeros((2,1)))
    E2 = Ellipsoid(np.eye(2), np.array([[0.5],[0.0]]))
    E = priv_andEllipsoidOA(E1, E2)
    assert isinstance(E, Ellipsoid)


