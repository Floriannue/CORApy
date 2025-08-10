import numpy as np
from cora_python.contSet.ellipsoid import Ellipsoid
from cora_python.contSet.ellipsoid.private.priv_lplus import priv_lplus


def test_priv_lplus_cells():
    E1 = Ellipsoid(np.array([[2.0,0.0],[0.0,1.0]]), np.zeros((2,1)))
    E2 = Ellipsoid(np.array([[1.0,0.0],[0.0,0.5]]), np.zeros((2,1)))
    L = np.eye(2)
    cells = priv_lplus([E1, E2], L, 'outer')
    assert isinstance(cells, list) and len(cells) == L.shape[1]


