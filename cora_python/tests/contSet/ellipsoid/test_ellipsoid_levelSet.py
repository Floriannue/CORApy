import numpy as np

from cora_python.contSet.ellipsoid import Ellipsoid
from cora_python.contSet.levelSet import LevelSet


def test_levelSet_basic():
    Q = np.array([[4.0, 2.0], [2.0, 4.0]])
    q = np.array([[1.0], [1.0]])
    E = Ellipsoid(Q, q)
    ls = E.levelSet()
    assert isinstance(ls, LevelSet)
    assert ls.compOp in ['<=']

