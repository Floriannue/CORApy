import numpy as np

from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.zonoBundle import ZonoBundle


def test_zonoBundle_reduce_returns_bundle():
    """
    Test zonoBundle.reduce reduces each zonotope without errors.
    """
    z = Zonotope(np.array([[0.0], [0.0]]), np.array([[1.0, 0.0], [0.0, 1.0]]))
    zB = ZonoBundle([z])

    res = zB.reduce('girard', 1)

    assert res.parallelSets == zB.parallelSets
