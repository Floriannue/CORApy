import numpy as np

from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.zonoBundle import ZonoBundle


def test_zonoBundle_quadMap_returns_bundle():
    """
    Test zonoBundle.quadMap applies quadMap to each zonotope.
    """
    z = Zonotope(np.array([[1.0], [0.0]]), np.array([[0.2, 0.0], [0.0, 0.1]]))
    zB = ZonoBundle([z])

    Q = [np.eye(2)]
    res = zB.quadMap(Q)

    assert res.parallelSets == 1
