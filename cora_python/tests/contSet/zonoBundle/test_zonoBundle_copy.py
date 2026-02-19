import numpy as np

from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.zonoBundle import ZonoBundle


def test_zonoBundle_copy():
    """
    Test zonoBundle copy constructor preserves zonotopes.
    """
    z1 = Zonotope(np.array([[1.0], [1.0]]), np.array([[1.0, 1.0], [-1.0, 1.0]]))
    z2 = Zonotope(np.array([[-1.0], [1.0]]), np.array([[1.0, 0.0], [0.0, 1.0]]))
    zB = ZonoBundle([z1, z2])

    zB_out = zB.copy()

    assert zB_out.parallelSets == zB.parallelSets
    for idx in range(zB.parallelSets):
        assert zB_out.Z[idx].isequal(zB.Z[idx])
