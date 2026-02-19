import numpy as np

from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.zonoBundle import ZonoBundle


def test_zonoBundle_enclose_with_zonotope():
    """
    Test zonoBundle.enclose with a zonotope applies to each member.
    """
    z1 = Zonotope(np.array([[0.0], [0.0]]), np.array([[1.0, 0.0], [0.0, 1.0]]))
    zB = ZonoBundle([z1])

    z2 = Zonotope(np.array([[1.0], [1.0]]), np.array([[0.5, 0.0], [0.0, 0.5]]))
    res = zB.enclose(z2)

    assert res.parallelSets == 1
    # enclose should return a zonotope; check center dimension
    assert res.Z[0].c.shape == z1.c.shape
