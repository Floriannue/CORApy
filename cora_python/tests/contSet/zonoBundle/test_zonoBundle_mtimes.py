import numpy as np

from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.zonoBundle import ZonoBundle


def test_zonoBundle_mtimes_matrix_left():
    """
    Test matrix * zonoBundle multiplies each zonotope.
    """
    z = Zonotope(np.array([[1.0], [2.0]]), np.array([[0.1, 0.0], [0.0, 0.2]]))
    zB = ZonoBundle([z])

    M = np.array([[2.0, 0.0], [0.0, 3.0]])
    res = M @ zB

    assert res.parallelSets == 1
    assert np.allclose(res.Z[0].c, M @ z.c)


def test_zonoBundle_mtimes_scalar_right():
    """
    Test zonoBundle * scalar multiplies each zonotope.
    """
    z = Zonotope(np.array([[1.0], [2.0]]), np.array([[0.1, 0.0], [0.0, 0.2]]))
    zB = ZonoBundle([z])

    res = zB * 2.0

    assert res.parallelSets == 1
    assert np.allclose(res.Z[0].c, z.c * 2.0)
