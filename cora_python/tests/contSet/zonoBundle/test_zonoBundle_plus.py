import numpy as np

from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.zonoBundle import ZonoBundle


def test_zonoBundle_plus_vector_translates_all_sets():
    """
    Test zonoBundle + vector shifts each zonotope by the vector.
    """
    z = Zonotope(np.array([[1.0], [2.0]]), np.array([[0.1, 0.0], [0.0, 0.2]]))
    zB = ZonoBundle([z])

    shift = np.array([[0.5], [-0.5]])
    res = zB + shift

    assert res.parallelSets == 1
    assert np.allclose(res.Z[0].c, z.c + shift)
