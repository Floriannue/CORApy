import numpy as np

from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.zonoBundle import ZonoBundle


def _sort_vertices(V: np.ndarray) -> np.ndarray:
    if V.size == 0:
        return V
    order = np.lexsort(V[::-1, :])
    return V[:, order]


def test_zonoBundle_vertices_single_matches_zonotope():
    """Generated test: vertices_ for single set matches zonotope vertices."""
    z = Zonotope(np.array([[1.0], [2.0]]), np.array([[0.5, 0.0], [0.0, 0.25]]))
    zB = ZonoBundle([z])

    Vb = _sort_vertices(zB.vertices_())
    Vz = _sort_vertices(z.vertices_())

    assert Vb.shape == Vz.shape
    assert np.allclose(Vb, Vz)

