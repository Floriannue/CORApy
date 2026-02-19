import numpy as np

from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.zonoBundle import ZonoBundle


def _sort_vertices(V: np.ndarray) -> np.ndarray:
    if V.size == 0:
        return V
    order = np.lexsort(V[::-1, :])
    return V[:, order]


def test_zonoBundle_polytope_single_matches_zonotope_vertices():
    """Generated test: polytope of single-zonotope bundle matches vertices."""
    z = Zonotope(np.array([[0.0], [0.0]]), np.array([[1.0, 0.0], [0.0, 2.0]]))
    zB = ZonoBundle([z])

    P = zB.polytope()
    Vp = _sort_vertices(P.vertices_())
    Vz = _sort_vertices(z.vertices_())

    assert Vp.shape == Vz.shape
    assert np.allclose(Vp, Vz)

