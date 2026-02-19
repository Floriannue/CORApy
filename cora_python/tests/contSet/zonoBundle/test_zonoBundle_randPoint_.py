import numpy as np

from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.zonoBundle import ZonoBundle


def _sort_vertices(V: np.ndarray) -> np.ndarray:
    if V.size == 0:
        return V
    order = np.lexsort(V[::-1, :])
    return V[:, order]


def test_zonoBundle_randPoint_all_extreme_equals_vertices():
    """Generated test: randPoint_('all','extreme') returns vertices."""
    z = Zonotope(np.array([[0.0], [0.0]]), np.array([[1.0, 0.0], [0.0, 1.0]]))
    zB = ZonoBundle([z])

    V = _sort_vertices(zB.vertices_())
    P = _sort_vertices(zB.randPoint_('all', 'extreme'))

    assert P.shape == V.shape
    assert np.allclose(P, V)


def test_zonoBundle_randPoint_standard_shape():
    """Generated test: randPoint_('standard') returns correct shape."""
    np.random.seed(0)
    z = Zonotope(np.array([[0.0], [0.0]]), np.array([[1.0, 0.0], [0.0, 1.0]]))
    zB = ZonoBundle([z])

    pts = zB.randPoint_(5, 'standard')
    assert pts.shape == (2, 5)

