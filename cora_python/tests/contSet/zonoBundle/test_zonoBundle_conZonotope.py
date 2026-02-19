import numpy as np

from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.zonoBundle import ZonoBundle
from cora_python.contSet.conZonotope import ConZonotope


def test_zonoBundle_conZonotope_single_matches_zonotope():
    """Generated test: conZonotope of single-zonotope bundle matches."""
    z = Zonotope(np.array([[1.0], [2.0]]), np.array([[0.5, 0.0], [0.0, 0.25]]))
    zB = ZonoBundle([z])

    cZ = zB.conZonotope()

    assert isinstance(cZ, ConZonotope)
    assert np.allclose(cZ.c, z.c)
    assert np.allclose(cZ.G, z.G)

