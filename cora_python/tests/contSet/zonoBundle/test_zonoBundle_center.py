import numpy as np

from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.zonoBundle import ZonoBundle


def test_zonoBundle_center_matches_conZonotope():
    """Generated test: center uses conZonotope center when non-empty."""
    z1 = Zonotope(np.array([[0.0], [0.0]]), np.array([[1.0, 0.0], [0.0, 1.0]]))
    z2 = Zonotope(np.array([[0.2], [0.2]]), np.array([[1.0, 0.0], [0.0, 1.0]]))
    zB = ZonoBundle([z1, z2])

    c_bundle = zB.center()
    c_cz = zB.conZonotope().center()

    assert c_bundle.shape == c_cz.shape
    assert np.allclose(c_bundle, c_cz)

