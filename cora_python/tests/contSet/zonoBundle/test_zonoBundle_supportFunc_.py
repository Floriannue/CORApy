import numpy as np

from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.zonoBundle import ZonoBundle


def test_zonoBundle_supportFunc_matches_zonotope_upper():
    """Generated test: support function for single set matches zonotope."""
    z = Zonotope(np.array([[0.0], [0.0]]), np.array([[1.0, 0.5], [0.0, 1.0]]))
    zB = ZonoBundle([z])

    direction = np.array([[1.0], [0.0]])
    val_b, _ = zB.supportFunc_(direction, 'upper')
    val_z, _, _ = z.supportFunc_(direction, 'upper')

    assert np.isclose(val_b, val_z)

