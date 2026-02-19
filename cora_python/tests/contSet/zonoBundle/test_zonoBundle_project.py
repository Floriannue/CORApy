import numpy as np

from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.zonoBundle import ZonoBundle


def test_zonoBundle_project_reduces_dimension():
    """Generated test: basic dimension reduction via project()."""
    z1 = Zonotope(np.zeros((3, 1)), np.eye(3))
    z2 = Zonotope(np.ones((3, 1)), np.eye(3))
    zB = ZonoBundle([z1, z2])

    res = zB.project([0, 2])

    assert res.dim() == 2
    assert res.Z[0].c.shape == (2, 1)
    assert res.Z[1].c.shape == (2, 1)

