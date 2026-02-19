import numpy as np

from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.zonoBundle import ZonoBundle


def test_zonoBundle_cartProd_with_zonotope():
    """
    Test zonoBundle.cartProd_ with a zonotope.
    """
    z = Zonotope(np.array([[1.0], [2.0]]), np.array([[0.1], [0.2]]))
    zB = ZonoBundle([z])

    z2 = Zonotope(np.array([[0.5]]), np.array([[0.05]]))
    res = zB.cartProd_(z2, 'exact')

    assert res.parallelSets == 1
    assert res.Z[0].dim() == z.dim() + z2.dim()
