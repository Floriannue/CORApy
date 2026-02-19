import numpy as np

from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.zonoBundle import ZonoBundle
from cora_python.contSet.interval import Interval


def test_zonoBundle_and_interval_appends_zonotope():
    """
    Test zonoBundle.and_ with interval appends a converted zonotope.
    """
    z = Zonotope(np.array([[0.0], [0.0]]), np.eye(2))
    zB = ZonoBundle([z])

    I = Interval(np.array([[-1.0], [-2.0]]), np.array([[2.0], [1.0]]))
    res = zB.and_(I)

    assert res.parallelSets == 2

    expected = Zonotope(I)
    assert res.Z[-1].isequal(expected)
