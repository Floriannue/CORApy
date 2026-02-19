import numpy as np

from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.zonoBundle import ZonoBundle
from cora_python.contSet.interval import Interval


def test_zonoBundle_split_one_dim_interval_bounds():
    """
    Test zonoBundle.split for a specific dimension returns correct interval bounds.
    """
    z0 = Zonotope(np.array([[1.4], [2.3]]), np.array([[0.3, 0.0], [0.0, 0.05]]))
    zB = ZonoBundle([z0])

    # Python uses 0-based indexing for split dimension
    zsplit = zB.split(0)

    assert len(zsplit) == 2
    assert zsplit[0].parallelSets == 2
    assert zsplit[1].parallelSets == 2

    # Expected bounds after splitting x1 at midpoint 1.4
    left_lb = np.array([[1.1], [2.25]])
    left_ub = np.array([[1.4], [2.35]])
    right_lb = np.array([[1.4], [2.25]])
    right_ub = np.array([[1.7], [2.35]])

    I_left = zsplit[0].interval()
    I_right = zsplit[1].interval()

    assert np.allclose(I_left.infimum(), left_lb)
    assert np.allclose(I_left.supremum(), left_ub)
    assert np.allclose(I_right.infimum(), right_lb)
    assert np.allclose(I_right.supremum(), right_ub)
