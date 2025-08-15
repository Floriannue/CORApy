import numpy as np
import pytest

from cora_python.contSet.polytope import Polytope
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def test_polytope_zonoBundle_bounded_square():
    A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    b = np.array([[1], [1], [1], [1]])
    P = Polytope(A, b)
    zB = Polytope.zonoBundle(P)
    # Bundle should have at least one zonotope
    assert hasattr(zB, 'Z')
    assert len(zB.Z) > 0


def test_polytope_zonoBundle_unbounded_raises():
    P_inf = Polytope.Inf(2)
    with pytest.raises(CORAerror):
        Polytope.zonoBundle(P_inf)


