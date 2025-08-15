import pytest

from cora_python.contSet.polytope import Polytope
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def test_polytope_not_raises():
    with pytest.raises(CORAerror):
        Polytope.not_()


