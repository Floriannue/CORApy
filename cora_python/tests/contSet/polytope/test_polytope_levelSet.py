import numpy as np
import pytest
import sympy as sp
from cora_python.contSet.polytope.polytope import Polytope
from cora_python.contSet.levelSet.levelSet import LevelSet
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def test_levelset_basic_polytope():
    """Test basic conversion from Polytope to LevelSet."""
    A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    b = np.array([[1], [1], [1], [1]])
    P = Polytope(A, b)

    ls = P.levelSet()

    assert isinstance(ls, LevelSet)
    assert ls.dim == 2
    assert ls.compOp == ['<=', '<=', '<=', '<=']

    # Verify symbolic equations (example for x0 <= 1)
    x0, x1 = sp.symbols('x0 x1')
    expected_eq_0 = x0 - 1.0
    assert ls.eq[0] == expected_eq_0
    assert ls.eq[1] == -x0 - 1.0
    assert ls.eq[2] == x1 - 1.0
    assert ls.eq[3] == -x1 - 1.0
    assert ls.vars == [x0, x1]

def test_levelset_polytope_with_equality_constraints():
    """Test conversion of Polytope with equality constraints to LevelSet (should raise error)."""
    A = np.array([[1, 0]])
    b = np.array([[1]])
    Ae = np.array([[0, 1]])
    be = np.array([[0]])
    P = Polytope(A, b, Ae=Ae, be=be)

    with pytest.raises(CORAerror) as excinfo:
        P.levelSet()
    assert "Equality constraints not supported" in str(excinfo.value)

def test_levelset_empty_polytope():
    """Test conversion of an empty Polytope to LevelSet."""
    P = Polytope.empty(2)

    ls = P.levelSet()

    assert isinstance(ls, LevelSet)
    assert ls.dim == 2
    # For an empty polytope, if it has no inequality constraints, eq_sym will be 0
    # The LevelSet constructor should handle this, potentially with a default equality.
    # Based on the implementation, it should yield sp.Integer(0) for eq_sym
    assert ls.eq == sp.Float(1.0) # Expect 1.0 for infeasible constraint of empty set
    assert ls.vars == [sp.symbols('x0'), sp.symbols('x1')]
    assert ls.compOp == '<=' # Default for single equation, even if it's 0
