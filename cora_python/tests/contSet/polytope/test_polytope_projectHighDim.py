import numpy as np

from cora_python.contSet.polytope import Polytope
from cora_python.g.functions.matlab.validate.check.compareMatrices import compareMatrices
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def test_polytope_projectHighDim_empty_and_reorder():
    # 1D, fully empty (no constraints -> fullspace) mapped to N=3 fixing others
    A = np.zeros((0, 1)); b = np.zeros((0, 1))
    P = Polytope(A, b)
    P_high = Polytope.projectHighDim(P, 3, [2])
    Ae_true = np.array([[1, 0, 0], [0, 0, 1]])
    be_true = np.array([[0.0], [0.0]])
    P_true = Polytope(np.zeros((0, 3)), np.zeros((0, 1)), Ae_true, be_true)
    assert Polytope.isequal(P_high, P_true)

    # 2D V-rep, vertex instantiation
    V = np.array([[2, -1, 0], [1, 0, -2]])
    P = Polytope(V)
    P_high = Polytope.projectHighDim(P, 3, [3, 1])
    V_true = np.array([[1, 0, -2], [0, 0, 0], [2, -1, 0]])
    assert compareMatrices(P_high.V, V_true)


def test_polytope_projectHighDim_errors():
    P = Polytope(np.array([[1.0]]), np.array([[1.0]]))
    try:
        Polytope.projectHighDim(P, 0)
        assert False
    except CORAerror:
        pass

    try:
        Polytope.projectHighDim(P, 2, [1, 2, 3])
        assert False
    except CORAerror:
        pass


