import numpy as np
from cora_python.contSet.polytope.polytope import Polytope
from cora_python.contSet.polytope.copy import copy

def test_copy():
    # Create a polytope
    A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    b = np.array([1, 0, 1, 0])
    P = Polytope(A, b)

    # Copy the polytope
    P_copy = copy(P)

    # Check that the copied polytope is a different object
    assert P is not P_copy

    # Check that the properties of the copied polytope are the same
    assert np.array_equal(P.A, P_copy.A)
    assert np.array_equal(P.b, P_copy.b)

    # Check with V-representation
    V = np.array([[0,0],[1,0],[0,1]]).T
    P_v = Polytope(V)
    P_v_copy = copy(P_v)
    assert P_v is not P_v_copy
    assert np.array_equal(P_v.V, P_v_copy.V) 