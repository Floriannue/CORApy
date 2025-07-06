import pytest
import numpy as np
from cora_python.contSet.interval.interval import Interval
from cora_python.contSet.polytope.polytope import Polytope

def test_interval_polytope_bounded():
    # Bounded case
    I = Interval(np.array([[-1], [0]]), np.array([[1], [3]]))
    P = I.polytope()

    assert isinstance(P, Polytope)
    
    # Expected A and b
    expected_A = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    expected_b = np.array([[1], [3], [1], [0]])
    
    assert np.allclose(P.A, expected_A)
    assert np.allclose(P.b, expected_b)

def test_interval_polytope_unbounded():
    # Unbounded case
    I = Interval(np.array([[-np.inf], [0]]), np.array([[1], [np.inf]]))
    P = I.polytope()

    assert isinstance(P, Polytope)
    
    # Expected A and b (unbounded directions removed)
    # b = [sup; -inf] -> b = [1, inf, inf, 0]'
    # A = [I; -I] = [[1,0],[0,1];[-1,0],[0,-1]]
    # Rows 2 and 3 are removed from A and b
    expected_A = np.array([[1, 0], [0, -1]])
    expected_b = np.array([[1], [0]])
    
    assert np.allclose(P.A, expected_A)
    assert np.allclose(P.b, expected_b) 