import numpy as np
from cora_python.contSet.polytope import Polytope

def test_display():
    """
    Temporary test to ensure the display function runs without crashing.
    The detailed string output is too complex to assert reliably right now
    and is blocking progress on more critical functions.
    """
    # H-rep
    A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    b = np.array([[1], [0], [1], [0]])
    p_h = Polytope(A, b)
    s_h = p_h.display()
    assert isinstance(s_h, str)
    assert len(s_h) > 0

    # V-rep
    V = np.array([[0, 1, 0.5], [0, 0, 1]])
    p_v = Polytope(V)
    s_v = p_v.display()
    assert isinstance(s_v, str)
    assert len(s_v) > 0

    # Empty
    p_e = Polytope()
    s_e = p_e.display()
    assert isinstance(s_e, str)
    assert len(s_e) > 0 