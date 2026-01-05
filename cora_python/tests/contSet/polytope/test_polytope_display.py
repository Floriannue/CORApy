import numpy as np
import io
import sys
from cora_python.contSet.polytope import Polytope

def test_display():
    """
    Test to ensure the display function runs without crashing and follows the pattern.
    """
    # H-rep
    A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    b = np.array([[1], [0], [1], [0]])
    p_h = Polytope(A, b)
    s_h = p_h.display_()
    assert isinstance(s_h, str)
    assert len(s_h) > 0
    
    # Test that display() prints the same
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    try:
        p_h.display()
        printed_output = buffer.getvalue()
        assert printed_output == s_h
    finally:
        sys.stdout = old_stdout
    
    # Test that __str__ returns the same
    assert str(p_h) == s_h

    # V-rep
    V = np.array([[0, 1, 0.5], [0, 0, 1]])
    p_v = Polytope(V)
    s_v = p_v.display_()
    assert isinstance(s_v, str)
    assert len(s_v) > 0

    # Empty
    p_e = Polytope.empty(0) # Use .empty() for explicit empty polytope
    s_e = p_e.display_()
    assert isinstance(s_e, str)
    assert len(s_e) > 0

    # Unbounded (Inf)
    p_inf = Polytope.Inf(2) # Use .Inf() for explicit unbounded polytope
    s_inf = p_inf.display_()
    assert isinstance(s_inf, str)
    assert len(s_inf) > 0 