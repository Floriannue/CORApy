import pytest
import numpy as np
from cora_python.contSet import Interval

def test_enclose_and_convhull():
    """
    Test that enclose() and convHull_() produce the same result as union (|).
    This test covers both functions since enclose() simply calls convHull_(),
    and convHull_() for intervals calls the union operator.
    """
    I1 = Interval([-1, 2], [1, 3])
    I2 = Interval([0, 1], [2, 4])
    
    # Expected result from union
    # inf = min(I1.inf, I2.inf) = min([-1, 2], [0, 1]) = [-1, 1]
    # sup = max(I1.sup, I2.sup) = max([1, 3], [2, 4]) = [2, 4]
    expected_inf = np.array([-1, 1])
    expected_sup = np.array([2, 4])
    
    # --- Test convHull_ ---
    res_convhull = I1.convHull_(I2)
    assert np.allclose(res_convhull.inf, expected_inf)
    assert np.allclose(res_convhull.sup, expected_sup)
    
    # --- Test enclose ---
    res_enclose = I1.enclose(I2)
    assert np.allclose(res_enclose.inf, expected_inf)
    assert np.allclose(res_enclose.sup, expected_sup)
    
    # --- Test union operator for sanity check ---
    res_union = I1 | I2
    assert np.allclose(res_union.inf, expected_inf)
    assert np.allclose(res_union.sup, expected_sup)

def test_enclose_with_numeric():
    """Test enclose with a numeric type, which should be converted to an interval."""
    I1 = Interval([-1, 2], [1, 3])
    p = np.array([0, 5]) # A point
    
    # Expected result
    # inf = min([-1, 2], [0, 5]) = [-1, 2]
    # sup = max([1, 3], [0, 5]) = [1, 5]
    expected_inf = np.array([-1, 2])
    expected_sup = np.array([1, 5])
    
    res_convhull = I1.convHull_(p)
    assert np.allclose(res_convhull.inf, expected_inf)
    assert np.allclose(res_convhull.sup, expected_sup)
    
    # Enclose requires two contSet objects, so direct numeric test is not applicable
    # but we test it via convHull_ which is its core.

def test_empty_set_handling():
    """Test that enclosing with an empty set returns the other set."""
    I1 = Interval([-1, 2], [1, 3])
    I_empty = Interval.empty(2)
    
    res1 = I1.enclose(I_empty)
    assert res1 == I1
    
    res2 = I_empty.enclose(I1)
    assert res2 == I1 