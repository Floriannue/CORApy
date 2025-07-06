import pytest
import numpy as np

from cora_python.contSet import Interval

def test_cat_vert():
    # Test concatenation along dimension 1 (vertical)
    i1 = Interval(np.array([-0.5]), np.array([0.5]))
    i2 = Interval(np.array([0.3]), np.array([0.7]))
    
    res = Interval.cat(1, i1, i2)
    
    expected_inf = np.array([[-0.5], [0.3]])
    expected_sup = np.array([[0.5], [0.7]])
    
    # Reshape for comparison as cat(1,...) on scalars creates column vector
    assert np.allclose(res.inf, expected_inf)
    assert np.allclose(res.sup, expected_sup)

def test_cat_horz():
    # Test concatenation along dimension 2 (horizontal)
    # Example from MATLAB file
    # I1 = interval([-0.5;0.3]);
    # I2 = interval([2;3]);
    # I = cat(2,I1,I2);
    i1 = Interval(np.array([[-0.5], [0.3]]))
    i2 = Interval(np.array([[2], [3]]))
    
    res = Interval.cat(2, i1, i2)
    
    expected_inf = np.array([[-0.5, 2], [0.3, 3]])
    expected_sup = np.array([[-0.5, 2], [0.3, 3]]) # In this example, sup is same as inf for point intervals
    
    # The example in MATLAB doc is a bit misleading, let's make a better one.
    i1 = Interval(np.array([[-1], [1]]), np.array([[0], [2]]))
    i2 = Interval(np.array([[5], [7]]), np.array([[6], [8]]))
    res = Interval.cat(2, i1, i2)
    
    expected_inf_2 = np.array([[-1, 5], [1, 7]])
    expected_sup_2 = np.array([[0, 6], [2, 8]])
    
    assert np.allclose(res.inf, expected_inf_2)
    assert np.allclose(res.sup, expected_sup_2)
    
def test_cat_with_numeric():
    i1 = Interval(1, 2)
    res = Interval.cat(2, i1, 5, Interval(3,4))
    
    expected_inf = np.array([[1, 5, 3]])
    expected_sup = np.array([[2, 5, 4]])
    
    assert np.allclose(res.inf, expected_inf)
    assert np.allclose(res.sup, expected_sup)

def test_cat_dim_mismatch():
    i1 = Interval(np.array([1,2])) # shape (2,) -> treated as (1,2) for cat 2
    i2 = Interval(np.array([[3],[4]])) # shape (2,1)
    
    with pytest.raises(ValueError):
        Interval.cat(2, i1, i2) 