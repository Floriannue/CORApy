import pytest
import numpy as np

from cora_python.contSet import Interval

def test_ctranspose():
    # Example from MATLAB file
    # I = interval([-1;-2],[3;4]);
    # I'
    
    inf = np.array([[-1], [-2]])
    sup = np.array([[3], [4]])
    i = Interval(inf, sup)
    
    res = i.ctranspose()
    
    expected_inf = np.array([[-1, -2]])
    expected_sup = np.array([[3, 4]])
    
    assert np.array_equal(res.inf, expected_inf)
    assert np.array_equal(res.sup, expected_sup)
    
def test_ctranspose_is_transpose():
    inf = np.array([[-1, 5], [-2, 8]])
    sup = np.array([[3, 6], [4, 9]])
    i = Interval(inf, sup)
    
    res_ct = i.ctranspose()
    res_t = i.transpose()
    
    assert np.array_equal(res_ct.inf, res_t.inf)
    assert np.array_equal(res_ct.sup, res_t.sup) 