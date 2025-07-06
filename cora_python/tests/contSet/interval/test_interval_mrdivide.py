import pytest
import numpy as np

from cora_python.contSet import Interval
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def test_mrdivide_scalar_denominator():
    # Test case from MATLAB example
    # numerator = interval([-2;1],[3;2]);
    # denominator = 2;
    # numerator / denominator
    
    inf = np.array([[-2], [1]])
    sup = np.array([[3], [2]])
    i1 = Interval(inf, sup)
    
    res = i1.mrdivide(2)
    
    expected_inf = np.array([[-1], [0.5]])
    expected_sup = np.array([[1.5], [1]])
    
    assert np.allclose(res.inf, expected_inf)
    assert np.allclose(res.sup, expected_sup)

def test_mrdivide_scalar_interval_denominator():
    inf = np.array([[-2], [1]])
    sup = np.array([[3], [2]])
    i1 = Interval(inf, sup)
    
    i2 = Interval(2, 2)
    
    res = i1.mrdivide(i2)
    
    expected_inf = np.array([[-1], [0.5]])
    expected_sup = np.array([[1.5], [1]])
    
    assert np.allclose(res.inf, expected_inf)
    assert np.allclose(res.sup, expected_sup)
    
def test_mrdivide_non_scalar_denominator():
    inf = np.array([[-2], [1]])
    sup = np.array([[3], [2]])
    i1 = Interval(inf, sup)
    
    i2 = Interval(np.array([1, 2]), np.array([1, 2]))

    with pytest.raises(CORAerror) as e:
        i1.mrdivide(i2)
    assert e.value.identifier == "CORA:noops"

def test_mrdivide_operator_overload():
    # In MATLAB, '/' is mrdivide. In Python, we can't have both mrdivide and rdivide on '/'
    # The __truediv__ is mapped to rdivide which is more general.
    # So we don't test the operator here, but the method call.
    # However, we can test that the method call is equivalent to the operator call
    # if the denominator is a scalar.
    
    inf = np.array([[-2], [1]])
    sup = np.array([[3], [2]])
    i1 = Interval(inf, sup)
    
    res_method = i1.mrdivide(2)
    res_operator = i1 / 2
    
    assert np.allclose(res_method.inf, res_operator.inf)
    assert np.allclose(res_method.sup, res_operator.sup) 