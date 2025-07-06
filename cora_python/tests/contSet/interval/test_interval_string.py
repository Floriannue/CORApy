import pytest
import numpy as np

from cora_python.contSet import Interval

def test_string():
    # Example from MATLAB file
    # I = interval([-1;-2],[2;1]);
    # str = string(I)
    
    inf = np.array([[-1], [-2]])
    sup = np.array([[2], [1]])
    i = Interval(inf, sup)
    
    str_array = i.string()
    
    expected_shape = (2, 1)
    assert str_array.shape == expected_shape
    
    expected_content = np.array([["[-1.0,2.0]"], ["[-2.0,1.0]"]], dtype=object)
    assert np.array_equal(str_array, expected_content)

def test_string_scalar():
    i = Interval(-1, 2)
    str_array = i.string()
    
    expected_shape = (1,)
    assert str_array.shape == expected_shape
    
    expected_content = np.array(["[-1.0,2.0]"], dtype=object)
    # np.array_equal needs arrays, so we wrap the scalar result
    assert np.array_equal(np.atleast_1d(str_array), np.atleast_1d(expected_content))

def test_string_matrix():
    inf = np.array([[1, 3], [5, 7]])
    sup = np.array([[2, 4], [6, 8]])
    i = Interval(inf, sup)
    
    str_array = i.string()
    
    expected_shape = (2, 2)
    assert str_array.shape == expected_shape
    
    expected_content = np.array([["[1.0,2.0]", "[3.0,4.0]"], ["[5.0,6.0]", "[7.0,8.0]"]], dtype=object)
    assert np.array_equal(str_array, expected_content) 