import numpy as np
import pytest
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def test_withinTol_scalar_basic():
    a = 3
    b = 3.1
    assert withinTol(a, a)
    assert not withinTol(a, b)
    assert withinTol(a, b, b - a)

def test_withinTol_scalar_inf():
    a = -np.inf
    b = np.inf
    assert withinTol(a, a)
    assert not withinTol(a, b)

def test_withinTol_scalar_vs_vector():
    a = 0
    b = np.array([0.001, 0.002, -0.001, 0, 0.003])
    result = withinTol(a, b)
    assert result.shape == b.shape
    assert np.all(withinTol(a, b, np.max(np.abs(b)) - a))

def test_withinTol_scalar_vs_vector_inf():
    a = -np.inf
    b = np.array([[-np.inf], [-np.inf]])
    result = withinTol(a, b)
    assert result.shape == b.shape
    assert np.all(withinTol(a, b))

def test_withinTol_vector_vs_vector():
    a = np.array([3, 4, 5])
    b = np.array([4, 5, 6])
    result = withinTol(a, b)
    assert result.shape == a.shape
    assert not np.any(withinTol(a, b))
    assert np.all(withinTol(a, b, 1))

def test_withinTol_scalar_vs_matrix():
    a = 0
    B = np.array([[0, 0.001, 0.002], [-0.001, -0.004, 0.003]])
    result = withinTol(a, B)
    assert result.shape == B.shape
    assert np.all(withinTol(a, B, np.max(np.abs(B))))

def test_withinTol_scalar_vs_matrix_inf():
    a = np.inf
    B = np.full((3, 2), np.inf)
    result = withinTol(a, B)
    assert result.shape == B.shape
    assert np.all(withinTol(a, B))

def test_withinTol_vector_vs_matrix():
    a = np.array([2, 1])
    B = np.array([[2, 1.001], [1.999, 0.999], [2.002, 1.003]])
    result = withinTol(a, B)
    assert result.shape == B.shape
    assert np.all(withinTol(a, B, np.max(np.abs(B - a))))

def test_withinTol_vector_vs_matrix_inf():
    a = np.array([[-np.inf], [np.inf]])
    B = np.array([[-np.inf, -np.inf], [np.inf, np.inf]])
    result = withinTol(a, B)
    assert result.shape == B.shape
    assert np.all(withinTol(a, B))

def test_withinTol_matrix_vs_matrix():
    A = np.array([[0, 1], [1, 0], [2, 0]])
    B = np.array([[-0.001, 1], [1.001, 0.002], [1.999, -0.002]])
    result = withinTol(A, B)
    assert result.shape == A.shape
    assert np.all(withinTol(A, B, np.max(np.abs(B - A))))

def test_withinTol_wrong_dimensions_vector():
    with pytest.raises(CORAerror, match=r"Dimension mismatch between objects\. \[1 0\]"):
        withinTol(np.array([1, 0]), np.array([1, 0, 0]))

def test_withinTol_wrong_dimensions_matrix():
    with pytest.raises(CORAerror, match=r"Dimension mismatch between objects\. \[\[1 0\]\n \[1 2\]\]"):
        withinTol(np.array([[1, 0], [1, 2]]), np.array([[1, 0, 0], [0, 1, 2]]))

def test_withinTol_tolerance_not_scalar():
    with pytest.raises(CORAerror, match=r"Wrong value for the third input argument\.\n\s+The right value: nonnegative scalar\n\s+Type 'help withinTol' for more information\."):
        withinTol(np.array([1, 0]), np.array([1, 0]), np.array([1, 0]))

def test_withinTol_tolerance_not_nonnegative():
    with pytest.raises(CORAerror, match=r"Wrong value for the third input argument\.\n\s+The right value: nonnegative scalar\n\s+Type 'help withinTol' for more information\."):
        withinTol(np.array([1, 0]), np.array([1, 0]), -1)

def test_withinTol_first_input_not_numeric():
    with pytest.raises(CORAerror, match=r"Wrong value for the first input argument\.\n\s+The right value: double\n\s+Type 'help withinTol' for more information\."):
        withinTol("not a number", 1)

def test_withinTol_second_input_not_numeric():
    with pytest.raises(CORAerror, match=r"Wrong value for the second input argument\.\n\s+The right value: double\n\s+Type 'help withinTol' for more information\."):
        withinTol(1, "not a number")