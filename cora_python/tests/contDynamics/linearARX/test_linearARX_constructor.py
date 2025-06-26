import pytest
import numpy as np
from cora_python.contDynamics.linearARX import LinearARX
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def test_linearARX_constructor():
    # Test case from MATLAB example
    dt = 0.1
    A_bar = [np.array([[-0.4, 0.6], [0.6, -0.4]]), np.array([[0.1, 0], [0.2, -0.5]])]
    B_bar = [np.array([[0], [0]]), np.array([[0.3], [-0.7]]), np.array([[0.1], [0]])]
    
    # 1. Test with name
    sys = LinearARX("myARX", A_bar, B_bar, dt)
    assert sys.name == "myARX"
    assert sys.dt == dt
    assert all(np.array_equal(sys.A_bar[i], A_bar[i]) for i in range(len(A_bar)))
    assert all(np.array_equal(sys.B_bar[i], B_bar[i]) for i in range(len(B_bar)))
    assert sys.n_p == 2
    assert sys.nr_of_outputs == 2
    assert sys.nr_of_inputs == 1

    # 2. Test without name (default name)
    sys_default = LinearARX(A_bar, B_bar, dt)
    assert sys_default.name == "linearARX"
    assert sys_default.dt == dt
    assert all(np.array_equal(sys_default.A_bar[i], A_bar[i]) for i in range(len(A_bar)))
    assert all(np.array_equal(sys_default.B_bar[i], B_bar[i]) for i in range(len(B_bar)))
    assert sys_default.nr_of_outputs == 2
    assert sys_default.nr_of_inputs == 1

def test_invalid_constructor_args():
    dt = 0.1
    A_bar = [np.array([[-0.4, 0.6], [0.6, -0.4]])]
    B_bar = [np.array([[0], [0]]), np.array([[0.3], [-0.7]])]

    # Too few arguments
    with pytest.raises(CORAerror, match=r"CORA:numInputArgsConstructor: \[3, 4\]"):
        LinearARX(A_bar, B_bar)

    # Too many arguments
    with pytest.raises(CORAerror, match=r"CORA:numInputArgsConstructor: \[3, 4\]"):
        LinearARX("name", A_bar, B_bar, dt, "extra")

    # Invalid combination
    with pytest.raises(ValueError, match="Invalid combination of input arguments."):
        LinearARX("name", A_bar, B_bar)

    # Incorrect length of B_bar
    A_bar_valid = [np.array([[-0.4, 0.6], [0.6, -0.4]])]
    B_bar_wrong_len = [np.array([[0], [0]])]
    with pytest.raises(ValueError, match="Length of B_bar must be length of A_bar \\+ 1."):
        LinearARX(A_bar_valid, B_bar_wrong_len, dt)

    # Inconsistent B_bar row dimensions
    B_bar_wrong_row = [np.array([[0]]), np.array([[0.3, -0.7]])]
    with pytest.raises(ValueError, match="Row dimension of matrices in B_bar must match dimension of matrices in A_bar."):
        LinearARX(A_bar_valid, B_bar_wrong_row, dt)

    # Inconsistent B_bar column dimensions
    B_bar_wrong_col = [np.array([[0], [0]]), np.array([[0.3, -0.7], [0, 0]]), np.array([[0], [0]])]
    with pytest.raises(ValueError, match="All matrices in B_bar must have the same number of columns."):
        LinearARX(A_bar_valid, B_bar_wrong_col, dt)

def test_invalid_matrix_dimensions():
    dt = 0.1
    # A_bar not square
    A_bar_rect = [np.array([[-0.4, 0.6]])]
    B_bar_valid = [np.array([[0], [0]]), np.array([[0.3], [-0.7]])]
    with pytest.raises(ValueError, match="Matrices in A_bar must be square."):
        LinearARX(A_bar_rect, B_bar_valid, dt)

    # Inconsistent A_bar dimensions
    A_bar_inconsistent = [np.array([[-0.4, 0.6], [0.6, -0.4]]), np.array([[1]])]
    with pytest.raises(ValueError, match="All matrices in A_bar must have the same dimensions."):
        LinearARX(A_bar_inconsistent, B_bar_valid, dt)
        
    # Incorrect length of B_bar
    A_bar_valid = [np.array([[-0.4, 0.6], [0.6, -0.4]])]
    B_bar_wrong_len = [np.array([[0], [0]])]
    with pytest.raises(ValueError, match="Length of B_bar must be length of A_bar \\+ 1."):
        LinearARX(A_bar_valid, B_bar_wrong_len, dt)

    # Inconsistent B_bar row dimensions
    B_bar_wrong_row = [np.array([[0]]), np.array([[0.3, -0.7]])]
    with pytest.raises(ValueError, match="Row dimension of matrices in B_bar must match dimension of matrices in A_bar."):
        LinearARX(A_bar_valid, B_bar_wrong_row, dt)
        
    # Inconsistent B_bar col dimensions
    B_bar_wrong_col = [np.array([[0], [0]]), np.array([[0.3, -0.7], [0, 0]]), np.array([[0], [0]])]
    with pytest.raises(ValueError, match="All matrices in B_bar must have the same number of columns."):
        LinearARX(A_bar_valid, B_bar_wrong_col, dt)

def test_empty_A_bar():
    with pytest.raises(CORAerror, match="Wrong value for the 2nd input argument."):
        LinearARX([], [np.array([[1]])], 0.1) 