import numpy as np
import pytest
from cora_python.matrixSet.matPolytope.matPolytope import MatPolytope
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def test_matpolytope_empty_constructor():
    """Test MatPolytope constructor with no arguments (empty matrix polytope)."""
    matP = MatPolytope()
    assert isinstance(matP, MatPolytope)
    assert matP.V.shape == (0, 0, 0)

def test_matpolytope_single_matrix_vertex():
    """Test MatPolytope constructor with a single 2D matrix as a vertex."""
    V_in = np.array([[1, 2], [3, 4]])
    # MATLAB reshape(V,[dim(P),1,size(V,2)]) would make it (n, m, 1) for a single matrix
    # Here V_in.ndim is 2, but constructor expects 3 for non-empty. This will be an issue.
    # The Polytope.matPolytope will reshape it before passing. For direct constructor test, we need 3D.
    V_in_3d = V_in.reshape(V_in.shape[0], V_in.shape[1], 1)

    matP = MatPolytope(V_in_3d)
    assert isinstance(matP, MatPolytope)
    assert matP.V.shape == (2, 2, 1)
    assert np.array_equal(matP.V[:,:,0], V_in)

def test_matpolytope_multiple_matrix_vertices():
    """Test MatPolytope constructor with multiple 2D matrices as vertices."""
    V1 = np.array([[1, 2], [3, 4]])
    V2 = np.array([[5, 6], [7, 8]])
    V3 = np.array([[9, 10], [11, 12]])

    V_in = np.stack((V1, V2, V3), axis=2) # Stacks along a new 3rd axis, (n, m, N)

    matP = MatPolytope(V_in)
    assert isinstance(matP, MatPolytope)
    assert matP.V.shape == (2, 2, 3)
    assert np.array_equal(matP.V[:,:,0], V1)
    assert np.array_equal(matP.V[:,:,1], V2)
    assert np.array_equal(matP.V[:,:,2], V3)

def test_matpolytope_non_numeric_input_error():
    """Test MatPolytope constructor with non-numeric input (should raise error)."""
    with pytest.raises(CORAerror) as excinfo:
        MatPolytope("invalid")
    assert "Input vertices V must be a NumPy array" in str(excinfo.value)

def test_matpolytope_nan_input_error():
    """Test MatPolytope constructor with NaN values in vertices (should raise error)."""
    V_nan = np.array([[[1, np.nan], [3, 4]]])
    with pytest.raises(CORAerror) as excinfo:
        MatPolytope(V_nan)
    assert "Input vertices V must not contain NaN values" in str(excinfo.value)

def test_matpolytope_incorrect_dim_input_error():
    """Test MatPolytope constructor with incorrect number of dimensions for input V."""
    # 2D array, should be 3D
    V_2d = np.array([[1, 2], [3, 4]])
    with pytest.raises(CORAerror) as excinfo:
        MatPolytope(V_2d)
    assert "Input vertices V must be a 3D NumPy array" in str(excinfo.value)
