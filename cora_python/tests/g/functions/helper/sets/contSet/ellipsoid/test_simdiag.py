import numpy as np
import pytest
from cora_python.g.functions.helper.sets.contSet.ellipsoid.simdiag import simdiag
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

# Helper function to check if a matrix is symmetric within a tolerance
def is_symmetric(M, tol):
    return np.allclose(M, M.T, atol=tol)

# Helper function to check if a matrix is positive definite within a tolerance
def is_positive_definite(M, tol):
    return np.all(np.linalg.eigvalsh(M) > tol)

# Helper function to check if a matrix is positive semi-definite within a tolerance
def is_positive_semi_definite(M, tol):
    return np.all(np.linalg.eigvalsh(M) >= -tol)

class TestSimdiag:
    def test_simdiag_basic(self):
        M1 = np.array([[2, 1], [1, 2]])
        M2 = np.array([[3, 0], [0, 3]])
        TOL = 1e-6

        T, D = simdiag(M1, M2, TOL)

        # Verify T * M1 * T.T is close to identity
        assert np.allclose(T @ M1 @ T.T, np.eye(2), atol=TOL)
        # Verify D is diagonal
        assert np.allclose(D, np.diag(np.diag(D)), atol=TOL)

    def test_simdiag_non_diagonal_M2(self):
        M1 = np.array([[2, 0], [0, 2]])
        M2 = np.array([[3, 1], [1, 3]])
        TOL = 1e-6

        T, D = simdiag(M1, M2, TOL)
        assert np.allclose(T @ M1 @ T.T, np.eye(2), atol=TOL)
        assert np.allclose(D, np.diag(np.diag(D)), atol=TOL)

    def test_simdiag_different_scales(self):
        M1 = np.array([[10, 0], [0, 20]])
        M2 = np.array([[1, 0], [0, 2]])
        TOL = 1e-6

        T, D = simdiag(M1, M2, TOL)
        assert np.allclose(T @ M1 @ T.T, np.eye(2), atol=TOL)
        assert np.allclose(D, np.diag(np.diag(D)), atol=TOL)

    def test_simdiag_mismatch_dimensions(self):
        M1 = np.array([[1, 2], [3, 4]])
        M2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        TOL = 1e-6

        with pytest.raises(CORAerror) as excinfo:
            simdiag(M1, M2, TOL)
        assert excinfo.value.identifier == "CORA:dimensionMismatch"

    def test_simdiag_non_symmetric_M1(self):
        M1 = np.array([[1, 2], [0, 1]])  # Non-symmetric
        M2 = np.array([[1, 0], [0, 1]])
        TOL = 1e-6

        with pytest.raises(CORAerror) as excinfo:
            simdiag(M1, M2, TOL)
        assert excinfo.value.identifier == "CORA:specialError"

    def test_simdiag_non_positive_definite_M1(self):
        M1 = np.array([[-1, 0], [0, -1]])  # Not positive definite
        M2 = np.array([[1, 0], [0, 1]])
        TOL = 1e-6

        with pytest.raises(CORAerror) as excinfo:
            simdiag(M1, M2, TOL)
        assert excinfo.value.identifier == "CORA:specialError"

    def test_simdiag_non_positive_semi_definite_M2(self):
        M1 = np.array([[1, 0], [0, 1]])
        M2 = np.array([[-1, 0], [0, -1]])  # Not positive semi-definite
        TOL = 1e-6

        with pytest.raises(CORAerror) as excinfo:
            simdiag(M1, M2, TOL)
        assert excinfo.value.identifier == "CORA:specialError" 