import pytest
import numpy as np
from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol
from cora_python.contSet.contSet.representsa_ import representsa_
from scipy.linalg import block_diag


class TestEllipsoid:
    def test_empty_ellipsoid(self):
        """Test creation of an empty ellipsoid"""
        E = Ellipsoid.empty(2)
        assert representsa_(E, 'emptySet'), "Empty ellipsoid should represent an empty set"

    def test_init_only_shape_matrix(self):
        """Test ellipsoid initialization with only a shape matrix"""
        tol = 1e-12
        Q = np.array([[5.4387811500952807, 12.4977183618314545], 
                      [12.4977183618314545, 29.6662117284481646]])
        E = Ellipsoid(Q)
        assert np.all(withinTol(E.Q, Q, tol)), "Shape matrix should match"
        assert np.all(withinTol(E.q, np.zeros((Q.shape[0], 1)), tol)), "Center should be zero vector"

    def test_init_shape_matrix_and_center(self):
        """Test ellipsoid initialization with shape matrix and center"""
        tol = 1e-12
        Q = np.array([[5.4387811500952807, 12.4977183618314545], 
                      [12.4977183618314545, 29.6662117284481646]])
        q = np.array([[-0.7445068341257537], [3.5800647524843665]])
        E = Ellipsoid(Q, q)
        assert np.all(withinTol(E.Q, Q, tol)), "Shape matrix should match"
        assert np.all(withinTol(E.q, q, tol)), "Center should match"

    def test_init_shape_matrix_non_psd(self):
        """Test ellipsoid initialization with non-PSD shape matrix"""
        n = 2  # Assuming n > 1 for this test as per MATLAB source
        if n > 1:
            Q_nonpsd = np.array([[-1, 0], [0, -1]]) # Simple non-psd matrix
            with pytest.raises(CORAerror) as excinfo:
                Ellipsoid(Q_nonpsd)
            assert "CORA:wrongInputInConstructor" in str(excinfo.value) or \
                   "The shape matrix needs to be positive semidefinite/symmetric." in str(excinfo.value)

    def test_init_shape_matrix_and_center_different_dimensions(self):
        """Test ellipsoid initialization with shape matrix and center of different dimensions"""
        Q = np.array([[1, 0], [0, 1]])
        q = np.array([[1], [2], [3]]) # Mismatched dimension
        with pytest.raises(CORAerror) as excinfo:
            Ellipsoid(Q, q)
        assert excinfo.value.identifier == "CORA:wrongInputInConstructor"

        Q_plus1 = block_diag(Q, 1) # Add a dimension to Q
        q_orig = np.array([[1], [2]])
        with pytest.raises(CORAerror) as excinfo:
            Ellipsoid(Q_plus1, q_orig)
        assert excinfo.value.identifier == "CORA:wrongInputInConstructor"

    def test_init_center_is_matrix(self):
        """Test ellipsoid initialization when center is a matrix instead of a column vector"""
        n = 2
        if n != 1:
            Q = np.array([[1, 0], [0, 1]])
            q_mat = np.array([[1, 2], [3, 4]]) # 2x2 matrix instead of 2x1 vector
            with pytest.raises(CORAerror) as excinfo:
                Ellipsoid(Q, q_mat)
            assert excinfo.value.identifier in ["CORA:wrongInputInConstructor", "CORA:wrongValue"]

    def test_init_too_many_input_arguments(self):
        """Test ellipsoid initialization with too many input arguments"""
        Q = np.array([[1, 0], [0, 1]])
        q = np.array([[1], [2]])
        # MATLAB uses 'eps' for tolerance, which is machine epsilon. In Python, use a small float.
        eps_val = np.finfo(float).eps # Machine epsilon
        with pytest.raises(CORAerror) as excinfo:
            Ellipsoid(Q, q, eps_val, q) # 4 arguments, max is 3
        assert excinfo.value.identifier == "CORA:numInputArgsConstructor"

    @pytest.mark.long
    @pytest.mark.parametrize("i", range(100))
    def test_long_random_ellipsoid_init(self, i):
        """Long running test for random ellipsoid initializations"""
        tol = 1e-12
        # random dimension
        n = np.random.randint(1, 16)  # randi(15) in MATLAB, assuming min dim is 1
        # random shape matrix (psd and non-psd) and random center
        q = np.random.randn(n, 1)
        Q_nonpsd = np.random.randn(n, n)
        # wrong initializations
        q_plus1 = np.random.randn(n + 1, 1)
        q_mat = np.random.randn(n, n)
        temp = np.random.randn(n + 1, n + 1)

        Q = Q_nonpsd @ Q_nonpsd.T  # Make Q positive semi-definite

        # admissible initializations
        # only shape matrix
        E = Ellipsoid(Q)
        assert np.all(withinTol(E.Q, Q, tol)), f"Test {i}: Shape matrix mismatch for Q only"
        assert np.all(withinTol(E.q, np.zeros((Q.shape[0], 1)), tol)), f"Test {i}: Center mismatch for Q only"

        # shape matrix and center
        E = Ellipsoid(Q, q)
        assert np.all(withinTol(E.Q, Q, tol)), f"Test {i}: Shape matrix mismatch for Q and q"
        assert np.all(withinTol(E.q, q, tol)), f"Test {i}: Center mismatch for Q and q"

        Q_plus1 = temp @ temp.T

        # shape matrix non-psd (only n > 1)
        if n > 1:
            with pytest.raises(CORAerror) as excinfo:
                Ellipsoid(Q_nonpsd)
            assert excinfo.value.identifier == "CORA:wrongInputInConstructor", f"Test {i}: Expected wrongInputInConstructor for non-PSD Q"

        # shape matrix and center of different dimensions
        with pytest.raises(CORAerror) as excinfo:
            Ellipsoid(Q, q_plus1)
        assert excinfo.value.identifier == "CORA:wrongInputInConstructor", f"Test {i}: Expected wrongInputInConstructor for dim mismatch (Q, q+1)"

        with pytest.raises(CORAerror) as excinfo:
            Ellipsoid(Q_plus1, q)
        assert excinfo.value.identifier == "CORA:wrongInputInConstructor", f"Test {i}: Expected wrongInputInConstructor for dim mismatch (Q+1, q)"

        # center is a matrix
        if n > 1:
            with pytest.raises(CORAerror) as excinfo:
                Ellipsoid(Q, q_mat)
            assert excinfo.value.identifier in ["CORA:wrongInputInConstructor", "CORA:wrongValue"], f"Test {i}: Expected wrongValue for matrix center"

        # too many input arguments
        with pytest.raises(CORAerror) as excinfo:
            Ellipsoid(Q, q, np.finfo(float).eps, q)
        assert excinfo.value.identifier == "CORA:numInputArgsConstructor", f"Test {i}: Expected numInputArgsConstructor" 