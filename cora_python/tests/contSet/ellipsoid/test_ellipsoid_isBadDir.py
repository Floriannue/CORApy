import numpy as np
import pytest
from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid
from cora_python.contSet.ellipsoid.isBadDir import isBadDir
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

class TestEllipsoidIsBadDir:
    # Helper function to create a simple ellipsoid
    def create_ellipsoid(self, Q, q=None, TOL=1e-6):
        if q is None:
            q = np.zeros((Q.shape[0], 1))
        return Ellipsoid(Q, q, TOL)

    def test_isBadDir_basic_false(self):
        # Example where it should return false
        E1 = self.create_ellipsoid(np.eye(2) * 2)
        E2 = self.create_ellipsoid(np.eye(2) * 1)
        L = np.array([[1], [0]]) # A single direction
        res = isBadDir(E1, E2, L)
        assert not res, "Expected isBadDir to be false"

    def test_isBadDir_expected_false_2(self):
        # Example where it should return false
        E1 = self.create_ellipsoid(np.array([[10, 0], [0, 10]]))
        E2 = self.create_ellipsoid(np.array([[1, 0], [0, 1]]))
        L = np.array([[1], [0]])
        res = isBadDir(E1, E2, L)
        assert not res, "Expected isBadDir to be false"

    def test_isBadDir_expected_false_multiple_directions(self):
        E1 = self.create_ellipsoid(np.array([[10, 0], [0, 10]]))
        E2 = self.create_ellipsoid(np.array([[1, 0], [0, 1]]))
        L = np.array([[1, 0], [0, 1]]) # Two directions
        res = isBadDir(E1, E2, L)
        assert isinstance(res, np.ndarray)
        assert res.shape == (1, 2)
        assert not any(res[0, :]), "Expected both directions to be false"

    def test_isBadDir_expected_false_mixed_directions(self):
        E1 = self.create_ellipsoid(np.array([[10, 0], [0, 10]]))
        E2 = self.create_ellipsoid(np.array([[1, 0], [0, 1]]))
        L = np.array([[1, 0.1], [0, 1]]) # One bad, one good (or less bad)
        res = isBadDir(E1, E2, L)
        assert isinstance(res, np.ndarray)
        assert res.shape == (1, 2)
        assert not any(res[0, :]), "Expected both directions to be false for mixed case"

    def test_isBadDir_dimension_mismatch_L(self):
        E1 = self.create_ellipsoid(np.eye(2))
        E2 = self.create_ellipsoid(np.eye(2))
        L = np.array([[1, 2, 3]]) # Incorrect dimension for L

        with pytest.raises(CORAerror) as excinfo:
            isBadDir(E1, E2, L)
        assert excinfo.value.identifier == "CORA:dimensionMismatch"

    def test_isBadDir_ellipsoid_dimension_mismatch(self):
        E1 = self.create_ellipsoid(np.eye(2))
        E2 = self.create_ellipsoid(np.eye(3)) # Mismatch
        L = np.array([[1], [0]])

        with pytest.raises(CORAerror) as excinfo:
            isBadDir(E1, E2, L)
        assert excinfo.value.identifier == "CORA:dimensionMismatch" 