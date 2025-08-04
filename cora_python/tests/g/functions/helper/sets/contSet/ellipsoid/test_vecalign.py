import numpy as np
import pytest
from cora_python.g.functions.helper.sets.contSet.ellipsoid.vecalign import vecalign
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

class TestVecalign:
    def test_basic_alignment_2d(self):
        x = np.array([1, 0])
        y = np.array([0, 1])
        T = vecalign(x, y)
        # Expected T should rotate y to x: [[0, 1], [-1, 0]] or [[0, -1], [1, 0]]
        # The SVD method gives a rotation. Let's check the transformed vector.
        aligned_y = T @ y.reshape(-1, 1)
        # Check if aligned_y is parallel to x
        assert np.isclose(np.linalg.norm(np.cross(x, aligned_y.flatten())), 0)
        # Check if T is orthogonal
        assert np.allclose(T @ T.T, np.eye(x.shape[0]))

    def test_basic_alignment_3d(self):
        x = np.array([1, 0, 0])
        y = np.array([0, 1, 0])
        T = vecalign(x, y)
        aligned_y = T @ y.reshape(-1, 1)
        # Check if aligned_y is parallel to x
        assert np.isclose(np.linalg.norm(np.cross(x, aligned_y.flatten())), 0)
        # Check if T is orthogonal
        assert np.allclose(T @ T.T, np.eye(x.shape[0]))

    def test_same_direction(self):
        x = np.array([1, 1])
        y = np.array([2, 2])
        T = vecalign(x, y)
        # T should be close to identity
        assert np.allclose(T, np.eye(x.shape[0]))

    def test_opposite_direction(self):
        x = np.array([1, 0])
        y = np.array([-1, 0])
        T = vecalign(x, y)
        aligned_y = T @ y.reshape(-1, 1)
        # aligned_y should be parallel to x
        assert np.isclose(np.linalg.norm(np.cross(x, aligned_y.flatten())), 0)
        # T should be a 180-degree rotation (e.g., [[-1, 0], [0, -1]])
        assert np.allclose(T @ T.T, np.eye(x.shape[0]))

    def test_random_vectors(self):
        np.random.seed(0) # for reproducibility
        for _ in range(5):
            n = np.random.randint(2, 5) # Random dimension
            x = np.random.rand(n)
            y = np.random.rand(n)
            T = vecalign(x, y)
            aligned_y = T @ y.reshape(-1, 1)

            # Check if aligned_y is parallel to x using dot product for general dimensions
            # abs(dot(x_norm, aligned_y_norm)) should be 1
            x_norm = x / np.linalg.norm(x)
            aligned_y_norm = aligned_y.flatten() / np.linalg.norm(aligned_y.flatten())
            assert np.isclose(np.abs(np.dot(x_norm, aligned_y_norm)), 1.0)
            
            # Check if T is orthogonal
            assert np.allclose(T @ T.T, np.eye(n))

    def test_zero_vectors(self):
        # Both zero
        x_zero = np.array([0, 0])
        y_zero = np.array([0, 0])
        T_identity = vecalign(x_zero, y_zero)
        assert np.allclose(T_identity, np.eye(x_zero.shape[0]))

        # One zero, one non-zero
        x_nonzero = np.array([1, 2])
        y_zero = np.array([0, 0])
        with pytest.raises(CORAerror) as excinfo:
            vecalign(x_nonzero, y_zero)
        assert 'Cannot align a non-zero vector with a zero vector.' in str(excinfo.value)
        
        with pytest.raises(CORAerror) as excinfo:
            vecalign(y_zero, x_nonzero)
        assert 'Cannot align a non-zero vector with a zero vector.' in str(excinfo.value)

    def test_dimension_mismatch(self):
        x = np.array([1, 0])
        y = np.array([0, 1, 0])
        with pytest.raises(CORAerror) as excinfo:
            vecalign(x, y)
        assert 'Input vectors x and y must have the same dimension.' in str(excinfo.value)

    def test_empty_input(self):
        x_empty = np.array([])
        y_empty = np.array([])
        with pytest.raises(CORAerror) as excinfo:
            vecalign(x_empty, y_empty)
        assert 'Input vectors cannot be empty.' in str(excinfo.value)

    def test_column_vector_input(self):
        x_col = np.array([[1], [0]])
        y_col = np.array([[0], [1]])
        T = vecalign(x_col, y_col)
        aligned_y = T @ y_col
        assert np.isclose(np.linalg.norm(np.cross(x_col.flatten(), aligned_y.flatten())), 0)
        assert np.allclose(T @ T.T, np.eye(x_col.shape[0])) 