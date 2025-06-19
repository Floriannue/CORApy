"""
test_mat2vec - unit test function for mat2vec

Tests the mat2vec function for converting matrices to vectors.

Authors: Python translation by AI Assistant
Date: 2025
"""

import pytest
import numpy as np
from cora_python.g.functions.matlab.converter import mat2vec


class TestMat2vec:
    def test_mat2vec_basic(self):
        """Test basic mat2vec functionality"""
        # Test 2x2 matrix
        mat = np.array([[1, 2], [3, 4]])
        vec = mat2vec(mat)
        # MATLAB uses column-major order: [1, 3, 2, 4]
        expected = np.array([[1], [3], [2], [4]])
        assert np.allclose(vec, expected)
    
    def test_mat2vec_3x3(self):
        """Test 3x3 matrix"""
        mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        vec = mat2vec(mat)
        # Column-major order: [1, 4, 7, 2, 5, 8, 3, 6, 9]
        expected = np.array([[1], [4], [7], [2], [5], [8], [3], [6], [9]])
        assert np.allclose(vec, expected)
    
    def test_mat2vec_row_vector(self):
        """Test row vector input"""
        mat = np.array([[1, 2, 3]])
        vec = mat2vec(mat)
        expected = np.array([[1], [2], [3]])
        assert np.allclose(vec, expected)
    
    def test_mat2vec_column_vector(self):
        """Test column vector input"""
        mat = np.array([[1], [2], [3]])
        vec = mat2vec(mat)
        expected = np.array([[1], [2], [3]])
        assert np.allclose(vec, expected)
    
    def test_mat2vec_scalar(self):
        """Test scalar input"""
        mat = np.array([[5]])
        vec = mat2vec(mat)
        expected = np.array([[5]])
        assert np.allclose(vec, expected)
    
    def test_mat2vec_empty(self):
        """Test empty matrix"""
        mat = np.array([]).reshape(0, 0)
        vec = mat2vec(mat)
        expected = np.array([]).reshape(0, 1)
        assert vec.shape == expected.shape
    
    def test_mat2vec_rectangular(self):
        """Test rectangular matrices"""
        # 2x3 matrix
        mat = np.array([[1, 2, 3], [4, 5, 6]])
        vec = mat2vec(mat)
        expected = np.array([[1], [4], [2], [5], [3], [6]])
        assert np.allclose(vec, expected)
        
        # 3x2 matrix
        mat = np.array([[1, 2], [3, 4], [5, 6]])
        vec = mat2vec(mat)
        expected = np.array([[1], [3], [5], [2], [4], [6]])
        assert np.allclose(vec, expected)
    
    def test_mat2vec_output_shape(self):
        """Test that output is always a column vector"""
        for shape in [(2, 2), (3, 3), (2, 4), (4, 2), (1, 5), (5, 1)]:
            mat = np.random.rand(*shape)
            vec = mat2vec(mat)
            assert vec.shape[1] == 1  # Always column vector
            assert vec.shape[0] == mat.size  # Total elements preserved


if __name__ == "__main__":
    pytest.main([__file__]) 