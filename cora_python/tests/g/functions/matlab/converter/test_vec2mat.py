"""
test_vec2mat - unit test function for vec2mat

Tests the vec2mat function for converting vectors to matrices.

Authors: Python translation by AI Assistant
Date: 2025
"""

import pytest
import numpy as np
from cora_python.g.functions.matlab.converter import vec2mat, mat2vec


class TestVec2mat:
    def test_vec2mat_basic(self):
        """Test basic vec2mat functionality"""
        # Test converting 4-element vector to 2x2 matrix
        vec = np.array([[1], [3], [2], [4]])
        mat = vec2mat(vec, 2)
        # Column-major order: first column [1, 3], second column [2, 4]
        expected = np.array([[1, 2], [3, 4]])
        assert np.allclose(mat, expected)
    
    def test_vec2mat_3x3(self):
        """Test 3x3 matrix reconstruction"""
        vec = np.array([[1], [4], [7], [2], [5], [8], [3], [6], [9]])
        mat = vec2mat(vec, 3)
        expected = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        assert np.allclose(mat, expected)
    
    def test_vec2mat_rectangular(self):
        """Test rectangular matrices"""
        # 3x2 matrix (6 elements, 2 columns)
        vec = np.array([[1], [4], [2], [5], [3], [6]])
        mat = vec2mat(vec, 2)
        # Column-major: col1=[1,4,2], col2=[5,3,6] -> [[1,5],[4,3],[2,6]]
        expected = np.array([[1, 5], [4, 3], [2, 6]])
        assert np.allclose(mat, expected)
        
        # 2x3 matrix (6 elements, 3 columns)
        vec = np.array([[1], [3], [2], [4], [5], [6]])
        mat = vec2mat(vec, 3)
        # Column-major: col1=[1,3], col2=[2,4], col3=[5,6] -> [[1,2,5],[3,4,6]]
        expected = np.array([[1, 2, 5], [3, 4, 6]])
        assert np.allclose(mat, expected)
    
    def test_vec2mat_single_column(self):
        """Test single column matrix"""
        vec = np.array([[1], [2], [3]])
        mat = vec2mat(vec, 1)
        expected = np.array([[1], [2], [3]])
        assert np.allclose(mat, expected)
    
    def test_vec2mat_single_row(self):
        """Test single row matrix"""
        vec = np.array([[1], [2], [3]])
        mat = vec2mat(vec, 3)
        expected = np.array([[1, 2, 3]])
        assert np.allclose(mat, expected)
    
    def test_vec2mat_scalar(self):
        """Test scalar case"""
        vec = np.array([[5]])
        mat = vec2mat(vec, 1)
        expected = np.array([[5]])
        assert np.allclose(mat, expected)
    
    def test_vec2mat_empty(self):
        """Test empty vector"""
        vec = np.array([]).reshape(0, 1)
        mat = vec2mat(vec, 0)
        expected = np.array([]).reshape(0, 0)
        assert mat.shape == expected.shape
    
    def test_vec2mat_roundtrip(self):
        """Test roundtrip conversion with mat2vec"""
        
        # Test various matrix shapes
        for shape in [(2, 2), (3, 3), (2, 4), (4, 2), (1, 5), (5, 1)]:
            original = np.random.rand(*shape)
            vec = mat2vec(original)
            reconstructed = vec2mat(vec, shape[1])
            assert np.allclose(original, reconstructed)
    
    def test_vec2mat_output_shape(self):
        """Test output shape correctness"""
        # Test that the output has correct dimensions
        vec_lengths = [4, 6, 9, 12]
        for vec_len in vec_lengths:
            vec = np.random.rand(vec_len, 1)
            for n_cols in range(1, vec_len + 1):
                if vec_len % n_cols == 0:  # Only test valid combinations
                    mat = vec2mat(vec, n_cols)
                    expected_rows = vec_len // n_cols
                    assert mat.shape == (expected_rows, n_cols)


if __name__ == "__main__":
    pytest.main([__file__]) 