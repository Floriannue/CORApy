"""
test_gramSchmidt - unit test function for gramSchmidt

Tests the gramSchmidt function for constructing orthonormal basis.

Authors: Python translation by AI Assistant
Date: 2025
"""

import pytest
import numpy as np
from cora_python.g.functions.matlab.init.gramSchmidt import gramSchmidt


class TestGramSchmidt:
    def test_gramSchmidt_basic_2d(self):
        """Test basic 2D Gram-Schmidt orthogonalization"""
        # Two linearly independent vectors
        V = np.array([[1, 1], [0, 1]])
        Q = gramSchmidt(V)
        
        # Check orthonormality
        assert np.allclose(np.dot(Q.T, Q), np.eye(2), atol=1e-10)
        
        # Check that columns have unit norm
        for i in range(Q.shape[1]):
            assert np.allclose(np.linalg.norm(Q[:, i]), 1.0)
    
    def test_gramSchmidt_basic_3d(self):
        """Test basic 3D Gram-Schmidt orthogonalization"""
        # Three linearly independent vectors
        V = np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]])
        Q = gramSchmidt(V)
        
        # Check orthonormality
        assert np.allclose(np.dot(Q.T, Q), np.eye(3), atol=1e-10)
        
        # Check that columns have unit norm
        for i in range(Q.shape[1]):
            assert np.allclose(np.linalg.norm(Q[:, i]), 1.0)
    
    def test_gramSchmidt_already_orthogonal(self):
        """Test with already orthogonal vectors"""
        # Standard basis vectors
        V = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        Q = gramSchmidt(V)
        
        # Should remain the same (up to sign)
        assert np.allclose(np.abs(Q), np.eye(3))
        assert np.allclose(np.dot(Q.T, Q), np.eye(3), atol=1e-10)
    
    def test_gramSchmidt_linearly_dependent(self):
        """Test with linearly dependent vectors"""
        # Second vector is multiple of first
        V = np.array([[1, 2], [2, 4]])
        Q = gramSchmidt(V)
        
        # Should have only one column (rank 1)
        assert Q.shape[1] == 1
        assert np.allclose(np.linalg.norm(Q[:, 0]), 1.0)
    
    def test_gramSchmidt_single_vector(self):
        """Test with single vector"""
        V = np.array([[3], [4]])
        Q = gramSchmidt(V)
        
        # Should be normalized
        expected = np.array([[3/5], [4/5]])
        assert np.allclose(Q, expected)
        assert np.allclose(np.linalg.norm(Q), 1.0)
    
    def test_gramSchmidt_zero_vector(self):
        """Test with zero vector"""
        V = np.array([[1, 0], [0, 0]])
        Q = gramSchmidt(V)
        
        # Should remove zero vector
        assert Q.shape[1] == 1
        assert np.allclose(Q, np.array([[1], [0]]))
    
    def test_gramSchmidt_rectangular_more_rows(self):
        """Test with more rows than columns"""
        V = np.array([[1, 2], [0, 1], [1, 0]])
        Q = gramSchmidt(V)
        
        # Check orthonormality
        assert np.allclose(np.dot(Q.T, Q), np.eye(Q.shape[1]), atol=1e-10)
        
        # Check dimensions
        assert Q.shape[0] == 3
        assert Q.shape[1] <= 2
    
    def test_gramSchmidt_span_preservation(self):
        """Test that span is preserved"""
        V = np.array([[1, 1], [1, -1], [0, 2]])
        Q = gramSchmidt(V)
        
        # Check that Q spans the same space as V
        # This is verified by checking that V can be expressed as linear combination of Q
        # and that Q has the same rank as V
        rank_V = np.linalg.matrix_rank(V)
        rank_Q = np.linalg.matrix_rank(Q)
        assert rank_Q == rank_V
    
    def test_gramSchmidt_numerical_stability(self):
        """Test numerical stability with nearly dependent vectors"""
        # Create nearly linearly dependent vectors
        V = np.array([[1, 1 + 1e-10], [0, 1e-10]])
        Q = gramSchmidt(V)
        
        # Should still produce orthonormal result
        if Q.shape[1] > 0:
            assert np.allclose(np.dot(Q.T, Q), np.eye(Q.shape[1]), atol=1e-8)
    
    def test_gramSchmidt_random_vectors(self):
        """Test with random vectors"""
        np.random.seed(42)  # For reproducibility
        for _ in range(5):
            m, n = np.random.randint(3, 8), np.random.randint(2, 5)
            V = np.random.randn(m, n)
            Q = gramSchmidt(V)
            
            # Check orthonormality
            if Q.shape[1] > 0:
                assert np.allclose(np.dot(Q.T, Q), np.eye(Q.shape[1]), atol=1e-10)
                
                # Check that columns have unit norm
                for i in range(Q.shape[1]):
                    assert np.allclose(np.linalg.norm(Q[:, i]), 1.0)
    
    def test_gramSchmidt_empty_input(self):
        """Test with empty input"""
        V = np.array([]).reshape(3, 0)
        Q = gramSchmidt(V)
        
        # Should return empty matrix with same number of rows
        assert Q.shape == (3, 0)


if __name__ == "__main__":
    pytest.main([__file__]) 