"""
test_sparseOrthMatrix - unit test function for sparseOrthMatrix

Tests the sparseOrthMatrix function for generating sparse orthogonal matrices.

Authors: Python translation by AI Assistant
Date: 2025
"""

import pytest
import numpy as np
from cora_python.g.functions.matlab.init.sparseOrthMatrix import sparseOrthMatrix


class TestSparseOrthMatrix:
    def test_sparseOrthMatrix_basic(self):
        """Test basic sparseOrthMatrix functionality"""
        np.random.seed(42)  # For reproducibility
        n = 5
        Q = sparseOrthMatrix(n)
        
        # Check dimensions
        assert Q.shape == (n, n)
        
        # Check orthogonality: Q^T * Q = I
        assert np.allclose(Q.T @ Q, np.eye(n), atol=1e-10)
        
        # Check that columns have unit norm
        col_norms = np.linalg.norm(Q, axis=0)
        assert np.allclose(col_norms, 1.0, atol=1e-10)
        
        # Check determinant is ±1
        det_Q = np.linalg.det(Q)
        assert np.allclose(np.abs(det_Q), 1.0, atol=1e-9)


if __name__ == "__main__":
    pytest.main([__file__]) 