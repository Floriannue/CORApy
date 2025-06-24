"""
test_ndimCross - unit test function for ndimCross

Syntax:
    pytest test_ndimCross.py

Inputs:
    -

Outputs:
    test results

Other modules required: none
Subfunctions: none

See also: none

Authors: AI Assistant
Written: 2025
Last update: ---
Last revision: ---
"""

import pytest
import numpy as np
from cora_python.g.functions.helper.sets.contSet.zonotope.ndimCross import ndimCross


class TestNdimCross:
    """Test class for ndimCross function"""
    
    def test_ndimCross_2d(self):
        """Test ndimCross for 2D case (standard cross product)"""
        
        # 2D case: 2x1 matrix (single vector)
        Q = np.array([[1], [2]], dtype=float)
        v = ndimCross(Q)
        expected = np.array([[2], [-1]], dtype=float)
        assert np.allclose(v, expected)
        
        # Another 2D case
        Q = np.array([[3], [4]], dtype=float)
        v = ndimCross(Q)
        expected = np.array([[4], [-3]], dtype=float)
        assert np.allclose(v, expected)
    
    def test_ndimCross_3d(self):
        """Test ndimCross for 3D case"""
        
        # 3D case: 3x2 matrix
        Q = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        v = ndimCross(Q)
        
        # Manual calculation: v = [det([3,4;5,6]), -det([1,2;5,6]), det([1,2;3,4])]
        # det([3,4;5,6]) = 3*6 - 4*5 = 18 - 20 = -2
        # det([1,2;5,6]) = 1*6 - 2*5 = 6 - 10 = -4
        # det([1,2;3,4]) = 1*4 - 2*3 = 4 - 6 = -2
        expected = np.array([[-2], [4], [-2]], dtype=float)
        assert np.allclose(v, expected)
    
    def test_ndimCross_4d(self):
        """Test ndimCross for 4D case"""
        
        # 4D case: 4x3 matrix
        Q = np.array([
            [1, 2, 3],
            [4, 5, 6], 
            [7, 8, 9],
            [10, 11, 12]
        ], dtype=float)
        v = ndimCross(Q)
        
        # The result should be a 4x1 vector
        assert v.shape == (4, 1)
        
        # Check that result is orthogonal to all input vectors
        for i in range(3):
            dot_product = np.dot(v.flatten(), Q[:, i])
            assert np.abs(dot_product) < 1e-10
    
    def test_ndimCross_orthogonality(self):
        """Test that result is orthogonal to all input vectors"""
        
        # Random 3D case
        np.random.seed(42)
        Q = np.random.randn(3, 2)
        v = ndimCross(Q)
        
        # Check orthogonality
        for i in range(2):
            dot_product = np.dot(v.flatten(), Q[:, i])
            assert np.abs(dot_product) < 1e-10
        
        # Random 4D case
        Q = np.random.randn(4, 3)
        v = ndimCross(Q)
        
        # Check orthogonality
        for i in range(3):
            dot_product = np.dot(v.flatten(), Q[:, i])
            assert np.abs(dot_product) < 1e-10
    
    def test_ndimCross_standard_basis(self):
        """Test ndimCross with standard basis vectors"""
        
        # 3D standard basis
        Q = np.array([[1, 0], [0, 1], [0, 0]], dtype=float)
        v = ndimCross(Q)
        expected = np.array([[0], [0], [1]], dtype=float)
        assert np.allclose(v, expected)
        
        # Another 3D standard basis combination
        Q = np.array([[1, 0], [0, 0], [0, 1]], dtype=float)
        v = ndimCross(Q)
        expected = np.array([[0], [-1], [0]], dtype=float)
        assert np.allclose(v, expected)
        
        # Yet another 3D standard basis combination
        Q = np.array([[0, 1], [1, 0], [0, 0]], dtype=float)
        v = ndimCross(Q)
        expected = np.array([[0], [0], [-1]], dtype=float)
        assert np.allclose(v, expected)
    
    def test_ndimCross_linearly_dependent(self):
        """Test ndimCross with linearly dependent vectors"""
        
        # Linearly dependent vectors should give zero cross product
        Q = np.array([[1, 2], [2, 4], [3, 6]], dtype=float)
        v = ndimCross(Q)
        
        # Result should be zero vector (or very close to zero)
        assert np.allclose(v, np.zeros((3, 1)))
    
    def test_ndimCross_error_cases(self):
        """Test error cases for ndimCross"""
        
        # Wrong dimensions - not n x (n-1)
        with pytest.raises(ValueError, match="Q must be a n x \\(n-1\\) matrix"):
            Q = np.array([[1, 2, 3], [4, 5, 6]])  # 2x3 instead of 2x1
            ndimCross(Q)
        
        with pytest.raises(ValueError, match="Q must be a n x \\(n-1\\) matrix"):
            Q = np.array([[1], [2], [3]])  # 3x1 instead of 3x2
            ndimCross(Q)
        
        # 1D array instead of 2D matrix
        with pytest.raises(ValueError, match="Q must be a 2D matrix"):
            Q = np.array([1, 2, 3])
            ndimCross(Q)
        
        # 3D array
        with pytest.raises(ValueError, match="Q must be a 2D matrix"):
            Q = np.array([[[1, 2], [3, 4]]])
            ndimCross(Q)
    
    def test_ndimCross_edge_cases(self):
        """Test edge cases for ndimCross"""
        
        # 1D case (n=1, k=0) - empty matrix
        Q = np.array([[]], dtype=float).reshape(1, 0)
        v = ndimCross(Q)
        expected = np.array([[1]], dtype=float)
        assert np.allclose(v, expected)
        
        # Very small numbers
        Q = np.array([[1e-10], [1e-10]], dtype=float)
        v = ndimCross(Q)
        assert v.shape == (2, 1)
        
        # Very large numbers
        Q = np.array([[1e10], [1e10]], dtype=float)
        v = ndimCross(Q)
        assert v.shape == (2, 1)
    
    def test_ndimCross_determinant_property(self):
        """Test that magnitude equals volume of parallelepiped"""
        
        # For 3D case, |v| should equal volume of parallelepiped
        Q = np.array([[1, 0], [0, 1], [0, 0]], dtype=float)
        v = ndimCross(Q)
        magnitude = np.linalg.norm(v)
        
        # Volume is area of parallelogram formed by the two vectors
        # Area = |det([[1,0],[0,1]])| = 1
        expected_magnitude = 1.0
        assert np.abs(magnitude - expected_magnitude) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__]) 