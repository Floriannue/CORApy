"""
test_zonotope_mtimes - unit test function for zonotope mtimes method

This module tests the mtimes method (matrix multiplication) for zonotope objects.

Authors:       Matthias Althoff, Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       26-July-2016 (MATLAB)
Python translation: 2025
"""

import pytest
import numpy as np
from cora_python.contSet.zonotope.zonotope import zonotope
from cora_python.contSet.zonotope.mtimes import mtimes
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAError


class TestZonotopeMtimes:
    """Test class for zonotope mtimes method"""
    
    def test_matrix_times_empty_zonotope(self):
        """Test matrix multiplication with empty zonotope"""
        Z = zonotope.empty(2)
        
        # Square matrix
        M = np.array([[-1, 2], [3, -4]])
        Z_mtimes = mtimes(M, Z)
        assert Z_mtimes.is_empty() and Z_mtimes.dim() == 2
        
        # Projection onto subspace
        M = np.array([[-1, 2]])
        Z_mtimes = mtimes(M, Z)
        assert Z_mtimes.is_empty() and Z_mtimes.dim() == 1
        
        # Projection to higher-dimensional space
        M = np.array([[-1, 2], [3, -4], [5, 6]])
        Z_mtimes = mtimes(M, Z)
        assert Z_mtimes.is_empty() and Z_mtimes.dim() == 3
    
    def test_matrix_times_nonempty_zonotope(self):
        """Test matrix multiplication with non-empty zonotope"""
        Z = zonotope(np.array([-4, 1]), np.array([[-3, -2, -1], [2, 3, 4]]))
        
        # Square matrix
        M = np.array([[-1, 2], [3, -4]])
        Z_mtimes = mtimes(M, Z)
        Z_true = zonotope(np.array([6, -16]), np.array([[7, 8, 9], [-17, -18, -19]]))
        
        assert np.allclose(Z_mtimes.c, Z_true.c)
        assert np.allclose(Z_mtimes.G, Z_true.G)
    
    def test_matrix_times_zonotope_operator(self):
        """Test matrix multiplication using * operator"""
        Z = zonotope(np.array([-4, 1]), np.array([[-3, -2, -1], [2, 3, 4]]))
        M = np.array([[-1, 2], [3, -4]])
        
        Z_mtimes = M * Z
        Z_true = zonotope(np.array([6, -16]), np.array([[7, 8, 9], [-17, -18, -19]]))
        
        assert np.allclose(Z_mtimes.c, Z_true.c)
        assert np.allclose(Z_mtimes.G, Z_true.G)
    
    def test_projection_onto_subspace(self):
        """Test projection onto subspace"""
        Z = zonotope(np.array([-4, 1]), np.array([[-3, -2, -1], [2, 3, 4]]))
        M = np.array([[-1, 2]])
        
        Z_mtimes = mtimes(M, Z)
        Z_true = zonotope(np.array([6]), np.array([[7, 8, 9]]))
        
        assert np.allclose(Z_mtimes.c, Z_true.c)
        assert np.allclose(Z_mtimes.G, Z_true.G)
    
    def test_projection_to_higher_dimension(self):
        """Test projection to higher-dimensional space"""
        Z = zonotope(np.array([-4, 1]), np.array([[-3, -2, -1], [2, 3, 4]]))
        M = np.array([[-1, 2], [1, 0], [0, 1]])
        
        Z_mtimes = mtimes(M, Z)
        Z_true = zonotope(np.array([6, -4, 1]), np.array([[7, 8, 9], [-3, -2, -1], [2, 3, 4]]))
        
        assert np.allclose(Z_mtimes.c, Z_true.c)
        assert np.allclose(Z_mtimes.G, Z_true.G)
    
    def test_zonotope_times_scalar(self):
        """Test zonotope multiplication with scalar"""
        Z = zonotope(np.array([-4, 1]), np.array([[-3, -2, -1], [2, 3, 4]]))
        scalar = 2
        
        Z_mtimes = mtimes(Z, scalar)
        Z_true = zonotope(np.array([-8, 2]), np.array([[-6, -4, -2], [4, 6, 8]]))
        
        assert np.allclose(Z_mtimes.c, Z_true.c)
        assert np.allclose(Z_mtimes.G, Z_true.G)
    
    def test_zonotope_times_scalar_operator(self):
        """Test zonotope multiplication with scalar using * operator"""
        Z = zonotope(np.array([-4, 1]), np.array([[-3, -2, -1], [2, 3, 4]]))
        scalar = 2
        
        Z_mtimes = Z * scalar
        Z_true = zonotope(np.array([-8, 2]), np.array([[-6, -4, -2], [4, 6, 8]]))
        
        assert np.allclose(Z_mtimes.c, Z_true.c)
        assert np.allclose(Z_mtimes.G, Z_true.G)
    
    def test_scalar_times_zonotope(self):
        """Test scalar multiplication with zonotope (commutative)"""
        Z = zonotope(np.array([-4, 1]), np.array([[-3, -2, -1], [2, 3, 4]]))
        scalar = 2
        
        Z_mtimes = scalar * Z
        Z_true = zonotope(np.array([-8, 2]), np.array([[-6, -4, -2], [4, 6, 8]]))
        
        assert np.allclose(Z_mtimes.c, Z_true.c)
        assert np.allclose(Z_mtimes.G, Z_true.G)
    
    def test_zero_scalar_multiplication(self):
        """Test multiplication with zero scalar"""
        Z = zonotope(np.array([1, 2]), np.array([[1, 0], [0, 1]]))
        scalar = 0
        
        Z_mtimes = mtimes(Z, scalar)
        
        # Result should be origin
        assert np.allclose(Z_mtimes.c, np.zeros(2))
        assert np.allclose(Z_mtimes.G, np.zeros((2, 2)))
    
    def test_negative_scalar_multiplication(self):
        """Test multiplication with negative scalar"""
        Z = zonotope(np.array([1, 2]), np.array([[1, 0], [0, 1]]))
        scalar = -3
        
        Z_mtimes = mtimes(Z, scalar)
        
        # Expected result
        expected_c = np.array([-3, -6])
        expected_G = np.array([[-3, 0], [0, -3]])
        
        assert np.allclose(Z_mtimes.c, expected_c)
        assert np.allclose(Z_mtimes.G, expected_G)
    
    def test_identity_matrix_multiplication(self):
        """Test multiplication with identity matrix"""
        Z = zonotope(np.array([1, 2]), np.array([[1, 0], [0, 1]]))
        I = np.eye(2)
        
        Z_mtimes = mtimes(I, Z)
        
        # Result should be unchanged
        assert np.allclose(Z_mtimes.c, Z.c)
        assert np.allclose(Z_mtimes.G, Z.G)
    
    def test_dimension_mismatch(self):
        """Test multiplication with dimension mismatch raises error"""
        Z = zonotope(np.array([1, 2]), np.array([[1, 0], [0, 1]]))  # 2D
        M = np.array([[1, 0, 0], [0, 1, 0]])  # 2x3 matrix
        
        with pytest.raises(CORAError):
            mtimes(M, Z)
    
    def test_1d_zonotope_multiplication(self):
        """Test multiplication with 1D zonotope"""
        Z = zonotope(np.array([5]), np.array([[2, -1]]))
        scalar = 3
        
        Z_mtimes = mtimes(Z, scalar)
        
        # Expected result
        expected_c = np.array([15])
        expected_G = np.array([[6, -3]])
        
        assert np.allclose(Z_mtimes.c, expected_c)
        assert np.allclose(Z_mtimes.G, expected_G)
    
    def test_matrix_with_1d_zonotope(self):
        """Test matrix multiplication with 1D zonotope"""
        Z = zonotope(np.array([5]), np.array([[2, -1]]))
        M = np.array([[2], [3]])  # 2x1 matrix
        
        Z_mtimes = mtimes(M, Z)
        
        # Expected result
        expected_c = np.array([10, 15])
        expected_G = np.array([[4, -2], [6, -3]])
        
        assert np.allclose(Z_mtimes.c, expected_c)
        assert np.allclose(Z_mtimes.G, expected_G)
    
    def test_zonotope_no_generators(self):
        """Test multiplication with zonotope having no generators"""
        Z = zonotope(np.array([1, 2]), np.array([]).reshape(2, 0))
        M = np.array([[2, 1], [0, 3]])
        
        Z_mtimes = mtimes(M, Z)
        
        # Expected result: transformed center, no generators
        expected_c = np.array([4, 6])
        expected_G = np.array([]).reshape(2, 0)
        
        assert np.allclose(Z_mtimes.c, expected_c)
        assert Z_mtimes.G.shape == expected_G.shape
    
    def test_high_dimensional_multiplication(self):
        """Test multiplication with high-dimensional zonotope"""
        n = 10
        m = 5
        c = np.random.randn(n)
        G = np.random.randn(n, 7)
        M = np.random.randn(m, n)
        
        Z = zonotope(c, G)
        Z_mtimes = mtimes(M, Z)
        
        # Expected result
        expected_c = M @ c
        expected_G = M @ G
        
        assert np.allclose(Z_mtimes.c, expected_c)
        assert np.allclose(Z_mtimes.G, expected_G)
    
    def test_vector_as_matrix(self):
        """Test multiplication with vector (treated as matrix)"""
        Z = zonotope(np.array([1, 2]), np.array([[1, 0], [0, 1]]))
        v = np.array([2, 3])  # Row vector
        
        Z_mtimes = mtimes(v.reshape(1, -1), Z)
        
        # Expected result: 1D zonotope
        expected_c = np.array([8])  # 2*1 + 3*2
        expected_G = np.array([[2, 3]])  # [2*1+3*0, 2*0+3*1]
        
        assert np.allclose(Z_mtimes.c, expected_c)
        assert np.allclose(Z_mtimes.G, expected_G) 