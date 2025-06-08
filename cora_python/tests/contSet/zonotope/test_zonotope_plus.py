"""
test_zonotope_plus - unit test function for zonotope plus method

This module tests the plus method (Minkowski addition) for zonotope objects.

Authors:       Matthias Althoff, Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       26-July-2016 (MATLAB)
Python translation: 2025
"""

import pytest
import numpy as np
from cora_python.contSet.zonotope.zonotope import zonotope
from cora_python.contSet.zonotope.plus import plus
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAError


class TestZonotopePlus:
    """Test class for zonotope plus method"""
    
    def test_zonotope_plus_zonotope(self):
        """Test Minkowski addition of two zonotopes"""
        # Test case from MATLAB unit test
        Z1 = zonotope(np.array([-4, 1]), np.array([[-3, -2, -1], [2, 3, 4]]))
        Z2 = zonotope(np.array([1, -1]), np.array([[10], [-10]]))
        
        Z_plus = plus(Z1, Z2)
        
        # Expected result
        c_plus = np.array([-3, 0])
        G_plus = np.array([[-3, -2, -1, 10], [2, 3, 4, -10]])
        
        assert np.allclose(Z_plus.c, c_plus)
        assert np.allclose(Z_plus.G, G_plus)
    
    def test_zonotope_plus_zonotope_operator(self):
        """Test Minkowski addition using + operator"""
        Z1 = zonotope(np.array([-4, 1]), np.array([[-3, -2, -1], [2, 3, 4]]))
        Z2 = zonotope(np.array([1, -1]), np.array([[10], [-10]]))
        
        Z_plus = Z1 + Z2
        
        # Expected result
        c_plus = np.array([-3, 0])
        G_plus = np.array([[-3, -2, -1, 10], [2, 3, 4, -10]])
        
        assert np.allclose(Z_plus.c, c_plus)
        assert np.allclose(Z_plus.G, G_plus)
    
    def test_zonotope_plus_vector(self):
        """Test addition of zonotope with vector"""
        Z = zonotope(np.array([1, 2]), np.array([[1, 0], [0, 1]]))
        v = np.array([3, -1])
        
        Z_plus = plus(Z, v)
        
        # Expected: center shifts by vector, generators unchanged
        expected_c = np.array([4, 1])
        expected_G = np.array([[1, 0], [0, 1]])
        
        assert np.allclose(Z_plus.c, expected_c)
        assert np.allclose(Z_plus.G, expected_G)
    
    def test_zonotope_plus_vector_operator(self):
        """Test addition of zonotope with vector using + operator"""
        Z = zonotope(np.array([1, 2]), np.array([[1, 0], [0, 1]]))
        v = np.array([3, -1])
        
        Z_plus = Z + v
        
        # Expected: center shifts by vector, generators unchanged
        expected_c = np.array([4, 1])
        expected_G = np.array([[1, 0], [0, 1]])
        
        assert np.allclose(Z_plus.c, expected_c)
        assert np.allclose(Z_plus.G, expected_G)
    
    def test_vector_plus_zonotope(self):
        """Test addition of vector with zonotope (commutative)"""
        Z = zonotope(np.array([1, 2]), np.array([[1, 0], [0, 1]]))
        v = np.array([3, -1])
        
        Z_plus = v + Z
        
        # Expected: center shifts by vector, generators unchanged
        expected_c = np.array([4, 1])
        expected_G = np.array([[1, 0], [0, 1]])
        
        assert np.allclose(Z_plus.c, expected_c)
        assert np.allclose(Z_plus.G, expected_G)
    
    def test_zonotope_plus_scalar(self):
        """Test addition of zonotope with scalar"""
        Z = zonotope(np.array([1, 2]), np.array([[1, 0], [0, 1]]))
        s = 5
        
        Z_plus = plus(Z, s)
        
        # Expected: center shifts by scalar, generators unchanged
        expected_c = np.array([6, 7])
        expected_G = np.array([[1, 0], [0, 1]])
        
        assert np.allclose(Z_plus.c, expected_c)
        assert np.allclose(Z_plus.G, expected_G)
    
    def test_zonotope_plus_empty(self):
        """Test Minkowski addition with empty set"""
        Z1 = zonotope(np.array([-4, 1]), np.array([[-3, -2, -1], [2, 3, 4]]))
        Z_empty = zonotope.empty(2)
        
        Z_plus = plus(Z1, Z_empty)
        
        # Result should be empty
        assert Z_plus.is_empty()
    
    def test_empty_plus_zonotope(self):
        """Test Minkowski addition of empty set with zonotope"""
        Z1 = zonotope(np.array([-4, 1]), np.array([[-3, -2, -1], [2, 3, 4]]))
        Z_empty = zonotope.empty(2)
        
        Z_plus = plus(Z_empty, Z1)
        
        # Result should be empty
        assert Z_plus.is_empty()
    
    def test_dimension_mismatch(self):
        """Test addition with dimension mismatch raises error"""
        Z1 = zonotope(np.array([1, 2]), np.array([[1, 0], [0, 1]]))  # 2D
        Z2 = zonotope(np.array([1, 2, 3]), np.array([[1, 0], [0, 1], [1, 0]]))  # 3D
        
        with pytest.raises(CORAError):
            plus(Z1, Z2)
    
    def test_vector_dimension_mismatch(self):
        """Test addition with vector of wrong dimension raises error"""
        Z = zonotope(np.array([1, 2]), np.array([[1, 0], [0, 1]]))  # 2D
        v = np.array([1, 2, 3])  # 3D vector
        
        with pytest.raises(CORAError):
            plus(Z, v)
    
    def test_zonotope_plus_no_generators(self):
        """Test addition of zonotopes with no generators"""
        Z1 = zonotope(np.array([1, 2]), np.array([]).reshape(2, 0))
        Z2 = zonotope(np.array([3, -1]), np.array([]).reshape(2, 0))
        
        Z_plus = plus(Z1, Z2)
        
        # Expected: centers add, no generators
        expected_c = np.array([4, 1])
        expected_G = np.array([]).reshape(2, 0)
        
        assert np.allclose(Z_plus.c, expected_c)
        assert Z_plus.G.shape == expected_G.shape
    
    def test_zonotope_plus_mixed_generators(self):
        """Test addition where one zonotope has no generators"""
        Z1 = zonotope(np.array([1, 2]), np.array([[1, 0], [0, 1]]))
        Z2 = zonotope(np.array([3, -1]), np.array([]).reshape(2, 0))
        
        Z_plus = plus(Z1, Z2)
        
        # Expected: centers add, generators from Z1 only
        expected_c = np.array([4, 1])
        expected_G = np.array([[1, 0], [0, 1]])
        
        assert np.allclose(Z_plus.c, expected_c)
        assert np.allclose(Z_plus.G, expected_G)
    
    def test_1d_zonotope_plus(self):
        """Test addition of 1D zonotopes"""
        Z1 = zonotope(np.array([2]), np.array([[1, -1]]))
        Z2 = zonotope(np.array([3]), np.array([[0.5]]))
        
        Z_plus = plus(Z1, Z2)
        
        # Expected result
        expected_c = np.array([5])
        expected_G = np.array([[1, -1, 0.5]])
        
        assert np.allclose(Z_plus.c, expected_c)
        assert np.allclose(Z_plus.G, expected_G)
    
    def test_high_dimensional_plus(self):
        """Test addition of high-dimensional zonotopes"""
        n = 10
        c1 = np.random.randn(n)
        G1 = np.random.randn(n, 5)
        c2 = np.random.randn(n)
        G2 = np.random.randn(n, 3)
        
        Z1 = zonotope(c1, G1)
        Z2 = zonotope(c2, G2)
        
        Z_plus = plus(Z1, Z2)
        
        # Expected result
        expected_c = c1 + c2
        expected_G = np.hstack([G1, G2])
        
        assert np.allclose(Z_plus.c, expected_c)
        assert np.allclose(Z_plus.G, expected_G) 