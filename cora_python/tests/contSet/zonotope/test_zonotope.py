"""
test_zonotope - unit test function for zonotope class

This module tests the zonotope class constructor and basic functionality.

Authors:       Matthias Althoff, Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       26-July-2016 (MATLAB)
Python translation: 2025
"""

import pytest
import numpy as np
from cora_python.contSet.zonotope import Zonotope


class TestZonotope:
    """Test class for zonotope constructor and basic methods"""
    
    def test_constructor_center_generators(self):
        """Test constructor with center and generator matrix"""
        # 2D zonotope
        c = np.array([[1], [2]])
        G = np.array([[1, 0, 1], [0, 1, -1]])
        Z = Zonotope(c, G)
        
        assert np.allclose(Z.c, c)
        assert np.allclose(Z.G, G)
        assert Z.dim() == 2
    
    def test_constructor_combined_matrix(self):
        """Test constructor with combined matrix [c, G]"""
        # Combined matrix
        Z_mat = np.array([[1, 1, 0, 1], [2, 0, 1, -1]])
        Z = Zonotope(Z_mat)
        
        expected_c = np.array([[1], [2]])
        expected_G = np.array([[1, 0, 1], [0, 1, -1]])
        
        assert np.allclose(Z.c, expected_c)
        assert np.allclose(Z.G, expected_G)
        assert Z.dim() == 2
    
    def test_constructor_center_only(self):
        """Test constructor with center only (no generators)"""
        c = np.array([[1], [2]])
        Z = Zonotope(c)
        
        assert np.allclose(Z.c, c)
        assert Z.G.shape == (2, 0)
        assert Z.dim() == 2
    
    def test_constructor_copy(self):
        """Test copy constructor"""
        c = np.array([[1], [2]])
        G = np.array([[1, 0], [0, 1]])
        Z1 = Zonotope(c, G)
        Z2 = Zonotope(Z1.c.copy(), Z1.G.copy())
        
        assert np.allclose(Z2.c, Z1.c)
        assert np.allclose(Z2.G, Z1.G)
        assert Z2.dim() == Z1.dim()
        
        # Ensure it's a deep copy
        Z2.c[0] = 999
        assert Z1.c[0] != 999
    
    def test_empty_zonotope(self):
        """Test empty zonotope creation"""
        Z_empty = Zonotope.empty(2)
        
        assert Z_empty.dim() == 2
        assert Z_empty.isemptyobject()
        assert Z_empty.representsa_('emptySet')
    
    def test_origin_zonotope(self):
        """Test origin zonotope creation"""
        Z_origin = Zonotope.origin(3)
        
        assert Z_origin.dim() == 3
        assert not Z_origin.isemptyobject()
        assert np.allclose(Z_origin.c, np.zeros((3, 1)))
        assert Z_origin.G.shape == (3, 0)
    
    def test_dim_method(self):
        """Test dimension method"""
        # Non-empty zonotope
        Z = Zonotope(np.array([[1], [2]]), np.array([[1, 0], [0, 1]]))
        assert Z.dim() == 2
        
        # Empty zonotope
        Z_empty = Zonotope.empty(3)
        assert Z_empty.dim() == 3
    
    def test_isemptyobject_method(self):
        """Test isemptyobject method"""
        # Non-empty zonotope
        Z = Zonotope(np.array([[1], [2]]), np.array([[1, 0], [0, 1]]))
        assert not Z.isemptyobject()
        
        # Empty zonotope
        Z_empty = Zonotope.empty(2)
        assert Z_empty.isemptyobject()
    
    def test_string_representation(self):
        """Test string representation methods"""
        Z = Zonotope(np.array([[1], [2]]), np.array([[1, 0], [0, 1]]))
        str_repr = Z.display()
        
        assert isinstance(str_repr, str)
        
        # Empty zonotope
        Z_empty = Zonotope.empty(2)
        str_repr_empty = Z_empty.display()
        assert isinstance(str_repr_empty, str)
    
    def test_1d_zonotope(self):
        """Test 1D zonotope"""
        c = np.array([[5]])
        G = np.array([[2, -1, 3]])
        Z = Zonotope(c, G)
        
        assert Z.dim() == 1
        assert np.allclose(Z.c, c)
        assert np.allclose(Z.G, G)
    
    def test_high_dimensional_zonotope(self):
        """Test high-dimensional zonotope"""
        n = 10
        c = np.random.randn(n, 1)
        G = np.random.randn(n, 5)
        Z = Zonotope(c, G)
        
        assert Z.dim() == n
        assert np.allclose(Z.c, c)
        assert np.allclose(Z.G, G)
    
    def test_center_method(self):
        """Test center method"""
        c = np.array([[1], [2], [3]])
        G = np.array([[1, 0], [0, 1], [1, 1]])
        Z = Zonotope(c, G)
        
        center = Z.center()
        assert np.allclose(center, c)
    
    def test_plus_method(self):
        """Test plus (addition) method"""
        # Two zonotopes
        Z1 = Zonotope(np.array([[1], [2]]), np.array([[1, 0], [0, 1]]))
        Z2 = Zonotope(np.array([[2], [1]]), np.array([[0, 1], [1, 0]]))
        
        Z_sum = Z1.plus(Z2)
        expected_c = np.array([[3], [3]])
        expected_G = np.array([[1, 0, 0, 1], [0, 1, 1, 0]])
        
        assert np.allclose(Z_sum.c, expected_c)
        assert np.allclose(Z_sum.G, expected_G)
        
        # Test with overloaded operator
        Z_sum2 = Z1 + Z2
        assert np.allclose(Z_sum2.c, Z_sum.c)
        assert np.allclose(Z_sum2.G, Z_sum.G)
    
    def test_mtimes_method(self):
        """Test mtimes (multiplication) method"""
        Z = Zonotope(np.array([[1], [2]]), np.array([[1, 0], [0, 1]]))
        
        # Scalar multiplication
        Z_scaled = Z.mtimes(2)
        expected_c = np.array([[2], [4]])
        expected_G = np.array([[2, 0], [0, 2]])
        
        assert np.allclose(Z_scaled.c, expected_c)
        assert np.allclose(Z_scaled.G, expected_G)
        
        # Matrix multiplication
        M = np.array([[1, 0], [0, -1], [1, 1]])
        Z_transformed = M * Z  # Should use overloaded operator
        
        assert Z_transformed.dim() == 3
        assert np.allclose(Z_transformed.c, M @ Z.c)
        assert np.allclose(Z_transformed.G, M @ Z.G)


if __name__ == "__main__":
    test = TestZonotope()
    test.test_constructor_center_generators()
    test.test_constructor_combined_matrix()
    test.test_constructor_center_only()
    test.test_empty_zonotope()
    test.test_origin_zonotope()
    test.test_dim_method()
    test.test_isemptyobject_method()
    test.test_center_method()
    test.test_plus_method()
    test.test_mtimes_method()
    print("All zonotope tests passed!") 