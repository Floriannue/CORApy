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
from cora_python.contSet.zonotope.zonotope import zonotope
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAError


class TestZonotope:
    """Test class for zonotope constructor and basic methods"""
    
    def test_constructor_center_generators(self):
        """Test constructor with center and generator matrix"""
        # 2D zonotope
        c = np.array([1, 2])
        G = np.array([[1, 0, 1], [0, 1, -1]])
        Z = zonotope(c, G)
        
        assert np.allclose(Z.c, c)
        assert np.allclose(Z.G, G)
        assert Z.dim() == 2
        assert Z.precedence == 110
    
    def test_constructor_combined_matrix(self):
        """Test constructor with combined matrix [c, G]"""
        # Combined matrix
        Z_mat = np.array([[1, 1, 0, 1], [2, 0, 1, -1]])
        Z = zonotope(Z_mat)
        
        expected_c = np.array([1, 2])
        expected_G = np.array([[1, 0, 1], [0, 1, -1]])
        
        assert np.allclose(Z.c, expected_c)
        assert np.allclose(Z.G, expected_G)
        assert Z.dim() == 2
    
    def test_constructor_center_only(self):
        """Test constructor with center only (no generators)"""
        c = np.array([1, 2])
        Z = zonotope(c, np.array([]).reshape(2, 0))
        
        assert np.allclose(Z.c, c)
        assert Z.G.shape == (2, 0)
        assert Z.dim() == 2
    
    def test_constructor_copy(self):
        """Test copy constructor"""
        c = np.array([1, 2])
        G = np.array([[1, 0], [0, 1]])
        Z1 = zonotope(c, G)
        Z2 = zonotope(Z1)
        
        assert np.allclose(Z2.c, Z1.c)
        assert np.allclose(Z2.G, Z1.G)
        assert Z2.dim() == Z1.dim()
        
        # Ensure it's a deep copy
        Z2.c[0] = 999
        assert Z1.c[0] != 999
    
    def test_constructor_empty_args(self):
        """Test constructor with no arguments raises error"""
        with pytest.raises(CORAError):
            zonotope()
    
    def test_constructor_dimension_mismatch(self):
        """Test constructor with dimension mismatch raises error"""
        c = np.array([1, 2])  # 2D center
        G = np.array([[1, 0, 1]])  # 1D generators
        
        with pytest.raises(CORAError):
            zonotope(c, G)
    
    def test_constructor_nan_values(self):
        """Test constructor with NaN values raises error"""
        c = np.array([1, np.nan])
        G = np.array([[1, 0], [0, 1]])
        
        with pytest.raises(CORAError):
            zonotope(c, G)
    
    def test_empty_zonotope(self):
        """Test empty zonotope creation"""
        Z_empty = zonotope.empty(2)
        
        assert Z_empty.dim() == 2
        assert Z_empty.is_empty()
        assert Z_empty.c.shape == (0,)  # Empty center for empty set
        assert Z_empty.G.shape == (2, 0)
    
    def test_origin_zonotope(self):
        """Test origin zonotope creation"""
        Z_origin = zonotope.origin(3)
        
        assert Z_origin.dim() == 3
        assert not Z_origin.is_empty()
        assert np.allclose(Z_origin.c, np.zeros(3))
        assert Z_origin.G.shape == (3, 0)
    
    def test_dim_method(self):
        """Test dimension method"""
        # Non-empty zonotope
        Z = zonotope(np.array([1, 2]), np.array([[1, 0], [0, 1]]))
        assert Z.dim() == 2
        
        # Empty zonotope
        Z_empty = zonotope.empty(3)
        assert Z_empty.dim() == 3
    
    def test_is_empty_method(self):
        """Test is_empty method"""
        # Non-empty zonotope
        Z = zonotope(np.array([1, 2]), np.array([[1, 0], [0, 1]]))
        assert not Z.is_empty()
        
        # Empty zonotope
        Z_empty = zonotope.empty(2)
        assert Z_empty.is_empty()
    
    def test_string_representation(self):
        """Test string representation methods"""
        Z = zonotope(np.array([1, 2]), np.array([[1, 0], [0, 1]]))
        str_repr = str(Z)
        
        assert "zonotope" in str_repr
        assert "dimension: 2" in str_repr
        assert "generators: 2" in str_repr
        
        # Empty zonotope
        Z_empty = zonotope.empty(2)
        str_repr_empty = str(Z_empty)
        assert "empty" in str_repr_empty
    
    def test_1d_zonotope(self):
        """Test 1D zonotope"""
        c = np.array([5])
        G = np.array([[2, -1, 3]])
        Z = zonotope(c, G)
        
        assert Z.dim() == 1
        assert np.allclose(Z.c, c)
        assert np.allclose(Z.G, G)
    
    def test_high_dimensional_zonotope(self):
        """Test high-dimensional zonotope"""
        n = 10
        c = np.random.randn(n)
        G = np.random.randn(n, 5)
        Z = zonotope(c, G)
        
        assert Z.dim() == n
        assert np.allclose(Z.c, c)
        assert np.allclose(Z.G, G) 