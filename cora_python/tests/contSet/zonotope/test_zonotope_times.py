"""
test_zonotope_times - unit test function of times (scalar multiplication)

Syntax:
    python -m pytest test_zonotope_times.py

Inputs:
    -

Outputs:
    test results

Authors: Python translation by AI Assistant
Written: 2025
"""

import pytest
import numpy as np

from cora_python.contSet.zonotope import Zonotope


class TestZonotopeTimes:
    """Test class for zonotope times method (scalar multiplication)"""
    
    def test_positive_scalar(self):
        """Test multiplication by positive scalar"""
        Z = Zonotope(np.array([1, 2]), np.array([[1, 0], [0, 1]]))
        factor = 3
        Z_scaled = Z * factor
        
        expected_c = factor * Z.c
        expected_G = factor * Z.G
        
        np.testing.assert_array_equal(Z_scaled.c, expected_c)
        np.testing.assert_array_equal(Z_scaled.G, expected_G)
    
    def test_negative_scalar(self):
        """Test multiplication by negative scalar"""
        Z = Zonotope(np.array([1, 2]), np.array([[2, 1], [1, 2]]))
        factor = -2
        Z_scaled = Z * factor
        
        expected_c = factor * Z.c
        expected_G = factor * Z.G
        
        np.testing.assert_array_equal(Z_scaled.c, expected_c)
        np.testing.assert_array_equal(Z_scaled.G, expected_G)
    
    def test_zero_scalar(self):
        """Test multiplication by zero"""
        Z = Zonotope(np.array([5, -3]), np.array([[1, 2, 3], [4, 5, 6]]))
        Z_zero = Z * 0
        
        # Should result in origin zonotope
        expected_c = np.zeros_like(Z.c)
        expected_G = np.zeros_like(Z.G)
        
        np.testing.assert_array_equal(Z_zero.c, expected_c)
        np.testing.assert_array_equal(Z_zero.G, expected_G)
    
    def test_fractional_scalar(self):
        """Test multiplication by fractional scalar"""
        Z = Zonotope(np.array([4, 8]), np.array([[2, 4], [6, 8]]))
        factor = 0.5
        Z_scaled = Z * factor
        
        expected_c = factor * Z.c
        expected_G = factor * Z.G
        
        np.testing.assert_array_equal(Z_scaled.c, expected_c)
        np.testing.assert_array_equal(Z_scaled.G, expected_G)
    
    def test_commutative_property(self):
        """Test that scalar multiplication is commutative"""
        Z = Zonotope(np.array([1, 2]), np.array([[1, 0], [0, 1]]))
        factor = 2.5
        
        Z1 = Z * factor
        Z2 = factor * Z
        
        assert Z1.isequal(Z2)
    
    def test_empty_zonotope(self):
        """Test multiplication of empty zonotope"""
        Z_empty = Zonotope.empty(2)
        factor = 3
        Z_scaled = Z_empty * factor
        
        assert Z_scaled.isemptyobject()
        assert Z_scaled.dim() == 2
    
    def test_identity_scalar(self):
        """Test multiplication by 1 (identity)"""
        Z = Zonotope(np.array([3, -1]), np.array([[1, 2], [3, 4]]))
        Z_identity = Z * 1
        
        assert Z.isequal(Z_identity)
    
    def test_chained_multiplication(self):
        """Test chained scalar multiplication"""
        Z = Zonotope(np.array([1, 1]), np.array([[1, 0], [0, 1]]))
        Z_chained = Z * 2 * 3
        Z_direct = Z * 6
        
        assert Z_chained.isequal(Z_direct)


if __name__ == "__main__":
    test_instance = TestZonotopeTimes()
    
    # Run all tests
    test_instance.test_positive_scalar()
    test_instance.test_negative_scalar()
    test_instance.test_zero_scalar()
    test_instance.test_fractional_scalar()
    test_instance.test_commutative_property()
    test_instance.test_empty_zonotope()
    test_instance.test_identity_scalar()
    test_instance.test_chained_multiplication()
    
    print("All zonotope times tests passed!") 