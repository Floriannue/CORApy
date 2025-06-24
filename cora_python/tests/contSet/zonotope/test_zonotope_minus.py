"""
test_zonotope_minus - unit test function of minus

Syntax:
    python -m pytest test_zonotope_minus.py

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


class TestZonotopeMinusOp:
    """Test class for zonotope minus method"""
    
    def test_zonotope_minus_zonotope(self):
        """Test subtraction of two zonotopes"""
        Z1 = Zonotope(np.array([2, 3]), np.array([[1, 0], [0, 1]]))
        Z2 = Zonotope(np.array([1, 1]), np.array([[0.5, 0], [0, 0.5]]))
        
        Z_diff = Z1 - Z2
        
        # Center should be difference of centers
        expected_center = Z1.c - Z2.c
        np.testing.assert_array_equal(Z_diff.c, expected_center)
        
        # Should contain difference of sample points
        points1 = Z1.randPoint_(10)
        points2 = Z2.randPoint_(10)
        
        for i in range(10):
            diff_point = points1[:, i] - points2[:, 0]  # First point from Z2
            assert Z_diff.contains_(diff_point)
    
    def test_zonotope_minus_vector(self):
        """Test subtraction of vector from zonotope"""
        Z = Zonotope(np.array([3, 4]), np.array([[1, 0], [0, 1]]))
        v = np.array([1, 2])
        
        Z_translated = Z - v
        
        # Should be translation by -v
        expected_center = Z.c.flatten() - v
        np.testing.assert_array_equal(Z_translated.c.flatten(), expected_center)
        
        # Generators should remain the same
        np.testing.assert_array_equal(Z_translated.G, Z.G)
    
    def test_vector_minus_zonotope(self):
        """Test subtraction of zonotope from vector"""
        v = np.array([5, 6])
        Z = Zonotope(np.array([1, 2]), np.array([[1, 0], [0, 1]]))
        
        Z_result = v - Z
        
        # Should be v - Z = v + (-Z)
        Z_negated = -Z
        Z_expected = v + Z_negated
        
        assert Z_result.isequal(Z_expected)
    
    def test_minus_with_scalar(self):
        """Test subtraction with scalar"""
        Z = Zonotope(np.array([3, 4]), np.array([[1, 0], [0, 1]]))
        scalar = 2
        
        Z_result = Z - scalar
        
        # Should subtract scalar from center
        expected_center = Z.c.flatten() - scalar
        np.testing.assert_array_equal(Z_result.c.flatten(), expected_center)
        
        # Generators should remain the same
        np.testing.assert_array_equal(Z_result.G, Z.G)
    
    def test_minus_empty_zonotope(self):
        """Test subtraction with empty zonotope"""
        Z = Zonotope(np.array([1, 2]), np.array([[1, 0], [0, 1]]))
        Z_empty = Zonotope.empty(2)
        
        Z_result = Z - Z_empty
        
        # Result should be empty
        assert Z_result.isemptyobject()
        
        Z_result2 = Z_empty - Z
        assert Z_result2.isemptyobject()
    
    def test_minus_origin(self):
        """Test subtraction with origin zonotope"""
        Z = Zonotope(np.array([2, 3]), np.array([[1, 0], [0, 1]]))
        Z_origin = Zonotope.origin(2)
        
        Z_result = Z - Z_origin
        
        # Z - {0} should equal Z
        assert Z.isequal(Z_result)
    
    def test_minus_self(self):
        """Test subtracting zonotope from itself"""
        Z = Zonotope(np.array([2, 3]), np.array([[1, 0.5], [0, 1]]))
        
        Z_result = Z - Z
        
        # Should contain the origin
        assert Z_result.contains_(np.array([0, 0]))
    
    def test_minus_associativity(self):
        """Test that (Z1 - Z2) - Z3 and Z1 - (Z2 + Z3) have similar containment"""
        Z1 = Zonotope(np.array([5, 6]), np.array([[1, 0], [0, 1]]))
        Z2 = Zonotope(np.array([2, 1]), np.array([[0.5, 0], [0, 0.5]]))
        Z3 = Zonotope(np.array([1, 1]), np.array([[0.3, 0], [0, 0.3]]))
        
        # (Z1 - Z2) - Z3
        Z_temp = Z1 - Z2
        Z_result1 = Z_temp - Z3
        
        # Z1 - (Z2 + Z3)
        Z_sum = Z2 + Z3
        Z_result2 = Z1 - Z_sum
        
        # Both should contain the point Z1.center - Z2.center - Z3.center
        target_point = Z1.c.flatten() - Z2.c.flatten() - Z3.c.flatten()
        assert Z_result1.contains_(target_point)
        assert Z_result2.contains_(target_point)
    
    def test_minus_1d(self):
        """Test subtraction of 1D zonotopes"""
        Z1 = Zonotope(np.array([5]), np.array([[2, 1]]))
        Z2 = Zonotope(np.array([2]), np.array([[1]]))
        
        Z_diff = Z1 - Z2
        
        assert Z_diff.dim() == 1
        
        # Should contain the difference of centers
        center_diff = Z1.c.flatten() - Z2.c.flatten()
        assert Z_diff.contains_(center_diff)
    
    def test_minus_different_dimensions_error(self):
        """Test that subtraction of different dimensional zonotopes raises error"""
        Z1 = Zonotope(np.array([1, 2]), np.array([[1, 0], [0, 1]]))
        Z2 = Zonotope(np.array([1, 2, 3]), np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        
        with pytest.raises(Exception):
            Z1 - Z2


if __name__ == "__main__":
    test_instance = TestZonotopeMinusOp()
    
    # Run all tests
    test_instance.test_zonotope_minus_zonotope()
    test_instance.test_zonotope_minus_vector()
    test_instance.test_vector_minus_zonotope()
    test_instance.test_minus_with_scalar()
    test_instance.test_minus_empty_zonotope()
    test_instance.test_minus_origin()
    test_instance.test_minus_self()
    test_instance.test_minus_associativity()
    test_instance.test_minus_1d()
    test_instance.test_minus_different_dimensions_error()
    
    print("All zonotope minus tests passed!") 