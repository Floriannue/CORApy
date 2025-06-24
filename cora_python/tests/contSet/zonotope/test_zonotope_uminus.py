"""
test_zonotope_uminus - unit test function of uminus

Syntax:
    python -m pytest test_zonotope_uminus.py

Inputs:
    -

Outputs:
    test results

Authors: Tobias Ladner (MATLAB)
         Python translation by AI Assistant
Written: 06-April-2023 (MATLAB)
Python translation: 2025
"""

import pytest
import numpy as np

from cora_python.contSet.zonotope import Zonotope


class TestZonotopeUminus:
    """Test class for zonotope uminus method"""
    
    def test_basic_negation(self):
        """Test basic negation of zonotope"""
        # Initialize
        c = np.array([0, 0])
        G = np.array([[2, 0, 2], 
                      [0, 2, 2]])
        Z = Zonotope(c, G)
        
        # Negate
        nZ = Z.uminus()
        
        # Check that center and generators are negated
        expected_c = -c
        expected_G = -G
        
        np.testing.assert_array_equal(nZ.c.flatten(), expected_c)
        np.testing.assert_array_equal(nZ.G, expected_G)
    
    def test_operator_overload(self):
        """Test that -Z works using operator overload"""
        c = np.array([1, -2])
        G = np.array([[1, 0], 
                      [0, 1]])
        Z = Zonotope(c, G)
        
        # Use operator overload
        nZ = -Z
        
        # Check result
        np.testing.assert_array_equal(nZ.c.flatten(), -c)
        np.testing.assert_array_equal(nZ.G, -G)
    
    def test_compare_with_scalar_multiplication(self):
        """Test that -Z equals -1 * Z"""
        c = np.array([0, 0])
        G = np.array([[2, 0, 2], 
                      [0, 2, 2]])
        Z = Zonotope(c, G)
        
        nZ1 = -Z
        nZ2 = -1 * Z
        
        # Should be equal
        assert nZ1.isequal(nZ2)
    
    def test_empty_case(self):
        """Test uminus with empty zonotope"""
        Z_empty = Zonotope.empty(2)
        nZ_empty = -Z_empty
        
        # Should still be empty
        assert nZ_empty.isemptyobject()
        assert nZ_empty.dim() == 2
    
    def test_different_dimensions(self):
        """Test uminus with different dimensional zonotopes"""
        # 1D case
        Z1 = Zonotope(np.array([3]), np.array([[2, 1]]))
        nZ1 = -Z1
        np.testing.assert_array_equal(nZ1.c.flatten(), np.array([-3]))
        np.testing.assert_array_equal(nZ1.G, np.array([[-2, -1]]))
        
        # 3D case
        Z3 = Zonotope(np.array([1, 2, 3]), np.array([[1, 0, 0], 
                                                      [0, 1, 0], 
                                                      [0, 0, 1]]))
        nZ3 = -Z3
        expected_c = np.array([-1, -2, -3])
        expected_G = np.array([[-1, 0, 0], 
                               [0, -1, 0], 
                               [0, 0, -1]])
        np.testing.assert_array_equal(nZ3.c.flatten(), expected_c)
        np.testing.assert_array_equal(nZ3.G, expected_G)
    
    def test_double_negation(self):
        """Test that --Z equals Z"""
        c = np.array([1, -1])
        G = np.array([[2, 1], 
                      [1, 2]])
        Z = Zonotope(c, G)
        
        # Double negation
        Z_double_neg = -(-Z)
        
        # Should equal original
        assert Z.isequal(Z_double_neg)


if __name__ == "__main__":
    test_instance = TestZonotopeUminus()
    
    # Run all tests
    test_instance.test_basic_negation()
    test_instance.test_operator_overload()
    test_instance.test_compare_with_scalar_multiplication()
    test_instance.test_empty_case()
    test_instance.test_different_dimensions()
    test_instance.test_double_negation()
    
    print("All zonotope uminus tests passed!") 