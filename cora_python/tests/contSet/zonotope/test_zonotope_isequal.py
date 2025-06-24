"""
test_zonotope_isequal - unit test function of isequal

Syntax:
    python -m pytest test_zonotope_isequal.py

Inputs:
    -

Outputs:
    test results

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 17-September-2019 (MATLAB)
Last update: 21-April-2020 (MATLAB), 09-August-2020 (MATLAB)
Python translation: 2025
"""

import pytest
import numpy as np

from cora_python.contSet.zonotope import Zonotope


class TestZonotopeIsequal:
    """Test class for zonotope isequal method"""
    
    def test_1d_minimal_and_non_minimal(self):
        """Test 1D zonotopes, minimal and non-minimal"""
        Z1 = Zonotope(np.array([4]), np.array([[1, 3, -2]]))
        Z2 = Zonotope(np.array([4]), np.array([[6]]))
        
        assert Z1.isequal(Z2)
        assert Z2.isequal(Z1)
    
    def test_2d_different_order_of_generators(self):
        """Test 2D zonotopes with different order of generators"""
        Z1 = Zonotope(
            np.array([1, 1]), 
            np.array([[1, 2, 5, 3, 3],
                      [2, 3, 0, 4, 1]])
        )
        Z2 = Zonotope(
            np.array([1, 1]), 
            np.array([[2, 1, 3, 5, 3],
                      [3, 2, 4, 0, 1]])
        )
        
        assert Z1.isequal(Z2)
        assert Z2.isequal(Z1)
    
    def test_2d_different_sign(self):
        """Test 2D zonotopes with different signs"""
        Z1 = Zonotope(
            np.array([0, 1]), 
            np.array([[1, 0, -1],
                      [1, 1, 2]])
        )
        Z2 = Zonotope(
            np.array([0, 1]), 
            np.array([[-1, 0, -1],
                      [-1, -1, 2]])
        )
        
        assert Z1.isequal(Z2)
        assert Z2.isequal(Z1)
    
    def test_3d_with_zero_generator(self):
        """Test 3D zonotopes with zero generators"""
        Z1 = Zonotope(
            np.array([1, 5, -1]), 
            np.array([[2, 4],
                      [6, 0],
                      [4, 8]])
        )
        Z2 = Zonotope(
            np.array([1, 4, -1]), 
            np.array([[2, 4],
                      [6, 0],
                      [4, 8]])
        )
        Z3 = Zonotope(
            np.array([1, 5, -1]), 
            np.array([[2, 0, 4],
                      [6, 0, 0],
                      [4, 0, 8]])
        )
        
        assert Z1.isequal(Z3)
        assert Z3.isequal(Z1)
        assert not Z1.isequal(Z2)
        assert not Z2.isequal(Z1)
    
    def test_different_types(self):
        """Test isequal with different object types"""
        Z = Zonotope(np.array([1, 2]), np.array([[1, 0], [0, 1]]))
        
        # Test with non-zonotope object
        assert not Z.isequal("not a zonotope")
        assert not Z.isequal(None)
        assert not Z.isequal(42)
    
    def test_empty_zonotopes(self):
        """Test isequal with empty zonotopes"""
        Z_empty1 = Zonotope.empty(2)
        Z_empty2 = Zonotope.empty(2)
        Z_empty3 = Zonotope.empty(3)
        Z_normal = Zonotope(np.array([1, 2]), np.array([[1, 0], [0, 1]]))
        
        assert Z_empty1.isequal(Z_empty2)
        assert not Z_empty1.isequal(Z_empty3)  # Different dimensions
        assert not Z_empty1.isequal(Z_normal)
        assert not Z_normal.isequal(Z_empty1)
    
    def test_tolerance(self):
        """Test isequal with different tolerances"""
        Z1 = Zonotope(np.array([1.0, 2.0]), np.array([[1.0, 0.0], [0.0, 1.0]]))
        Z2 = Zonotope(np.array([1.0001, 2.0001]), np.array([[1.0001, 0.0001], [0.0001, 1.0001]]))
        
        # Should be different with default tolerance
        assert not Z1.isequal(Z2)
        
        # Should be equal with relaxed tolerance
        assert Z1.isequal(Z2, tol=1e-3)


if __name__ == "__main__":
    test_instance = TestZonotopeIsequal()
    
    # Run all tests
    test_instance.test_1d_minimal_and_non_minimal()
    test_instance.test_2d_different_order_of_generators()
    test_instance.test_2d_different_sign()
    test_instance.test_3d_with_zero_generator()
    test_instance.test_different_types()
    test_instance.test_empty_zonotopes()
    test_instance.test_tolerance()
    
    print("All zonotope isequal tests passed!") 