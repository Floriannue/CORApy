"""
test_zonotope_copy - unit test function of copy

Syntax:
    python -m pytest test_zonotope_copy.py

Inputs:
    -

Outputs:
    test results

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 02-October-2024 (MATLAB)
Python translation: 2025
"""

import pytest
import numpy as np

from cora_python.contSet.zonotope import Zonotope


class TestZonotopeCopy:
    """Test class for zonotope copy method"""
    
    def test_basic_copy(self):
        """Test basic copy of zonotope"""
        # 2D zonotope
        Z = Zonotope(np.array([2]), np.array([[4]]))
        Z_copy = Z.copy()
        
        assert Z.isequal(Z_copy)
    
    def test_copy_independence(self):
        """Test that copy creates independent object"""
        Z = Zonotope(np.array([1, 2]), np.array([[1, 0], [0, 1]]))
        Z_copy = Z.copy()
        
        # Modify original
        Z.c[0] = 999
        
        # Copy should remain unchanged
        assert not Z.isequal(Z_copy)
        assert Z_copy.c[0] != 999
    
    def test_copy_different_dimensions(self):
        """Test copy with different dimensional zonotopes"""
        # 1D
        Z1 = Zonotope(np.array([5]), np.array([[2, 1, 3]]))
        Z1_copy = Z1.copy()
        assert Z1.isequal(Z1_copy)
        
        # 3D
        Z3 = Zonotope(np.array([1, 2, 3]), np.array([[1, 0, 2], [0, 1, 1], [0, 0, 1]]))
        Z3_copy = Z3.copy()
        assert Z3.isequal(Z3_copy)
    
    def test_copy_empty_zonotope(self):
        """Test copy of empty zonotope"""
        Z_empty = Zonotope.empty(2)
        Z_empty_copy = Z_empty.copy()
        
        assert Z_empty.isemptyobject()
        assert Z_empty_copy.isemptyobject()
        assert Z_empty.dim() == Z_empty_copy.dim()
    
    def test_copy_preserves_properties(self):
        """Test that copy preserves all properties"""
        Z = Zonotope(np.array([0, 0]), np.array([[2, 1, 0], [0, 1, 2]]))
        Z_copy = Z.copy()
        
        # Check dimensions
        assert Z.dim() == Z_copy.dim()
        
        # Check center and generators
        np.testing.assert_array_equal(Z.c, Z_copy.c)
        np.testing.assert_array_equal(Z.G, Z_copy.G)


if __name__ == "__main__":
    test_instance = TestZonotopeCopy()
    
    # Run all tests
    test_instance.test_basic_copy()
    test_instance.test_copy_independence()
    test_instance.test_copy_different_dimensions()
    test_instance.test_copy_empty_zonotope()
    test_instance.test_copy_preserves_properties()
    
    print("All zonotope copy tests passed!") 