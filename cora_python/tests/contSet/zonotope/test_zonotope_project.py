"""
test_zonotope_project - unit test function of project

Syntax:
    python -m pytest test_zonotope_project.py

Inputs:
    -

Outputs:
    test results

Authors: Matthias Althoff, Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 26-July-2016 (MATLAB)
Last update: 09-August-2020 (MATLAB)
Python translation: 2025
"""

import pytest
import numpy as np

from cora_python.contSet.zonotope import Zonotope


class TestZonotopeProject:
    """Test class for zonotope project method"""
    
    def test_basic_projection(self):
        """Test basic projection with index list"""
        # Create zonotope
        Z = Zonotope(np.array([-4, 1, 5]), np.array([[-3, -2, -1], [2, 3, 4], [5, 5, 5]]))
        
        # Project zonotope
        Z1 = Z.project([1, 3])  # Project to dimensions 1 and 3 (0-indexed: 0 and 2)
        c1 = Z1.c
        G1 = Z1.G
        
        # True result
        true_c = np.array([-4, 5])
        true_G = np.array([[-3, -2, -1], [5, 5, 5]])
        
        # Check result
        np.testing.assert_array_almost_equal(c1.flatten(), true_c)
        np.testing.assert_array_almost_equal(G1, true_G)
    
    def test_logical_indexing(self):
        """Test projection with logical indexing"""
        # Create zonotope
        Z = Zonotope(np.array([-4, 1, 5]), np.array([[-3, -2, -1], [2, 3, 4], [5, 5, 5]]))
        
        # Logical indexing
        Z2 = Z.project([True, False, True])
        c2 = Z2.c
        G2 = Z2.G
        
        # True result (same as index [0, 2])
        true_c = np.array([-4, 5])
        true_G = np.array([[-3, -2, -1], [5, 5, 5]])
        
        # Check result
        np.testing.assert_array_almost_equal(c2.flatten(), true_c)
        np.testing.assert_array_almost_equal(G2, true_G)
    
    def test_single_dimension_projection(self):
        """Test projection to single dimension"""
        Z = Zonotope(np.array([1, 2, 3]), np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]]))
        
        # Project to first dimension only
        Z_proj = Z.project([1])  # 0-indexed
        
        assert Z_proj.dim() == 1
        np.testing.assert_array_equal(Z_proj.c.flatten(), np.array([1]))
        np.testing.assert_array_equal(Z_proj.G, np.array([[1, 0, 1]]))
    
    def test_2d_to_1d_projection(self):
        """Test projection from 2D to 1D"""
        Z = Zonotope(np.array([5, -2]), np.array([[1, 2], [3, 4]]))
        
        # Project to x-dimension
        Z_x = Z.project([1])  # 0-indexed
        assert Z_x.dim() == 1
        np.testing.assert_array_equal(Z_x.c.flatten(), np.array([5]))
        
        # Project to y-dimension  
        Z_y = Z.project([2])  # 1-indexed for 2nd dimension
        assert Z_y.dim() == 1
        np.testing.assert_array_equal(Z_y.c.flatten(), np.array([-2]))
    
    def test_identity_projection(self):
        """Test projection to all dimensions (identity)"""
        Z = Zonotope(np.array([1, 2, 3]), np.array([[1, 0], [0, 1], [1, 1]]))
        
        # Project to all dimensions
        Z_proj = Z.project([1, 2, 3])  # All dimensions
        
        # Should be equal to original
        assert Z.isequal(Z_proj)
    
    def test_empty_zonotope_projection(self):
        """Test projection of empty zonotope"""
        Z_empty = Zonotope.empty(3)
        Z_proj = Z_empty.project([1, 2])
        
        assert Z_proj.isemptyobject()
        assert Z_proj.dim() == 2
    
    def test_reordering_projection(self):
        """Test projection with reordered dimensions"""
        Z = Zonotope(np.array([1, 2, 3]), np.array([[1, 0], [0, 1], [1, 1]]))
        
        # Project with different order
        Z_proj = Z.project([3, 1])  # Dimensions 3, 1 (2-indexed: 2, 0)
        
        assert Z_proj.dim() == 2
        # Should have dimensions in the order specified
        expected_c = np.array([3, 1])  # [z, x]
        np.testing.assert_array_equal(Z_proj.c.flatten(), expected_c)
    
    def test_origin_projection(self):
        """Test projection of origin zonotope"""
        Z_origin = Zonotope.origin(4)
        Z_proj = Z_origin.project([1, 3])
        
        # Should still be origin in lower dimension
        assert Z_proj.dim() == 2
        np.testing.assert_array_equal(Z_proj.c.flatten(), np.zeros(2))
    
    def test_invalid_dimensions(self):
        """Test projection with invalid dimensions"""
        Z = Zonotope(np.array([1, 2]), np.array([[1, 0], [0, 1]]))
        
        # Test out of bounds dimension
        with pytest.raises(Exception):
            Z.project([1, 5])  # Dimension 5 doesn't exist
        
        # Test empty dimension list
        with pytest.raises(Exception):
            Z.project([])


if __name__ == "__main__":
    test_instance = TestZonotopeProject()
    
    # Run all tests
    test_instance.test_basic_projection()
    test_instance.test_logical_indexing()
    test_instance.test_single_dimension_projection()
    test_instance.test_2d_to_1d_projection()
    test_instance.test_identity_projection()
    test_instance.test_empty_zonotope_projection()
    test_instance.test_reordering_projection()
    test_instance.test_origin_projection()
    test_instance.test_invalid_dimensions()
    
    print("All zonotope project tests passed!") 