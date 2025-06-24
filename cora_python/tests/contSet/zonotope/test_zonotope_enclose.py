"""
test_zonotope_enclose - unit test function of enclose

Syntax:
    python -m pytest test_zonotope_enclose.py

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


class TestZonotopeEnclose:
    """Test class for zonotope enclose method"""
    
    def test_basic_enclose(self):
        """Test basic enclosure of two zonotopes"""
        # Create zonotopes
        Z1 = Zonotope(np.array([1, 5]), np.array([[2, 3, 4], [6, 7, 8]]))
        Z2 = Zonotope(np.array([9, 12]), np.array([[10, 11], [13, 14]]))
        
        # Obtain enclosing zonotope
        Z_ = Z1.enclose(Z2)
        
        # Obtain zonotope matrix
        c_ = Z_.c
        G_ = Z_.G
        
        # True result
        true_c = np.array([5, 8.5])
        true_G = np.array([[6, 7, -4, -4, -4, 4],
                           [9.5, 10.5, -3.5, -3.5, -3.5, 8]])
        
        # Check result
        np.testing.assert_array_almost_equal(c_.flatten(), true_c)
        np.testing.assert_array_almost_equal(G_, true_G)
    
    def test_enclose_commutative(self):
        """Test that enclose is commutative"""
        Z1 = Zonotope(np.array([1, 5]), np.array([[2, 3, 4], [6, 7, 8]]))
        Z2 = Zonotope(np.array([9, 12]), np.array([[10, 11], [13, 14]]))
        
        Z12 = Z1.enclose(Z2)
        Z21 = Z2.enclose(Z1)
        
        assert Z12.isequal(Z21)
    
    def test_enclose_self(self):
        """Test enclosing zonotope with itself"""
        Z = Zonotope(np.array([1, 2]), np.array([[1, 0], [0, 1]]))
        Z_enclosed = Z.enclose(Z)
        
        # Should be equal to original zonotope
        assert Z.isequal(Z_enclosed)
    
    def test_enclose_empty(self):
        """Test enclosing with empty zonotope"""
        Z = Zonotope(np.array([1, 2]), np.array([[1, 0], [0, 1]]))
        Z_empty = Zonotope.empty(2)
        
        Z_result = Z.enclose(Z_empty)
        
        # Result should be equal to original zonotope
        assert Z.isequal(Z_result)
    
    def test_enclose_different_generators(self):
        """Test enclosing zonotopes with different number of generators"""
        Z1 = Zonotope(np.array([0, 0]), np.array([[1], [1]]))  # 1 generator
        Z2 = Zonotope(np.array([2, 2]), np.array([[1, 0, 1], [0, 1, 1]]))  # 3 generators
        
        Z_enclosed = Z1.enclose(Z2)
        
        # Should not raise error and contain both zonotopes
        assert isinstance(Z_enclosed, Zonotope)
        assert Z_enclosed.dim() == 2
    
    def test_enclose_containment(self):
        """Test that enclosed zonotope contains both original zonotopes"""
        Z1 = Zonotope(np.array([1, 1]), np.array([[1, 0], [0, 1]]))
        Z2 = Zonotope(np.array([-1, -1]), np.array([[0.5, 0], [0, 0.5]]))
        
        Z_enclosed = Z1.enclose(Z2)
        
        # Sample points from both zonotopes
        points1 = Z1.randPoint_(20)
        points2 = Z2.randPoint_(20)
        
        # All points should be contained in the enclosed zonotope
        for i in range(points1.shape[1]):
            point = points1[:, i]
            assert Z_enclosed.contains_(point)
        
        for i in range(points2.shape[1]):
            point = points2[:, i]
            assert Z_enclosed.contains_(point)
    
    def test_enclose_1d(self):
        """Test enclosing 1D zonotopes"""
        Z1 = Zonotope(np.array([0]), np.array([[1, 2]]))
        Z2 = Zonotope(np.array([5]), np.array([[1]]))
        
        Z_enclosed = Z1.enclose(Z2)
        
        assert Z_enclosed.dim() == 1
        assert isinstance(Z_enclosed, Zonotope)
    
    def test_enclose_origin(self):
        """Test enclosing with origin zonotope"""
        Z = Zonotope(np.array([3, 4]), np.array([[1, 0], [0, 1]]))
        Z_origin = Zonotope.origin(2)
        
        Z_enclosed = Z.enclose(Z_origin)
        
        # Should contain both the original zonotope and the origin
        assert Z_enclosed.contains_(np.array([0, 0]))
        assert Z_enclosed.contains_(Z.c.flatten())


if __name__ == "__main__":
    test_instance = TestZonotopeEnclose()
    
    # Run all tests
    test_instance.test_basic_enclose()
    test_instance.test_enclose_commutative()
    test_instance.test_enclose_self()
    test_instance.test_enclose_empty()
    test_instance.test_enclose_different_generators()
    test_instance.test_enclose_containment()
    test_instance.test_enclose_1d()
    test_instance.test_enclose_origin()
    
    print("All zonotope enclose tests passed!") 