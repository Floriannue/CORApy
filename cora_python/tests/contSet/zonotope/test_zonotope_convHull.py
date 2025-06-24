"""
test_zonotope_convHull - unit test function of convHull

Syntax:
    python -m pytest test_zonotope_convHull.py

Inputs:
    -

Outputs:
    test results

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 23-April-2023 (MATLAB)
Python translation: 2025
"""

import pytest
import numpy as np

from cora_python.contSet.zonotope import Zonotope


class TestZonotopeConvHull:
    """Test class for zonotope convHull method"""
    
    def test_basic_convex_hull(self):
        """Test basic convex hull of two zonotopes"""
        # Create two simple zonotopes
        Z1 = Zonotope(np.array([0, 0]), np.array([[1, 0], [0, 1]]))
        Z2 = Zonotope(np.array([2, 2]), np.array([[0.5, 0], [0, 0.5]]))
        
        # Compute convex hull
        Z = Z1.convHull_(Z2)
        
        # Both original zonotopes should be contained in the hull
        # Sample points from both
        points1 = Z1.randPoint_(10)
        points2 = Z2.randPoint_(10)
        
        # All points should be contained in the convex hull
        for i in range(points1.shape[1]):
            assert Z.contains_(points1[:, i])
        
        for i in range(points2.shape[1]):
            assert Z.contains_(points2[:, i])
    
    def test_convex_hull_containment(self):
        """Test that convex hull contains linear combinations"""
        Z1 = Zonotope(np.array([1, 1]), np.array([[1, 0], [0, 1]]))
        Z2 = Zonotope(np.array([-1, -1]), np.array([[0.5, 0], [0, 0.5]]))
        
        Z = Z1.convHull_(Z2)
        
        # Sample points from both zonotopes
        nrRandPoints = 20
        p1 = Z1.randPoint_(nrRandPoints)
        p2 = Z2.randPoint_(nrRandPoints)
        
        # Compute linear combinations
        lambda_vals = np.random.rand(nrRandPoints)
        p_combined = lambda_vals * p1 + (1 - lambda_vals) * p2
        
        # All points should be contained in the convex hull
        for i in range(nrRandPoints):
            assert Z.contains_(p1[:, i])
            assert Z.contains_(p2[:, i])
            assert Z.contains_(p_combined[:, i])
    
    def test_convex_hull_self(self):
        """Test convex hull of zonotope with itself"""
        Z = Zonotope(np.array([1, 2]), np.array([[1, 0, 1], [0, 1, 0]]))
        Z_hull = Z.convHull_(Z)
        
        # Should be equal to original zonotope
        assert Z.isequal(Z_hull)
    
    def test_convex_hull_empty(self):
        """Test convex hull with empty zonotope"""
        Z = Zonotope(np.array([1, 2]), np.array([[1, 0], [0, 1]]))
        Z_empty = Zonotope.empty(2)
        
        Z_hull = Z.convHull_(Z_empty)
        
        # Result should be equal to the non-empty zonotope
        assert Z.isequal(Z_hull)
    
    def test_convex_hull_1d(self):
        """Test convex hull of 1D zonotopes"""
        Z1 = Zonotope(np.array([0]), np.array([[1, 2]]))
        Z2 = Zonotope(np.array([5]), np.array([[1]]))
        
        Z_hull = Z1.convHull_(Z2)
        
        assert Z_hull.dim() == 1
        assert isinstance(Z_hull, Zonotope)
    
    def test_convex_hull_different_generators(self):
        """Test convex hull of zonotopes with different number of generators"""
        Z1 = Zonotope(np.array([0, 0]), np.array([[1], [1]]))  # 1 generator
        Z2 = Zonotope(np.array([2, 2]), np.array([[1, 0, 0.5], [0, 1, 0.5]]))  # 3 generators
        
        Z_hull = Z1.convHull_(Z2)
        
        # Should not raise error
        assert isinstance(Z_hull, Zonotope)
        assert Z_hull.dim() == 2
    
    def test_convex_hull_origin(self):
        """Test convex hull with origin zonotope"""
        Z = Zonotope(np.array([3, 4]), np.array([[1, 0], [0, 1]]))
        Z_origin = Zonotope.origin(2)
        
        Z_hull = Z.convHull_(Z_origin)
        
        # Should contain both the original zonotope and the origin
        assert Z_hull.contains_(np.array([0, 0]))
        assert Z_hull.contains_(Z.c.flatten())
    
    def test_convex_hull_symmetric(self):
        """Test that convex hull is symmetric"""
        Z1 = Zonotope(np.array([1, 1]), np.array([[1, 0], [0, 1]]))
        Z2 = Zonotope(np.array([-1, -1]), np.array([[0.5, 0], [0, 0.5]]))
        
        Z12 = Z1.convHull_(Z2)
        Z21 = Z2.convHull_(Z1)
        
        # Both should contain the same points (though representation may differ)
        # Test by checking containment of sample points
        points = Z12.randPoint_(20)
        for i in range(points.shape[1]):
            assert Z21.contains_(points[:, i])
    
    def test_convex_hull_scaling(self):
        """Test convex hull behavior under scaling"""
        Z1 = Zonotope(np.array([1, 0]), np.array([[1, 0], [0, 1]]))
        Z2 = Zonotope(np.array([0, 1]), np.array([[1, 0], [0, 1]]))
        
        # Original hull
        Z_hull = Z1.convHull_(Z2)
        
        # Scaled versions
        factor = 2
        Z1_scaled = Z1 * factor
        Z2_scaled = Z2 * factor
        Z_hull_scaled = Z1_scaled.convHull_(Z2_scaled)
        
        # Scaled hull should be the same as scaling the original hull
        Z_hull_expected = Z_hull * factor
        
        # Check by sampling points
        points = Z_hull_scaled.randPoint_(20)
        for i in range(points.shape[1]):
            assert Z_hull_expected.contains_(points[:, i])


if __name__ == "__main__":
    test_instance = TestZonotopeConvHull()
    
    # Run all tests
    test_instance.test_basic_convex_hull()
    test_instance.test_convex_hull_containment()
    test_instance.test_convex_hull_self()
    test_instance.test_convex_hull_empty()
    test_instance.test_convex_hull_1d()
    test_instance.test_convex_hull_different_generators()
    test_instance.test_convex_hull_origin()
    test_instance.test_convex_hull_symmetric()
    test_instance.test_convex_hull_scaling()
    
    print("All zonotope convHull tests passed!") 