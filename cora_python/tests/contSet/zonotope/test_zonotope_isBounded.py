"""
test_zonotope_isBounded - unit test function of isBounded

Syntax:
    python -m pytest test_zonotope_isBounded.py

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


class TestZonotopeIsBounded:
    """Test class for zonotope isBounded method"""
    
    def test_finite_zonotope_is_bounded(self):
        """Test that finite zonotope is bounded"""
        Z = Zonotope(np.array([1, 2]), np.array([[1, 0.5], [0, 1]]))
        
        assert Z.isBounded()
    
    def test_origin_is_bounded(self):
        """Test that origin zonotope is bounded"""
        Z_origin = Zonotope.origin(3)
        
        assert Z_origin.isBounded()
    
    def test_large_zonotope_is_bounded(self):
        """Test that even large zonotope is bounded"""
        # Very large but finite zonotope
        c = np.array([1e6, 1e6])
        G = np.array([[1e3, 1e4], [1e5, 1e2]])
        Z = Zonotope(c, G)
        
        assert Z.isBounded()
    
    def test_1d_zonotope_is_bounded(self):
        """Test that 1D zonotope is bounded"""
        Z = Zonotope(np.array([5]), np.array([[2, 1, 3]]))
        
        assert Z.isBounded()
    
    def test_high_dimensional_zonotope_is_bounded(self):
        """Test that high-dimensional zonotope is bounded"""
        n = 10
        c = np.random.rand(n)
        G = np.random.rand(n, 15)
        Z = Zonotope(c, G)
        
        assert Z.isBounded()
    
    def test_empty_zonotope_is_bounded(self):
        """Test that empty zonotope is considered bounded"""
        Z_empty = Zonotope.empty(2)
        
        # Empty set is conventionally considered bounded
        assert Z_empty.isBounded()
    
    def test_single_point_is_bounded(self):
        """Test that single point (no generators) is bounded"""
        Z = Zonotope(np.array([3, 4, 5]))  # No generators
        
        assert Z.isBounded()
    
    def test_box_zonotope_is_bounded(self):
        """Test that axis-aligned box is bounded"""
        # Unit box
        c = np.zeros(3)
        G = np.eye(3)
        Z = Zonotope(c, G)
        
        assert Z.isBounded()
    
    def test_scaled_zonotope_is_bounded(self):
        """Test that scaled zonotope is bounded"""
        Z = Zonotope(np.array([1, 2]), np.array([[1, 0], [0, 1]]))
        
        # Scale by large factor
        Z_scaled = Z * 1e9
        
        assert Z_scaled.isBounded()
    
    def test_translated_zonotope_is_bounded(self):
        """Test that translated zonotope is bounded"""
        Z = Zonotope(np.array([0, 0]), np.array([[1, 0], [0, 1]]))
        
        # Translate by large vector
        Z_translated = Z + np.array([1e6, -1e6])
        
        assert Z_translated.isBounded()
    
    def test_multiple_generators_is_bounded(self):
        """Test zonotope with many generators is bounded"""
        c = np.array([0, 0])
        G = np.random.rand(2, 50)  # 50 generators
        Z = Zonotope(c, G)
        
        assert Z.isBounded()
    
    def test_sum_of_bounded_is_bounded(self):
        """Test that sum of bounded zonotopes is bounded"""
        Z1 = Zonotope(np.array([1, 2]), np.array([[1, 0], [0, 1]]))
        Z2 = Zonotope(np.array([3, 4]), np.array([[2, 1], [1, 2]]))
        
        Z_sum = Z1 + Z2
        
        assert Z_sum.isBounded()
    
    def test_random_zonotopes_are_bounded(self):
        """Test that randomly generated zonotopes are bounded"""
        for _ in range(10):
            n = np.random.randint(1, 6)  # Dimension 1-5
            m = np.random.randint(1, 2*n + 5)  # Number of generators
            
            c = np.random.randn(n) * 10
            G = np.random.randn(n, m) * 5
            Z = Zonotope(c, G)
            
            assert Z.isBounded()


if __name__ == "__main__":
    test_instance = TestZonotopeIsBounded()
    
    # Run all tests
    test_instance.test_finite_zonotope_is_bounded()
    test_instance.test_origin_is_bounded()
    test_instance.test_large_zonotope_is_bounded()
    test_instance.test_1d_zonotope_is_bounded()
    test_instance.test_high_dimensional_zonotope_is_bounded()
    test_instance.test_empty_zonotope_is_bounded()
    test_instance.test_single_point_is_bounded()
    test_instance.test_box_zonotope_is_bounded()
    test_instance.test_scaled_zonotope_is_bounded()
    test_instance.test_translated_zonotope_is_bounded()
    test_instance.test_multiple_generators_is_bounded()
    test_instance.test_sum_of_bounded_is_bounded()
    test_instance.test_random_zonotopes_are_bounded()
    
    print("All zonotope isBounded tests passed!") 