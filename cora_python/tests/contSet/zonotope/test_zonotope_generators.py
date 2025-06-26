"""
Test cases for zonotope generators method
"""

import pytest
import numpy as np
from cora_python.contSet.zonotope import Zonotope


class TestZonotopeGenerators:
    """Test class for zonotope generators method"""
    
    def test_generators_2D(self):
        """Test getting generators from 2D zonotope"""
        # Create a 2D zonotope
        c = np.array([[1], [2]])
        G = np.array([[1, 0, 1], [0, 1, -1]])
        Z = Zonotope(c, G)
        
        # Get generators
        G_result = Z.generators()
        
        # Check that generators are returned correctly
        np.testing.assert_array_almost_equal(G_result, G)
        
    def test_generators_1D(self):
        """Test getting generators from 1D zonotope"""
        # Create a 1D zonotope
        c = np.array([[3]])
        G = np.array([[2, -1, 0.5]])
        Z = Zonotope(c, G)
        
        # Get generators
        G_result = Z.generators()
        
        # Check that generators are returned correctly
        np.testing.assert_array_almost_equal(G_result, G)
        
    def test_generators_3D(self):
        """Test getting generators from 3D zonotope"""
        # Create a 3D zonotope
        c = np.array([[0], [1], [-1]])
        G = np.array([[1, 0, 0, 1], [0, 1, 0, -1], [0, 0, 1, 1]])
        Z = Zonotope(c, G)
        
        # Get generators
        G_result = Z.generators()
        
        # Check that generators are returned correctly
        np.testing.assert_array_almost_equal(G_result, G)
        
    def test_generators_no_generators(self):
        """Test getting generators from point zonotope"""
        # Create a point zonotope (no generators)
        c = np.array([[2], [3]])
        G = np.zeros((2, 0))
        Z = Zonotope(c, G)
        
        # Get generators
        G_result = Z.generators()
        
        # Check that empty generator matrix is returned
        assert G_result.shape == (2, 0)
        
    def test_generators_single_generator(self):
        """Test getting generators with single generator"""
        # Create zonotope with single generator
        c = np.array([[1], [2]])
        G = np.array([[3], [4]])
        Z = Zonotope(c, G)
        
        # Get generators
        G_result = Z.generators()
        
        # Check that generators are returned correctly
        np.testing.assert_array_almost_equal(G_result, G)
        
    def test_generators_zero_generators(self):
        """Test getting generators with zero generators"""
        # Create zonotope with zero generators
        c = np.array([[1], [2]])
        G = np.array([[0, 0], [0, 0]])
        Z = Zonotope(c, G)
        
        # Get generators
        G_result = Z.generators()
        
        # Check that generators are returned correctly
        np.testing.assert_array_almost_equal(G_result, G)
        
    def test_generators_reference(self):
        """Test that generators returns a reference to the actual generator matrix"""
        # Create a zonotope
        c = np.array([[1], [2]])
        G = np.array([[1, 2], [3, 4]])
        Z = Zonotope(c, G)
        
        # Get generators
        G_result = Z.generators()
        
        # Check that it's the same object (reference)
        assert G_result is Z.G
        
    def test_generators_different_shapes(self):
        """Test generators with different matrix shapes"""
        # Test various shapes
        shapes = [(1, 1), (2, 1), (1, 3), (3, 2), (4, 5)]
        
        for n, m in shapes:
            c = np.zeros((n, 1))
            G = np.random.rand(n, m)
            Z = Zonotope(c, G)
            
            G_result = Z.generators()
            
            # Check shape and values
            assert G_result.shape == (n, m)
            np.testing.assert_array_almost_equal(G_result, G)


if __name__ == "__main__":
    pytest.main([__file__]) 