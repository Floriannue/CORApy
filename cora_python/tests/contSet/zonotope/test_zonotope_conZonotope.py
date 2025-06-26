"""
Test cases for zonotope conZonotope method
"""

import pytest
import numpy as np
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.conZonotope import ConZonotope


class TestZonotopeConZonotope:
    """Test class for zonotope conZonotope method"""
    
    def test_conZonotope_2D(self):
        """Test conversion of 2D zonotope to conZonotope"""
        # Create a 2D zonotope
        c = np.array([[1], [2]])
        G = np.array([[1, 0, 1], [0, 1, -1]])
        Z = Zonotope(c, G)
        
        # Convert to conZonotope
        cZ = Z.conZonotope()
        
        # Check type
        assert isinstance(cZ, ConZonotope)
        
        # Check center and generators are preserved
        np.testing.assert_array_almost_equal(cZ.c, c)
        np.testing.assert_array_almost_equal(cZ.G, G)
        
    def test_conZonotope_1D(self):
        """Test conversion of 1D zonotope to conZonotope"""
        # Create a 1D zonotope
        c = np.array([[3]])
        G = np.array([[2, -1]])
        Z = Zonotope(c, G)
        
        # Convert to conZonotope
        cZ = Z.conZonotope()
        
        # Check type
        assert isinstance(cZ, ConZonotope)
        
        # Check center and generators are preserved
        np.testing.assert_array_almost_equal(cZ.c, c)
        np.testing.assert_array_almost_equal(cZ.G, G)
        
    def test_conZonotope_3D(self):
        """Test conversion of 3D zonotope to conZonotope"""
        # Create a 3D zonotope
        c = np.array([[0], [1], [-1]])
        G = np.array([[1, 0, 0, 1], [0, 1, 0, -1], [0, 0, 1, 1]])
        Z = Zonotope(c, G)
        
        # Convert to conZonotope
        cZ = Z.conZonotope()
        
        # Check type
        assert isinstance(cZ, ConZonotope)
        
        # Check center and generators are preserved
        np.testing.assert_array_almost_equal(cZ.c, c)
        np.testing.assert_array_almost_equal(cZ.G, G)
        
    def test_conZonotope_point(self):
        """Test conversion of point zonotope to conZonotope"""
        # Create a point zonotope (no generators)
        c = np.array([[2], [3]])
        G = np.zeros((2, 0))
        Z = Zonotope(c, G)
        
        # Convert to conZonotope
        cZ = Z.conZonotope()
        
        # Check type
        assert isinstance(cZ, ConZonotope)
        
        # Check center and generators are preserved
        np.testing.assert_array_almost_equal(cZ.c, c)
        assert cZ.G.shape == (2, 0)
        
    def test_conZonotope_single_generator(self):
        """Test conversion with single generator"""
        # Create zonotope with single generator
        c = np.array([[1], [2]])
        G = np.array([[3], [4]])
        Z = Zonotope(c, G)
        
        # Convert to conZonotope
        cZ = Z.conZonotope()
        
        # Check type
        assert isinstance(cZ, ConZonotope)
        
        # Check center and generators are preserved
        np.testing.assert_array_almost_equal(cZ.c, c)
        np.testing.assert_array_almost_equal(cZ.G, G)
        
    def test_conZonotope_zero_generators(self):
        """Test conversion with zero generators"""
        # Create zonotope with zero generators
        c = np.array([[1], [2]])
        G = np.array([[0, 0], [0, 0]])
        Z = Zonotope(c, G)
        
        # Convert to conZonotope
        cZ = Z.conZonotope()
        
        # Check type
        assert isinstance(cZ, ConZonotope)
        
        # Check center and generators are preserved
        np.testing.assert_array_almost_equal(cZ.c, c)
        np.testing.assert_array_almost_equal(cZ.G, G)


if __name__ == "__main__":
    pytest.main([__file__]) 