"""
Test cases for zonotope generatorLength method
"""

import pytest
import numpy as np
from cora_python.contSet.zonotope import Zonotope


class TestZonotopeGeneratorLength:
    """Test class for zonotope generatorLength method"""
    
    def test_generatorLength_2D(self):
        """Test generator lengths for 2D zonotope"""
        # Create a 2D zonotope
        c = np.array([[1], [2]])
        G = np.array([[3, 0, 1], [4, 1, 0]])  # lengths: 5, 1, 1
        Z = Zonotope(c, G)
        
        # Get generator lengths
        lengths = Z.generatorLength()
        
        # Expected lengths
        expected = np.array([5.0, 1.0, 1.0])
        
        # Check lengths
        np.testing.assert_array_almost_equal(lengths, expected)
        
    def test_generatorLength_1D(self):
        """Test generator lengths for 1D zonotope"""
        # Create a 1D zonotope
        c = np.array([[3]])
        G = np.array([[2, -1, 0.5]])  # lengths: 2, 1, 0.5
        Z = Zonotope(c, G)
        
        # Get generator lengths
        lengths = Z.generatorLength()
        
        # Expected lengths
        expected = np.array([2.0, 1.0, 0.5])
        
        # Check lengths
        np.testing.assert_array_almost_equal(lengths, expected)
        
    def test_generatorLength_3D(self):
        """Test generator lengths for 3D zonotope"""
        # Create a 3D zonotope
        c = np.array([[0], [1], [-1]])
        G = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # unit vectors
        Z = Zonotope(c, G)
        
        # Get generator lengths
        lengths = Z.generatorLength()
        
        # Expected lengths (all unit length)
        expected = np.array([1.0, 1.0, 1.0])
        
        # Check lengths
        np.testing.assert_array_almost_equal(lengths, expected)
        
    def test_generatorLength_no_generators(self):
        """Test generator lengths for point zonotope"""
        # Create a point zonotope (no generators)
        c = np.array([[2], [3]])
        G = np.zeros((2, 0))
        Z = Zonotope(c, G)
        
        # Get generator lengths
        lengths = Z.generatorLength()
        
        # Should return empty array
        assert lengths.shape == (0,)
        
    def test_generatorLength_single_generator(self):
        """Test generator lengths with single generator"""
        # Create zonotope with single generator
        c = np.array([[1], [2]])
        G = np.array([[3], [4]])  # length: 5
        Z = Zonotope(c, G)
        
        # Get generator lengths
        lengths = Z.generatorLength()
        
        # Expected length
        expected = np.array([5.0])
        
        # Check lengths
        np.testing.assert_array_almost_equal(lengths, expected)
        
    def test_generatorLength_zero_generators(self):
        """Test generator lengths with zero generators"""
        # Create zonotope with zero generators
        c = np.array([[1], [2]])
        G = np.array([[0, 0], [0, 0]])  # lengths: 0, 0
        Z = Zonotope(c, G)
        
        # Get generator lengths
        lengths = Z.generatorLength()
        
        # Expected lengths
        expected = np.array([0.0, 0.0])
        
        # Check lengths
        np.testing.assert_array_almost_equal(lengths, expected)
        
    def test_generatorLength_mixed_lengths(self):
        """Test generator lengths with mixed length generators"""
        # Create zonotope with generators of different lengths
        c = np.array([[0], [0]])
        G = np.array([[1, 2, 0, -3], 
                      [0, 0, 4, 4]])  # lengths: 1, 2, 4, 5
        Z = Zonotope(c, G)
        
        # Get generator lengths
        lengths = Z.generatorLength()
        
        # Expected lengths
        expected = np.array([1.0, 2.0, 4.0, 5.0])
        
        # Check lengths
        np.testing.assert_array_almost_equal(lengths, expected)
        
    def test_generatorLength_negative_values(self):
        """Test generator lengths with negative generator values"""
        # Create zonotope with negative generator values
        c = np.array([[0], [0]])
        G = np.array([[-3, 1], 
                      [4, -1]])  # lengths: 5, sqrt(2)
        Z = Zonotope(c, G)
        
        # Get generator lengths
        lengths = Z.generatorLength()
        
        # Expected lengths
        expected = np.array([5.0, np.sqrt(2)])
        
        # Check lengths
        np.testing.assert_array_almost_equal(lengths, expected)
        
    def test_generatorLength_high_dimension(self):
        """Test generator lengths in higher dimensions"""
        # Create a 4D zonotope
        c = np.zeros((4, 1))
        G = np.array([[1, 0, 1], 
                      [1, 1, 0], 
                      [1, 1, 1], 
                      [1, 0, 0]])  # lengths: sqrt(1^2+1^2+1^2+1^2)=2, sqrt(0^2+1^2+1^2+0^2)=sqrt(2), sqrt(1^2+0^2+1^2+0^2)=sqrt(2)
        Z = Zonotope(c, G)
        
        # Get generator lengths
        lengths = Z.generatorLength()
        
        # Expected lengths
        expected = np.array([2.0, np.sqrt(2), np.sqrt(2)])
        
        # Check lengths
        np.testing.assert_array_almost_equal(lengths, expected)


if __name__ == "__main__":
    pytest.main([__file__]) 