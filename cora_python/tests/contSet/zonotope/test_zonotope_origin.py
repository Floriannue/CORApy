"""
test_zonotope_origin - unit test function of origin

Syntax:
    python -m pytest test_zonotope_origin.py

Inputs:
    -

Outputs:
    test results

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 21-September-2024 (MATLAB)
Python translation: 2025
"""

import pytest
import numpy as np

from cora_python.contSet.zonotope import Zonotope


class TestZonotopeOrigin:
    """Test class for zonotope origin method"""
    
    def test_1d_origin(self):
        """Test 1D origin zonotope"""
        Z = Zonotope.origin(1)
        Z_true = Zonotope(np.array([0]))
        
        assert Z.isequal(Z_true)
        assert Z.contains_(np.array([0]))
    
    def test_2d_origin(self):
        """Test 2D origin zonotope"""
        Z = Zonotope.origin(2)
        Z_true = Zonotope(np.zeros(2))
        
        assert Z.isequal(Z_true)
        assert Z.contains_(np.zeros(2))
    
    def test_3d_origin(self):
        """Test 3D origin zonotope"""
        Z = Zonotope.origin(3)
        Z_true = Zonotope(np.zeros(3))
        
        assert Z.isequal(Z_true)
        assert Z.contains_(np.zeros(3))
    
    def test_origin_properties(self):
        """Test properties of origin zonotope"""
        for n in [1, 2, 5, 10]:
            Z = Zonotope.origin(n)
            
            # Should have correct dimension
            assert Z.dim() == n
            
            # Center should be at origin
            np.testing.assert_array_equal(Z.c.flatten(), np.zeros(n))
            
            # Should have no generators (or zero generators)
            assert Z.G.shape[1] == 0 or np.allclose(Z.G, 0)
    
    def test_wrong_calls(self):
        """Test wrong function calls"""
        # Zero dimension
        with pytest.raises(Exception):
            Zonotope.origin(0)
        
        # Negative dimension
        with pytest.raises(Exception):
            Zonotope.origin(-1)
        
        # Fractional dimension
        with pytest.raises(Exception):
            Zonotope.origin(0.5)
        
        # Array input
        with pytest.raises(Exception):
            Zonotope.origin([1, 2])
        
        # String input
        with pytest.raises(Exception):
            Zonotope.origin('text')


if __name__ == "__main__":
    test_instance = TestZonotopeOrigin()
    
    # Run all tests
    test_instance.test_1d_origin()
    test_instance.test_2d_origin()
    test_instance.test_3d_origin()
    test_instance.test_origin_properties()
    test_instance.test_wrong_calls()
    
    print("All zonotope origin tests passed!") 