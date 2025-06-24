"""
test_zonotope_display - unit test function of display

Syntax:
    python -m pytest test_zonotope_display.py

Inputs:
    -

Outputs:
    test results

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 28-April-2023 (MATLAB)
Python translation: 2025
"""

import pytest
import numpy as np

from cora_python.contSet.zonotope import Zonotope


class TestZonotopeDisplay:
    """Test class for zonotope display method"""
    
    def test_empty_zonotope_display(self):
        """Test display of empty zonotope"""
        n = 2
        Z = Zonotope.empty(n)
        display_str = Z.display()
        
        # Should not raise error and return string
        assert isinstance(display_str, str)
        assert len(display_str) > 0
    
    def test_2d_zonotope_display(self):
        """Test display of 2D zonotope"""
        c = np.array([-2, 1])
        G = np.array([[2, 4, 5, 3, 3], 
                      [0, 3, 5, 2, 3]])
        Z = Zonotope(c, G)
        display_str = Z.display()
        
        # Should not raise error and return string
        assert isinstance(display_str, str)
        assert len(display_str) > 0
    
    def test_no_generator_matrix_display(self):
        """Test display of zonotope with no generators"""
        c = np.array([-2, 1])
        Z = Zonotope(c)
        display_str = Z.display()
        
        # Should not raise error and return string
        assert isinstance(display_str, str)
        assert len(display_str) > 0
    
    def test_many_generators_display(self):
        """Test display of zonotope with many generators"""
        c = np.array([-2, 1])
        G = np.ones((2, 25))
        Z = Zonotope(c, G)
        display_str = Z.display()
        
        # Should not raise error and return string
        assert isinstance(display_str, str)
        assert len(display_str) > 0
    
    def test_1d_zonotope_display(self):
        """Test display of 1D zonotope"""
        Z = Zonotope(np.array([5]), np.array([[2, 1]]))
        display_str = Z.display()
        
        assert isinstance(display_str, str)
        assert len(display_str) > 0
    
    def test_high_dimensional_display(self):
        """Test display of high-dimensional zonotope"""
        c = np.zeros(10)
        G = np.eye(10)
        Z = Zonotope(c, G)
        display_str = Z.display()
        
        assert isinstance(display_str, str)
        assert len(display_str) > 0
    
    def test_str_method(self):
        """Test that __str__ method uses display"""
        Z = Zonotope(np.array([1, 2]), np.array([[1, 0], [0, 1]]))
        str_result = str(Z)
        display_result = Z.display()
        
        # Should be the same
        assert str_result == display_result


if __name__ == "__main__":
    test_instance = TestZonotopeDisplay()
    
    # Run all tests
    test_instance.test_empty_zonotope_display()
    test_instance.test_2d_zonotope_display()
    test_instance.test_no_generator_matrix_display()
    test_instance.test_many_generators_display()
    test_instance.test_1d_zonotope_display()
    test_instance.test_high_dimensional_display()
    test_instance.test_str_method()
    
    print("All zonotope display tests passed!") 