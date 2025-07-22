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
    """Test class for zonotope enclose method - basic tests"""
    
    def test_basic_enclose(self):
        """Test basic enclosure of two zonotopes - exact MATLAB test case"""
        # Create zonotopes (exact MATLAB test case)
        # MATLAB: Z1 = zonotope([1,2,3,4; 5 6 7 8])
        # This means: center = [1; 5], generators = [[2,3,4]; [6,7,8]]
        Z1 = Zonotope(np.array([1, 5]), np.array([[2, 3, 4], [6, 7, 8]]))
        
        # MATLAB: Z2 = zonotope([9, 10, 11; 12, 13, 14])
        # This means: center = [9; 12], generators = [[10, 11]; [13, 14]]
        Z2 = Zonotope(np.array([9, 12]), np.array([[10, 11], [13, 14]]))
        
        # Obtain enclosing zonotope
        Z_ = Z1.enclose(Z2)
        
        # Obtain zonotope matrix
        c_ = Z_.c
        G_ = Z_.G
        
        # True result from MATLAB test
        true_c = np.array([5, 8.5])
        true_G = np.array([[6, 7, -4, -4, -4, 4],
                           [9.5, 10.5, -3.5, -3.5, -3.5, 8]])
        
        # Check result
        np.testing.assert_array_almost_equal(c_.flatten(), true_c)
        np.testing.assert_array_almost_equal(G_, true_G)
    
    def test_enclose_commutative(self):
        """Test that enclose is commutative - exact MATLAB test case"""
        # Use exact MATLAB test case
        Z1 = Zonotope(np.array([1, 5]), np.array([[2, 3, 4], [6, 7, 8]]))
        Z2 = Zonotope(np.array([9, 12]), np.array([[10, 11], [13, 14]]))
        
        Z12 = Z1.enclose(Z2)
        Z21 = Z2.enclose(Z1)
        
        # Check that both results are equal (commutative property)
        assert Z12.isequal(Z21)


if __name__ == "__main__":
    test_instance = TestZonotopeEnclose()
    
    # Run basic tests only
    test_instance.test_basic_enclose()
    test_instance.test_enclose_commutative()
    
    print("All basic zonotope enclose tests passed!") 