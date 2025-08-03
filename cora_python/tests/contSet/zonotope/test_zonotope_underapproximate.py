"""
test_zonotope_underapproximate - unit test function of underapproximate

This test file mirrors the MATLAB test cases for the underapproximate method.
"""

import numpy as np
import pytest
from cora_python.contSet.zonotope import Zonotope
from cora_python.g.functions.matlab.validate.check import compareMatrices


def test_zonotope_underapproximate():
    """
    Test the underapproximate method for zonotope class.
    This test mirrors the MATLAB test cases exactly.
    """
    # Create zonotope (same as MATLAB test)
    Z1 = Zonotope(np.array([[-4], [1]]), np.array([[-3, -2, -1], [2, 3, 4]]))
    
    # Create direction matrix (same as MATLAB test)
    S = np.array([[1, 1], [0, 1]])
    
    # Test case 1: underapproximate without S parameter
    V_1 = Z1.underapproximate()
    
    # Test case 2: underapproximate with S parameter
    V_2 = Z1.underapproximate(S)
    
    # Expected results from MATLAB test
    true_V_1 = np.array([[-10, 2, -10, 2], [10, -8, 10, -8]])
    true_V_2 = np.array([[2, -10, -4, -4], [-8, 10, 6, -4]])
    
    # Check results using compareMatrices (same as MATLAB test)
    assert compareMatrices(V_1, true_V_1), f"V_1 does not match expected result. Got:\n{V_1}\nExpected:\n{true_V_1}"
    assert compareMatrices(V_2, true_V_2), f"V_2 does not match expected result. Got:\n{V_2}\nExpected:\n{true_V_2}"


if __name__ == "__main__":
    # Run the main test
    test_zonotope_underapproximate()
    print("All tests passed!") 