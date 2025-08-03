"""
test_zonotope_zonoBundle - unit test function of conversion to zonoBundle

This module tests the zonoBundle method for zonotope objects.

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       23-April-2023 (MATLAB)
Python translation: 2025
"""

import pytest
import numpy as np
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.zonoBundle import ZonoBundle
from cora_python.contSet.emptySet import EmptySet


def test_zonotope_zonoBundle():
    """Unit test function of zonoBundle - mirrors MATLAB test_zonotope_zonoBundle.m"""
    
    # Test 1: empty set
    Z = Zonotope.empty(2)
    zB = Z.zonoBundle()
    
    # Check that the result is an empty set and has correct dimension
    assert isinstance(zB, ZonoBundle)
    assert zB.parallelSets == 1
    assert zB.dim() == 2
    
    # Test 2: instantiate zonotope
    c = np.array([[1], [1], [-1]])
    G = np.array([[2, -1, 3, 4], 
                  [0, 2, 3, -1], 
                  [-1, 0, 0, 2]])
    Z = Zonotope(c, G)
    
    # convert to zonoBundle
    zB = Z.zonoBundle()
    
    # check converted set
    assert zB.parallelSets == 1
    assert len(zB.Z) == 1
    assert isinstance(zB.Z[0], Zonotope)
    
    # Check that the zonotope in the bundle is equal to the original
    assert np.allclose(zB.Z[0].c, Z.c)
    assert np.allclose(zB.Z[0].G, Z.G)



if __name__ == "__main__":
    # Run all tests
    test_zonotope_zonoBundle()
    print("All tests passed!") 