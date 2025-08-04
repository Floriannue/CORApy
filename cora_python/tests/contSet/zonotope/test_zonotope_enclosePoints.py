"""
test_zonotope_enclosePoints - unit test function for Zonotope.enclosePoints

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
"""

import numpy as np
import pytest
from cora_python.contSet.zonotope.zonotope import Zonotope


def test_zonotope_enclosePoints():
    """Test enclosePoints method for Zonotope."""
    
    # Test points from MATLAB test
    p = np.array([
        [1, 3, -2, 4, 3, -1, 1, 0],
        [2, -1, 1, -3, 2, 1, 0, 1]
    ])
    
    # Compute enclosing zonotope with default method (maiga)
    Z = Zonotope.enclosePoints(p)
    
    # Compute with different method
    Z_stursberg = Zonotope.enclosePoints(p, 'stursberg')
    
    # Check if all points are contained
    # Use contains method which returns just the boolean result
    assert Z.contains(p), "Default method should contain all points"
    assert Z_stursberg.contains(p), "Stursberg method should contain all points"
    
    # Test that both zonotopes are valid (have correct dimensions)
    assert Z.dim() == 2, "Zonotope should have 2 dimensions"
    assert Z_stursberg.dim() == 2, "Zonotope should have 2 dimensions"
    
    # Test that zonotopes have reasonable properties
    assert Z.c.shape[0] == 2, "Center should be 2D"
    assert Z.G.shape[0] == 2, "Generators should have 2 rows"
    assert Z_stursberg.c.shape[0] == 2, "Center should be 2D"
    assert Z_stursberg.G.shape[0] == 2, "Generators should have 2 rows"



if __name__ == "__main__":
    test_zonotope_enclosePoints()
    print("All tests passed!") 