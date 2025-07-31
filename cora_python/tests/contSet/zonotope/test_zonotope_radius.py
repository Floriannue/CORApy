"""
test_zonotope_radius - unit test function of radius

Syntax:
    res = test_zonotope_radius

Inputs:
    -

Outputs:
    res - true/false

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: -

Authors:       Mark Wetzlinger
Written:       27-August-2019
Last update:   ---
Last revision: ---
Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import pytest
from cora_python.contSet.zonotope import Zonotope


def test_zonotope_radius():
    """
    Unit test function of radius
    """
    # Create a zonotope
    Z = Zonotope(np.array([0, 0]), np.array([[1, 3], [2, 1]]))
    
    # Compute radius
    r = Z.radius()
    
    # Analytical solution
    r_true = 5
    
    # Check results
    assert np.isclose(r, r_true, atol=1e-10), f"Expected {r_true}, got {r}"
    
    return True


if __name__ == "__main__":
    test_zonotope_radius()
    print("MATLAB test case passed!") 