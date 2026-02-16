"""
testLong_zonotope_simplex - unit test function of simplex

Syntax:
    res = testLong_zonotope_simplex

Inputs:
    -

Outputs:
    res - true/false

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: -

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       23-April-2023 (MATLAB)
               2025 (Tiange Yang, Florian Nüssel, Python translation by AI Assistant)
"""

import numpy as np
import pytest
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.zonotope.simplex import simplex
from cora_python.contSet.zonotope.generateRandom import generateRandom
from cora_python.contSet.polytope.contains_ import contains_


def testLong_zonotope_simplex():
    """
    Test the simplex method for zonotopes with random test cases.
    
    The test verifies that the simplex tightly encloses the zonotope
    by checking containment with exact precision.
    """
    # Number of tests
    nr_tests = 5
    
    for i in range(nr_tests):
        # Random dimension
        n = np.random.randint(1, 11)  # 1 to 10
        # Number of generators
        nr_gens = np.random.randint(1, 5 * n + 1)  # 1 to 5*n
        
        # Initialize random zonotope
        Z = generateRandom(dimension=n, nr_generators=nr_gens)
        
        # Convert to simplex
        P = simplex(Z)
        
        # Check if zonotope is contained in simplex
        # Use exact method with high tolerance
        result, cert, _ = contains_(P, Z, method='exact', tol=1e-12)
        
        # Assert containment
        # Handle case where result might be an array
        if isinstance(result, np.ndarray):
            assert result.all(), f"Test {i+1}: Zonotope not contained in simplex for dimension {n}, generators {nr_gens}"
        else:
            assert result, f"Test {i+1}: Zonotope not contained in simplex for dimension {n}, generators {nr_gens}"



if __name__ == "__main__":
    # Run the long test
    testLong_zonotope_simplex()
    print("All tests passed!") 