"""
testLong_zonotope_norm - unit test function of norm

Syntax:
    res = testLong_zonotope_norm

Inputs:
    -

Outputs:
    res - true/false

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: -

Authors:       Victor Gassmann
Written:       31-July-2020
Last update:   ---
Last revision: ---
"""

import pytest
import numpy as np
from cora_python.contSet.zonotope import Zonotope
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol


def testLong_zonotope_norm():
    """Unit test function of norm - mirrors MATLAB testLong_zonotope_norm.m"""
    
    # assume true
    res = True
    
    TOL = 1e-6
    
    # loop over dimensions
    for i in range(2, 5):  # 2 to 4
        
        # loop over number of generators
        for j in range(i, 11, 2):  # i to 10, step 2
            
            # instantiate random zonotope
            c = np.zeros((i, 1))
            G = 10 * np.random.randn(i, j)
            Z = Zonotope(c, G)
            
            # 2 norm test
            val2_exact = Z.norm_(2, 'exact')
            val2_ub = Z.norm_(2, 'ub')
            val2_ubc = Z.norm_(2, 'ub_convex')
            
            # compute vertices
            V = Z.vertices_()
            
            # check exact vs. upper bound
            if val2_exact > val2_ub:
                assert withinTol(val2_exact, val2_ub, TOL), f"Dimension {i}, generators {j}: exact > ub"
            
            # check exact vs. upper bound (convex)
            if val2_exact > val2_ubc:
                # Use a more lenient tolerance for upper bound comparison
                # since optimization solvers might not be available or precise
                assert withinTol(val2_exact, val2_ubc, TOL), f"Dimension {i}, generators {j}: exact > ub_convex"
            
            # check exact vs. norm of all vertices
            if V.size > 0:
                vertex_norms = np.sqrt(np.sum(V**2, axis=0))
                max_vertex_norm = np.max(vertex_norms)
                if val2_exact > 0:
                    val = abs(val2_exact - max_vertex_norm) / val2_exact
                    if val > 0:
                                                    # Use a more lenient tolerance for vertex comparison
                            # since exact norm might find non-vertex points
                            assert withinTol(val, 0, TOL), f"Dimension {i}, generators {j}: exact vs vertex norm"
    
    return res


if __name__ == "__main__":
    pytest.main([__file__]) 