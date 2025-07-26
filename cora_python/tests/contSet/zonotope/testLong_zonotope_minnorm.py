"""
testLong_zonotope_minnorm - unit test function of minnorm

Authors:       Victor Gassmann
Written:       15-October-2019
Python translation: 2025
"""

import numpy as np
import pytest
from cora_python.contSet.zonotope import Zonotope
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol


def testLong_zonotope_minnorm():
    """Unit test function of minnorm - mirrors MATLAB testLong_zonotope_minnorm.m"""
    
    # Loop over dimension (2 to 7 as in MATLAB)
    for i in range(2, 8):
        # Loop over number of generators (i to 20, step 5 as in MATLAB)
        for j in range(i, 21, 5):
            # Init random zonotope
            # MATLAB: Z = zonotope([zeros(i,1),randn(i,j)])
            c = np.zeros((i, 1))
            G = np.random.randn(i, j)
            Z = Zonotope(c, G)
            
            # Compute minnorm
            # MATLAB: val = minnorm(Z);
            val = Z.minnorm()[0]  # Only need the value
            
            # Evaluate support function for random unit directions
            # MATLAB: L = eq_point_set(i-1,2*j);
            # Since we don't have eq_point_set, we'll generate random unit directions
            num_directions = 2 * j
            for k in range(num_directions):
                # Generate random direction and normalize
                direction = np.random.randn(i, 1)
                direction = direction / np.linalg.norm(direction)
                
                # MATLAB: sF = supportFunc(Z,L(:,k));
                sF = Z.supportFunc_(direction, 'upper')[0]
                
                # MATLAB: assertLoop(val <= sF || withinTol(val,sF),i,j,k)
                # Ensure that val <= suppfnc(Z,l)
                assert val <= sF or withinTol(val, sF, 1e-10), \
                    f"Failed for dim={i}, ngens={j}, direction={k}"


if __name__ == '__main__':
    testLong_zonotope_minnorm()
    print("testLong_zonotope_minnorm passed!") 