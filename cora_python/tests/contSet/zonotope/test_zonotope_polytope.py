"""
test_zonotope_polytope - unit test function of polytope

This module tests the polytope method for zonotope objects.

Authors:       Matthias Althoff (MATLAB)
               Python translation by AI Assistant
Written:       26-July-2016 (MATLAB)
Python translation: 2025
"""

import pytest
import numpy as np
from cora_python.contSet.zonotope import Zonotope

def test_zonotope_polytope():
    """Unit test function of polytope - mirrors MATLAB test_zonotope_polytope.m"""
    
    # create zonotope
    Z1 = Zonotope(np.array([[-4, -3, -2, -1], [1, 2, 3, 4]]))
    
    # obtain polytope (get first return value from tuple)
    P, comb, isDeg = Z1.polytope('exact')
    
    # true results (from MATLAB)
    true_C = np.array([[ 0.554700196225229,   0.832050294337844],
                       [ 0.832050294337844,   0.554700196225229],
                       [ 0.970142500145332,   0.242535625036333],
                       [-0.554700196225229,  -0.832050294337844],
                       [-0.832050294337844,  -0.554700196225229],
                       [-0.970142500145332,  -0.242535625036333]])
    
    true_d = np.array([2.773500981126146, 0, 0, 5.547001962252290, 5.547001962252290, 7.276068751089989])
    
    # Check that all constraints from MATLAB are present in the Python result
    # (order might be different, but all constraints should be there)
    found_constraints = 0
    for i in range(true_C.shape[0]):
        for j in range(P.A.shape[0]):
            b_python = P.b[j] if len(P.b.shape) == 1 else P.b[j, 0]
            if np.allclose(P.A[j, :], true_C[i, :], atol=1e-13) and np.allclose(b_python, true_d[i], atol=1e-13):
                found_constraints += 1
                break
    
    assert found_constraints == true_C.shape[0], f"Found {found_constraints} out of {true_C.shape[0]} expected constraints"
    
    # check that polytope is bounded
    assert P.bounded == True 