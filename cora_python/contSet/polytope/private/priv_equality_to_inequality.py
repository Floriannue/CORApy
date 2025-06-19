"""
priv_equality_to_inequality - rewrites all equality constraints as inequality constraints

Description:
    Converts equality constraints Ae*x = be to inequality constraints
    by replacing each equality with two inequalities:
    Ae*x <= be and -Ae*x <= -be

Syntax:
    A, b = priv_equality_to_inequality(A, b, Ae, be)

Inputs:
    A - inequality constraint matrix
    b - inequality constraint offset
    Ae - equality constraint matrix
    be - equality constraint offset

Outputs:
    A - inequality constraint matrix (augmented)
    b - inequality constraint offset (augmented)

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       03-October-2024 (MATLAB)
Python translation: 2025
"""

import numpy as np


def priv_equality_to_inequality(A, b, Ae, be):
    """
    Rewrites all equality constraints as inequality constraints
    
    Args:
        A: inequality constraint matrix
        b: inequality constraint offset  
        Ae: equality constraint matrix
        be: equality constraint offset
        
    Returns:
        A: augmented inequality constraint matrix
        b: augmented inequality constraint offset
    """
    if Ae is None or Ae.size == 0:
        return A, b
        
    if A is None or A.size == 0:
        A = np.vstack([Ae, -Ae])
        b = np.vstack([be, -be])
    else:
        A = np.vstack([A, Ae, -Ae])
        b = np.vstack([b, be, -be])
    
    return A, b 