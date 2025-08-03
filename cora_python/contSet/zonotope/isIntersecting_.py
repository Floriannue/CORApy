"""
isIntersecting_ - determines if zonotope intersects a set

Syntax:
    res = isIntersecting_(Z, S, type, tol)

Inputs:
    Z - zonotope object
    S - contSet object
    type - type of check ('exact' or 'approx')
    tol - tolerance

Outputs:
    res - true/false

Example: 
    Z1 = Zonotope(np.array([[0, 1, 1, 0], [0, 1, 0, 1]]))
    Z2 = Zonotope(np.array([[2, -1, 1, 0], [2, 1, 0, 1]]))
    Z3 = Zonotope(np.array([[3.5, -1, 1, 0], [3, 1, 0, 1]]))

    isIntersecting_(Z1, Z2)
    isIntersecting_(Z1, Z3)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: contSet/isIntersecting, conZonotope/isIntersecting_

Authors:       Niklas Kochdumper (MATLAB)
               Python translation by AI Assistant
Written:       21-November-2019 (MATLAB)
Last update:   27-March-2023 (MW, rename isIntersecting_) (MATLAB)
                2025 (Tiange Yang, Florian Nüssel, Python translation by AI Assistant)
"""

import numpy as np
from typing import Union, Any
from .zonotope import Zonotope
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def isIntersecting_(Z: Zonotope, S: Any, type: str = 'exact', tol: float = 1e-9) -> bool:
    """
    Determines if zonotope intersects a set
    
    Args:
        Z: zonotope object
        S: contSet object or numeric array
        type: type of check ('exact' or 'approx')
        tol: tolerance
        
    Returns:
        True if sets intersect, False otherwise
    """
    # Numeric case: check containment
    if isinstance(S, np.ndarray):
        return Z.contains_(S, type, tol, 0, False, False)[0]
    
    # Sets must not be empty
    if Z.representsa_('emptySet', 0) or S.representsa_('emptySet', 0):
        return False
    
    # Check if S is a contSet with lower precedence
    if hasattr(S, 'precedence') and S.precedence < Z.precedence:
        return S.isIntersecting_(Z, type, tol)
    
    # Handle different set types
    if hasattr(S, 'isBounded') and S.isBounded():
        # Both bounded, convert to constrained zonotope
        cZ1 = Z.conZonotope()
        cZ2 = S.conZonotope()
        return cZ1.isIntersecting_(cZ2, type, tol)
    else:
        # S unbounded, use polytope function
        P = S.polytope()
        return P.isIntersecting_(Z, type, tol)
    
    # If we get here, the operation is not supported
    raise CORAerror('CORA:noops', f"Intersection check not supported between {type(Z)} and {type(S)}") 