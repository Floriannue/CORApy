"""
isequal - checks if a zonotope is equal to another set or point

Syntax:
   res = isequal(Z, S)
   res = isequal(Z, S, tol)

Inputs:
   Z - zonotope object
   S - contSet object, numeric
   tol - (optional) tolerance

Outputs:
   res - true/false

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       16-September-2019 (MATLAB)
Last update:   09-June-2020 (MATLAB)
                13-November-2022 (MW, integrate modular check)
                05-October-2024 (MW, fix 1D case)
               2025 (Tiange Yang, Florian Nüssel, Python translation by AI Assistant)
"""

import numpy as np
from typing import Any, Optional
from .zonotope import Zonotope
from cora_python.contSet.interval.interval import Interval
from cora_python.contSet.zonotope.compact_ import compact_
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol
from cora_python.g.functions.matlab.validate.check.compareMatrices import compareMatrices

def isequal(Z, S, tol: Optional[float] = None):
    """
    Checks if a zonotope is equal to another set or point (order/sign/zero generators ignored)
    """
    if tol is None:
        tol = np.finfo(float).eps

    # Convert intervals and numeric to Zonotope
    if isinstance(S, Interval):
        S = Zonotope(S)
    elif isinstance(S, (int, float, np.ndarray, list, tuple)):
        S = Zonotope(np.asarray(S))
    if not isinstance(S, Zonotope):
        return False

    # Lower precedence call (not implemented for other contSet types in Python, skip)
    # if hasattr(S, 'precedence') and hasattr(Z, 'precedence') and S.precedence < Z.precedence:
    #     return isequal(S, Z, tol)

    # Dimension check
    if Z.dim() != S.dim():
        return False

    return aux_isequal_zonotope(Z, S, tol)

def aux_isequal_zonotope(Z, S, tol):
    # Both empty: check dimension
    if Z.is_empty() and S.is_empty():
        return Z.dim() == S.dim()
    # One empty, one not
    if Z.is_empty() or S.is_empty():
        return False
    # Compare centers
    if not np.all(withinTol(Z.c, S.c, tol)):
        return False
    # Obtain minimal representation
    G1 = compact_(Z, 'all', tol).G
    G2 = compact_(S, 'all', tol).G
    # Compare number of generators
    if G1.shape[1] != G2.shape[1]:
        return False
    # Compare generator matrices: must match with no remainder, order is irrelevant, sign may be inverted
    return compareMatrices(G1, G2, tol, flag='equal', ordered=False, signed=True) 