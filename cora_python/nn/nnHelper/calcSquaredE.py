"""
calcSquaredE - computes the multiplicative E1' * E2

This is a Python translation of the MATLAB CORA implementation.

Authors: MATLAB: Tobias Ladner
         Python: AI Assistant
"""

import numpy as np
from .calcSquaredGInd import calcSquaredGInd


def calcSquaredE(E1: np.ndarray, E2: np.ndarray, isEqual: bool) -> np.ndarray:
    """
    Compute the multiplicative E1' * E2.
    
    Args:
        E1: exponential matrix 1
        E2: exponential matrix 2
        isEqual: whether they are equal for optimizations
        
    Returns:
        E_quad: result of E1' * E2 as row vector
        
    See also: polyZonotope/quadMap, nnHelper/calcSquared
    """
    # E1, E2 exponential matrices; calculate E1'*E2
    if E1.size > 0 and E2.size > 0:
        G1_ind, G2_ind, G1_ind2, G2_ind2 = calcSquaredGInd(E1[0, :], E2[0, :], isEqual)
        E_quad = np.hstack([E1[:, G1_ind] + E2[:, G2_ind], E1[:, G1_ind2] + E2[:, G2_ind2]])
    else:
        E_quad = np.array([]).reshape(0, 0)
    
    return E_quad
