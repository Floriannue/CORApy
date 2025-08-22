"""
compBoundsPolyZono - compute the lower and upper bound of a polynomial
   zonotope

This is a Python translation of the MATLAB CORA implementation.

Authors: MATLAB: Niklas Kochdumper, Tobias Ladner
         Python: AI Assistant
"""

import numpy as np
from typing import Tuple


def compBoundsPolyZono(c: np.ndarray, G: np.ndarray, GI: np.ndarray, E: np.ndarray, 
                       ind: np.ndarray, ind_: np.ndarray, approx: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the lower and upper bound of a polynomial zonotope.
    
    Args:
        c: center of polyZonotope in a dimension
        G: corresponding dep. generator of polyZonotope as row vector
        GI: corresponding indep. generator of polyZonotope as row vector
        E: exponential matrix of polyZonotope
        ind: all even indices
        ind_: all odd indices
        approx: whether to use approximation
        
    Returns:
        l, u: lower and upper bound
        
    See also: -
    """
    if approx:
        # using zonotope over-approximation
        c_ = c + 0.5 * np.sum(G[:, ind], axis=1, keepdims=True)
        
        l = c_ - np.sum(np.abs(0.5 * G[:, ind]), axis=1, keepdims=True) - \
              np.sum(np.abs(G[:, ind_]), axis=1, keepdims=True) - \
              np.sum(np.abs(GI), axis=1, keepdims=True)
        u = c_ + np.sum(np.abs(0.5 * G[:, ind]), axis=1, keepdims=True) + \
              np.sum(np.abs(G[:, ind_]), axis=1, keepdims=True) + \
              np.sum(np.abs(GI), axis=1, keepdims=True)
    else:
        # tighter bounds using splitting
        from cora_python.contSet.polyZonotope import PolyZonotope
        
        # Create polyZonotope object
        pZ = PolyZonotope(c, G, GI, E)
        
        # Compute interval using split method
        int_result = pZ.interval('split')
        
        # Extract bounds
        l = int_result.inf.reshape(-1, 1)
        u = int_result.sup.reshape(-1, 1)
    
    return l, u
