"""
calcSquaredG - computes the multiplicative G1' * G2

This is a Python translation of the MATLAB CORA implementation.

Authors: MATLAB: Tobias Ladner
         Python: AI Assistant
"""

import numpy as np
from typing import Optional


def calcSquaredG(G1: np.ndarray, G2: np.ndarray, isEqual: Optional[bool] = None) -> np.ndarray:
    """
    Compute the multiplicative G1' * G2.
    
    Args:
        G1: row vector of generator matrix 1
        G2: row vector of generator matrix 2
        isEqual: whether they are equal for optimizations (default: False)
        
    Returns:
        G_quad: result of G1' * G2 as row vector
        
    See also: polyZonotope/quadMap, nnHelper/calcSquared
    """
    if isEqual is None:
        isEqual = False
    
    if G1.size > 0 and G2.size > 0:
        if isEqual:
            temp = G1.T @ G2
            
            # we can ignore the left lower triangle in this case
            # as it's the same as the right upper triangle
            # -> double right upper triangle
            n = G1.shape[1]
            G_quad = np.zeros((1, int(0.5 * n * (n + 1))))
            cnt = n
            
            for i in range(n - 1):
                G_quad[0, i] = temp[i, i]
                G_quad[0, cnt:cnt + n - i - 1] = 2 * temp[i, i + 1:n]
                cnt = cnt + n - i - 1
            G_quad[0, n - 1] = temp[-1, -1]
        else:
            # calculate all values
            G_quad = G1.T @ G2
            G_quad = G_quad.reshape(1, -1)  # row vector
    else:
        G_quad = np.array([]).reshape(1, 0)
    
    return G_quad
