"""
ndimCross - computes the n-dimensional cross product

This function computes the n-dimensional cross product according to [1, Sec. 5.1]

Syntax:
    v = ndimCross(Q)

Inputs:
    Q - matrix of column vectors; must be a n x (n-1) matrix

Outputs:
    v - n-dimensional cross product 

Example: 
    Q = [[1, 2], [3, 4], [5, 6]]
    v = ndimCross(Q)

Reference:
    [1] M. Althoff, O. Stursberg, M. Buss. "Computing reachable sets of
        hybrid systems using a combination of zonotopes and polytopes",
        Nonlinear Analysis: Hybrid Systems 4 (2010), p. 233-249.

Authors: Matthias Althoff, Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 14-September-2006 (MATLAB)
Last update: 22-March-2007 (MATLAB)
             01-August-2024 (MW, simplify computation) (MATLAB)
Python translation: 2025
"""

import numpy as np


def ndimCross(Q: np.ndarray) -> np.ndarray:
    """
    Compute the n-dimensional cross product
    
    Args:
        Q: Matrix of column vectors (n x (n-1))
        
    Returns:
        n-dimensional cross product vector (n x 1)
    """
    Q = np.asarray(Q, dtype=float)
    
    if Q.ndim != 2:
        raise ValueError('Q must be a 2D matrix')
    
    n, k = Q.shape
    if k != n - 1:
        raise ValueError('Q must be a n x (n-1) matrix')
    
    # Initialize result vector
    v = np.zeros((n, 1), dtype=float)
    
    # Compute cross product using determinant formula
    for i in range(n):
        # Create submatrix by removing i-th row
        rows_to_keep = list(range(i)) + list(range(i+1, n))
        submatrix = Q[rows_to_keep, :]
        
        # Compute determinant with alternating sign
        v[i, 0] = ((-1) ** (i + 1)) * np.linalg.det(submatrix)
    
    return v 