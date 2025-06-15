"""
gramSchmidt - constructs an orthonormal basis from the input vectors

Syntax:
    Q = gramSchmidt(V)

Inputs:
    V - matrix where each column is a vector

Outputs:
    Q - matrix where each column is an orthonormal vector

Example:
    V = [[1, 1], [0, 1]]
    Q = gramSchmidt(V)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors: Mark Wetzlinger
         Python translation by AI Assistant
Written: 07-October-2019
Last update: ---
Last revision: ---
"""

import numpy as np
from typing import Optional


def gramSchmidt(V: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """
    Constructs an orthonormal basis from the input vectors using Gram-Schmidt process.
    
    Args:
        V: Matrix where each column is a vector
        tol: Tolerance for determining linear dependence
        
    Returns:
        Q: Matrix where each column is an orthonormal vector
    """
    V = np.array(V, dtype=float)
    
    if V.size == 0:
        return V
    
    m, n = V.shape
    Q = np.zeros((m, 0))
    
    for i in range(n):
        # Get the i-th column
        v = V[:, i:i+1]  # Keep as column vector
        
        # Subtract projections onto previous orthonormal vectors
        for j in range(Q.shape[1]):
            q_j = Q[:, j:j+1]  # Keep as column vector
            proj = np.dot(q_j.T, v) * q_j
            v = v - proj
        
        # Normalize
        norm_v = np.linalg.norm(v)
        
        # Only add if not linearly dependent (norm is significant)
        if norm_v > tol:
            v_normalized = v / norm_v
            Q = np.hstack([Q, v_normalized])
    
    return Q 