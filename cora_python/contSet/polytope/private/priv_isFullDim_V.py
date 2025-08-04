"""
priv_isFullDim_V - checks if the dimension of the affine hull of a 
    polytope is equal to the dimension of its ambient space; additionally,
    one can obtain a basis of the subspace in which the polytope is
    contained

Syntax:
    res = priv_isFullDim_V(V, tol)
    res, subspace = priv_isFullDim_V(V, tol)

Inputs:
    V - vertex representation
    tol - tolerance

Outputs:
    res - true/false
    subspace - (optional) Returns a set of orthogonal unit vectors
               x_1,...,x_k such that P is strictly contained in
               center(P)+span(x_1,...,x_k)
               (here, 'strictly' means that k is minimal).
               Note that if P is just a point, subspace=[].

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       03-October-2024 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import Tuple, Optional


def priv_isFullDim_V(V: np.ndarray, tol: float) -> Tuple[bool, Optional[np.ndarray]]:
    """
    Checks if vertex representation defines a full-dimensional polytope.
    
    Args:
        V: vertex representation (n x num_vertices)
        tol: tolerance
        
    Returns:
        res: True/False whether polytope is full-dimensional
        subspace: orthogonal unit vectors defining the subspace
    """
    
    # read out dimension
    n = V.shape[0]
    
    # compute rank shifted by mean
    V_shifted = V - np.mean(V, axis=1, keepdims=True)
    rankV = np.linalg.matrix_rank(V_shifted, tol=tol)
    
    # compare to ambient dimension
    res = rankV == n
    
    # compute basis of affine hull
    if res:
        subspace = np.eye(n)
    else:
        Q, R = np.linalg.qr(V_shifted)
        subspace = Q[:, :rankV]
    
    return res, subspace