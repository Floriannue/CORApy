"""
arnoldi - computes the Arnoldi iteration for building a (Krylov) subspace

Syntax:
    [V,H,Hlast,happyBreakdown] = arnoldi(obj,options)

Inputs:
    A - system matrix
    vInit - initial value of the vector that is multiplied
    redDim - reduced dimension

Outputs:
    V - orthogonal basis of subspace
    H - transformation matrix (upper Hessenberg matrix)
    Hlast - last H(j+1,j) value
    happyBreakdown - boolean whether prematurely finished

Example: 
    -

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: ---

Authors:       Matthias Althoff
Written:       15-November-2016 
Last update:   22-December-2016
               06-November-2018
               09-June-2020 (MW, moved here from linearSys/private)
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from scipy.sparse import csc_matrix, issparse
from typing import Tuple


def arnoldi(A: np.ndarray, vInit: np.ndarray, redDim: int) -> Tuple[np.ndarray, np.ndarray, float, bool]:
    """
    Computes the Arnoldi iteration for building a (Krylov) subspace
    
    Args:
        A: system matrix
        vInit: initial value of the vector that is multiplied
        redDim: reduced dimension
        
    Returns:
        V: orthogonal basis of subspace
        H: transformation matrix (upper Hessenberg matrix)
        Hlast: last H(j+1,j) value
        happyBreakdown: boolean whether prematurely finished
    """
    
    # Convert to numpy arrays if sparse
    if issparse(A):
        A = A.toarray()
    if issparse(vInit):
        vInit = vInit.toarray().flatten()
    else:
        vInit = np.asarray(vInit).flatten()
    
    # preallocate H
    H = np.zeros((redDim, redDim))
    
    # initialize 
    v_norm = np.linalg.norm(vInit)
    if v_norm == 0:
        raise ValueError("vInit cannot be zero vector")
    V = np.zeros((len(vInit), redDim + 1))
    V[:, 0] = vInit / v_norm
    happyBreakdown = False
    
    # compute elements of transformation matrix
    j = 0
    for j in range(redDim):
        # update v
        w = A @ V[:, j]
        
        # generate column vector of H
        for i in range(j + 1):
            H[i, j] = w.T @ V[:, i]
            w = w - H[i, j] * V[:, i]
        
        # last element and update of q{k}
        H[j + 1, j] = np.linalg.norm(w)
        # happy-breakdown?
        if H[j + 1, j] <= np.finfo(float).eps:
            happyBreakdown = True
            break
        V[:, j + 1] = w / H[j + 1, j]
    
    # save H(j+1,j)
    Hlast = H[j + 1, j] if j < redDim else 0.0
    
    # remove last column of V
    if not happyBreakdown:  # no happy breakdown
        V = V[:, :redDim]  # Remove last column
        # remove last row of H
        H = H[:redDim, :]  # Remove last row
    else:
        # reduce H due to happy breakdown
        H = H[:j + 1, :j + 1]
        V = V[:, :j + 1]
    
    return V, H, Hlast, happyBreakdown

