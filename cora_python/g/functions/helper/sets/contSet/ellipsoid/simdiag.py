"""
simdiag - Find T such that
           T*M1*T' = I and T*M2*T' = D (diagonal)

Syntax:
   [T,D] = simdiag(M1,M2,TOL)

Inputs:
   M1 - numerical matrix
   M2 - numerical matrix
   TOL - tolerance

Outputs:
   T - transformation matrix
   D - result of T*M2*T' (diagonal matrix)

Example: 

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: ---

Authors:       Victor Gassmann
Written:       06-June-2022 
Last update:   ---
Last revision: ---
Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.matlab.validate.check.isApproxSymmetric import isApproxSymmetric
from typing import Tuple

def simdiag(M1: np.ndarray, M2: np.ndarray, TOL: float) -> Tuple[np.ndarray, np.ndarray]:

    if M1.shape != M2.shape:
        raise CORAerror('CORA:dimensionMismatch', M1, M2)

    # check if both are symmetric, M1 is pd, and M2 psd (probably too strict of a condition)
    # The MATLAB eig function returns eigenvalues. For positive definiteness, all eigenvalues should be > TOL.
    # For positive semi-definiteness, all eigenvalues should be >= -TOL.
    # np.linalg.eigvalsh is for symmetric/Hermitian matrices, which is appropriate here.
    if not isApproxSymmetric(M1, TOL) or not isApproxSymmetric(M2, TOL) or \
       np.min(np.linalg.eigvalsh(M1)) < TOL or np.min(np.linalg.eigvalsh(M2)) < -TOL:
        raise CORAerror('CORA:specialError',
                        'Both matrices need to be symmetric, first matrix needs to be pd, second needs to be psd!')

    # svd is singular value decomposition, used for symmetric matrices in MATLAB here
    # In Python, for symmetric matrices, eig or eigh are typically used for eigenvalues/eigenvectors.
    # However, for svd, we should stick to np.linalg.svd.
    U1, S1, V1_T = np.linalg.svd(M1)
    S1_12inv = np.diag(1./np.sqrt(S1))

    # MATLAB's svd(A) returns [U,S,V], where A = U*S*V'. Here, V1_T is V'.
    # So U1' * M2 * U1 is equivalent to U1.T @ M2 @ U1
    # S1_12inv @ U1.T @ M2 @ U1 @ S1_12inv
    # The multiplication order for S1_12inv needs to be correct. S1_12inv is diagonal.
    intermediate_matrix = S1_12inv @ U1.T @ M2 @ U1 @ S1_12inv
    U2, S2, V2_T = np.linalg.svd(intermediate_matrix)

    T = U2.T @ S1_12inv @ U1.T

    # The original MATLAB code implicitly returns D if nargout > 1.
    # In Python, we explicitly return both T and D.
    D = T @ M2 @ T.T
    
    return T, D 