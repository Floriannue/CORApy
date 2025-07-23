"""
vecalign - computes T such that x || T*y (x parallel to T*y)

Syntax:
   T = vecalign(x,y)

Inputs:
   x - vector (numpy array)
   y - vector (numpy array)

Outputs:
   T - transformation matrix (numpy array)

Example:
   # (No example provided in MATLAB source, will create one in test)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: ---

Authors:       ??? (MATLAB)
               Automatic python translation: Florian Nüssel BA 2025
Written:       --- (MATLAB)
Last update:   --- (MATLAB)
Last revision: --- (MATLAB)
"""

import numpy as np
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def vecalign(x, y):
    # Ensure x and y are treated as column vectors for SVD
    x_col = x.reshape(-1, 1) if x.ndim == 1 else x
    y_col = y.reshape(-1, 1) if y.ndim == 1 else y

    if x_col.shape[0] != y_col.shape[0]:
        raise CORAerror('CORA:vecalign:dimensionMismatch', 'Input vectors x and y must have the same dimension.')
    if x_col.shape[0] == 0:
        raise CORAerror('CORA:vecalign:emptyInput', 'Input vectors cannot be empty.')
    if np.linalg.norm(x_col) == 0 or np.linalg.norm(y_col) == 0:
        # Handle zero vectors: if one is zero, alignment is undefined or T can be arbitrary.
        # For simplicity, if one is zero, return identity or raise error depending on desired behavior.
        # MATLAB's svd on zero matrix gives zero singular values, U and Vh are arbitrary unitary.
        # Let's return identity if one is zero and dimensions match.
        if np.linalg.norm(x_col) == 0 and np.linalg.norm(y_col) == 0:
            return np.eye(x_col.shape[0])
        else:
            # If one is zero and other is not, it's problematic for 'parallel to T*y' unless T is zero matrix.
            # This case is likely an error or needs specific handling outside generic vecalign.
            # Raise error for now, as aligning a non-zero vector with a zero vector doesn't make sense for rotation.
            raise CORAerror('CORA:vecalign:zeroVectorInput', 'Cannot align a non-zero vector with a zero vector.')


    x_norm = x_col / np.linalg.norm(x_col)
    y_norm = y_col / np.linalg.norm(y_col)

    # U, s, Vh = np.linalg.svd(A)
    # For a vector (n, 1), svd returns U (n,n), s (1,), Vh (1,1)
    U1, _, _ = np.linalg.svd(x_norm)
    U2, _, _ = np.linalg.svd(y_norm)

    T = U1 @ U2.T

    return T 