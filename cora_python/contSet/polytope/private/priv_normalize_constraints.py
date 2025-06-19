"""
priv_normalize_constraints - normalizes constraints

Description:
    Normalizes constraints either with respect to the constraint matrix A
    or the offset vector b.

Syntax:
    A, b, Ae, be = priv_normalize_constraints(A, b, Ae, be, mode)

Inputs:
    A - inequality constraint matrix
    b - inequality constraint offset
    Ae - equality constraint matrix
    be - equality constraint offset
    mode - 'A' or 'b' for normalization mode

Outputs:
    A - inequality constraint matrix (normalized)
    b - inequality constraint offset (normalized)
    Ae - equality constraint matrix (normalized)
    be - equality constraint offset (normalized)

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       03-October-2024 (MATLAB)
Python translation: 2025
"""

import numpy as np
from cora_python.g.functions.matlab.validate.check import withinTol

def priv_normalize_constraints(A, b, Ae, be, mode):
    """
    Normalizes constraints either with respect to A or b
    
    Args:
        A: inequality constraint matrix
        b: inequality constraint offset
        Ae: equality constraint matrix
        be: equality constraint offset
        mode: 'A' or 'b' for normalization mode
        
    Returns:
        A: inequality constraint matrix (normalized)
        b: inequality constraint offset (normalized)
        Ae: equality constraint matrix (normalized)
        be: equality constraint offset (normalized)
    """
    # Make copies to avoid modifying the original arrays
    if A is not None:
        A = A.copy()
    if b is not None:
        b = b.copy().flatten()  # Ensure b is 1D
    if Ae is not None:
        Ae = Ae.copy()
    if be is not None:
        be = be.copy().flatten()  # Ensure be is 1D
    
    if mode == 'A':
        # Normalize with respect to A matrix
        if A is not None and A.size > 0:
            norms = np.linalg.norm(A, axis=1)  # Don't keep dims for broadcasting
            nonzero_norms = norms != 0
            if np.any(nonzero_norms):
                A[nonzero_norms, :] = A[nonzero_norms, :] / norms[nonzero_norms, np.newaxis]
                if b is not None:
                    b[nonzero_norms] = b[nonzero_norms] / norms[nonzero_norms]
        
        if Ae is not None and Ae.size > 0:
            norms = np.linalg.norm(Ae, axis=1)
            nonzero_norms = norms != 0
            if np.any(nonzero_norms):
                Ae[nonzero_norms, :] = Ae[nonzero_norms, :] / norms[nonzero_norms, np.newaxis]
                if be is not None:
                    be[nonzero_norms] = be[nonzero_norms] / norms[nonzero_norms]
    
    elif mode == 'b':
        # Normalize with respect to b vector
        if A is not None and A.size > 0 and b is not None and b.size > 0:
            nonzero_b = b != 0
            if np.any(nonzero_b):
                A[nonzero_b, :] = A[nonzero_b, :] / b[nonzero_b, np.newaxis]
                b[nonzero_b] = np.sign(b[nonzero_b])  # Preserve sign, normalize magnitude to 1
        
        if Ae is not None and Ae.size > 0 and be is not None and be.size > 0:
            nonzero_be = be != 0
            if np.any(nonzero_be):
                Ae[nonzero_be, :] = Ae[nonzero_be, :] / be[nonzero_be, np.newaxis]
                be[nonzero_be] = np.sign(be[nonzero_be])  # Preserve sign, normalize magnitude to 1
    
    return A, b, Ae, be 