"""
priv_compact_toEquality - rewrite all pairwise inequality constraints

Description:
    Rewrites all pairwise inequality constraints Ax <= b, -Ax <= -b 
    as equality constraints Ax == b
    note: expects normalized constraints with respect to 'A'

Syntax:
    A, b, Ae, be = priv_compact_toEquality(A, b, Ae, be, tol)

Inputs:
    A - inequality constraint matrix
    b - inequality constraint offset
    Ae - equality constraint matrix
    be - equality constraint offset
    tol - tolerance

Outputs:
    A - inequality constraint matrix
    b - inequality constraint offset
    Ae - equality constraint matrix
    be - equality constraint offset

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       03-October-2024 (MATLAB)
Python translation: 2025
"""

import numpy as np
from cora_python.g.functions.matlab.auxiliary.withinTol import withinTol


def priv_compact_toEquality(A, b, Ae, be, tol):
    """
    Rewrite all pairwise inequality constraints as equality constraints
    
    Args:
        A: inequality constraint matrix
        b: inequality constraint offset
        Ae: equality constraint matrix
        be: equality constraint offset
        tol: tolerance
        
    Returns:
        A: inequality constraint matrix
        b: inequality constraint offset
        Ae: equality constraint matrix
        be: equality constraint offset
    """
    if A is None or A.size == 0:
        return A, b, Ae, be
    
    # Concatenate normal vectors and offsets
    Ab = np.column_stack([A, b])
    
    nrIneq = Ab.shape[0]
    idxMoveToEq = np.zeros(nrIneq, dtype=bool)
    
    for i in range(nrIneq):
        # Skip constraint if already matched
        if not idxMoveToEq[i]:
            # Check if there exists a constraint with an inverted normal vector
            # by computing the sum of the entries (must be zero up to tolerance)
            sum_of_vectors = Ab.T + Ab[i, :].reshape(-1, 1)
            sum_of_vectors[withinTol(sum_of_vectors, 0, tol)] = 0
            idxInverted = ~np.any(sum_of_vectors, axis=0)
            
            # Search for a match
            if np.any(idxInverted):
                idxMoveToEq = idxMoveToEq | idxInverted
                idxMoveToEq[i] = True
                
                # Add to equality constraints
                if Ae is None or Ae.size == 0:
                    Ae = A[i, :].reshape(1, -1)
                    be = b[i].reshape(1, -1)
                else:
                    Ae = np.vstack([Ae, A[i, :]])
                    be = np.vstack([be, b[i]])
    
    # Remove all pairwise inequality constraints
    A = A[~idxMoveToEq, :]
    b = b[~idxMoveToEq]
    
    return A, b, Ae, be 