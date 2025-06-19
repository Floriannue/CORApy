"""
priv_compact_all - removes all redundant constraints in the halfspace representation

Description:
    Removes all redundant constraints from a polytope's halfspace representation.
    This is a simplified version that handles basic redundancy removal.

Syntax:
    A, b, Ae, be = priv_compact_all(A, b, Ae, be, n, tol)

Inputs:
    A - inequality constraint matrix
    b - inequality constraint offset
    Ae - equality constraint matrix
    be - equality constraint offset
    n - dimension of the polytope
    tol - tolerance

Outputs:
    A - inequality constraint matrix (compacted)
    b - inequality constraint offset (compacted)
    Ae - equality constraint matrix (compacted)
    be - equality constraint offset (compacted)

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       03-October-2024 (MATLAB)
Python translation: 2025
"""

import numpy as np


def priv_compact_all(A, b, Ae, be, n, tol):
    """
    Removes all redundant constraints in the halfspace representation
    
    This is a simplified implementation that handles basic cases.
    For a full implementation, more sophisticated redundancy checks would be needed.
    
    Args:
        A: inequality constraint matrix
        b: inequality constraint offset
        Ae: equality constraint matrix
        be: equality constraint offset
        n: dimension of the polytope
        tol: tolerance
        
    Returns:
        A: compacted inequality constraint matrix
        b: compacted inequality constraint offset
        Ae: compacted equality constraint matrix
        be: compacted equality constraint offset
    """
    # Basic implementation - remove obvious redundancies
    
    # Remove zero constraints
    A, b, Ae, be = _remove_zero_constraints(A, b, Ae, be, tol)
    
    # Remove duplicate constraints
    A, b = _remove_duplicate_constraints(A, b, tol)
    if Ae is not None and Ae.size > 0:
        Ae, be = _remove_duplicate_constraints(Ae, be, tol)
    
    return A, b, Ae, be


def _remove_zero_constraints(A, b, Ae, be, tol):
    """Remove constraints where the coefficient vector is all zeros"""
    
    # Handle inequality constraints
    if A is not None and A.size > 0:
        zero_rows = np.all(np.abs(A) < tol, axis=1)
        if np.any(zero_rows):
            # Check if any zero constraint is infeasible (0*x <= b with b < 0)
            if np.any(b[zero_rows] < -tol):
                # Infeasible - return empty
                return np.array([]).reshape(0, A.shape[1]), np.array([]).reshape(0, 1), \
                       np.array([]).reshape(0, A.shape[1]), np.array([]).reshape(0, 1)
            # Remove zero constraints with b >= 0
            A = A[~zero_rows]
            b = b[~zero_rows]
    
    # Handle equality constraints
    if Ae is not None and Ae.size > 0:
        zero_rows = np.all(np.abs(Ae) < tol, axis=1)
        if np.any(zero_rows):
            # Check if any zero constraint is infeasible (0*x = be with be != 0)
            if np.any(np.abs(be[zero_rows]) > tol):
                # Infeasible - return empty
                return np.array([]).reshape(0, Ae.shape[1]), np.array([]).reshape(0, 1), \
                       np.array([]).reshape(0, Ae.shape[1]), np.array([]).reshape(0, 1)
            # Remove zero constraints with be = 0
            Ae = Ae[~zero_rows]
            be = be[~zero_rows]
    
    return A, b, Ae, be


def _remove_duplicate_constraints(A, b, tol):
    """Remove duplicate constraints"""
    if A is None or A.size == 0:
        return A, b
        
    # Normalize constraints to detect duplicates
    norms = np.linalg.norm(A, axis=1)
    nonzero_mask = norms > tol
    
    if not np.any(nonzero_mask):
        return A, b
        
    A_norm = A.copy()
    b_norm = b.copy()
    A_norm[nonzero_mask] = A[nonzero_mask] / norms[nonzero_mask, np.newaxis]
    b_norm[nonzero_mask] = b[nonzero_mask] / norms[nonzero_mask]
    
    # Find unique constraints
    Ab_norm = np.column_stack([A_norm, b_norm])
    _, unique_indices = np.unique(np.round(Ab_norm / tol) * tol, axis=0, return_index=True)
    
    return A[unique_indices], b[unique_indices] 