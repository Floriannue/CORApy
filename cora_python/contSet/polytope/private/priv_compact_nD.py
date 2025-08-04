"""
priv_compact_nD - removes all redundant constraints for an nD polytope

Description:
    Removes all redundant constraints for an nD polytope using
    linear programming to check constraint redundancy.

Syntax:
    A, b, empty = priv_compact_nD(A, b, Ae, be, n, tol)

Inputs:
    A - inequality constraint matrix
    b - inequality constraint offset
    Ae - equality constraint matrix
    be - equality constraint offset
    n - dimension of the polytope
    tol - tolerance

Outputs:
    A - inequality constraint matrix
    b - inequality constraint offset
    empty - true/false whether polytope is empty

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       03-October-2024 (MATLAB)
Python translation: 2025
"""

import numpy as np
from scipy.optimize import linprog
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol


def priv_compact_nD(A, b, Ae, be, n, tol):
    """
    Removes all redundant constraints for an nD polytope
    
    Args:
        A: inequality constraint matrix
        b: inequality constraint offset
        Ae: equality constraint matrix
        be: equality constraint offset
        n: dimension of the polytope
        tol: tolerance
        
    Returns:
        A: inequality constraint matrix
        b: inequality constraint offset
        empty: true/false whether polytope is empty
    """
    empty = False
    
    if A is None or A.size == 0:
        return A, b, empty
    
    # First check if the polytope is empty
    from .priv_representsa_emptySet import priv_representsa_emptySet
    if priv_representsa_emptySet(A, b, Ae, be, n, tol):
        empty = True
        return np.array([]), np.array([]), empty
    
    # Check for redundant constraints using the MATLAB approach
    # For each constraint A[i,:] * x <= b[i], we compute the support function
    # for the polytope without the i-th constraint in direction A[i,:]
    
    nrConIneq = A.shape[0]
    idxKeep = np.ones(nrConIneq, dtype=bool)
    irredundantIneq = np.zeros(nrConIneq, dtype=bool)
    
    for i in range(nrConIneq):
        # Skip if already marked as irredundant
        if irredundantIneq[i]:
            continue
            
        # Remove i-th constraint temporarily
        H = A[idxKeep & (np.arange(nrConIneq) != i), :]
        d = b[idxKeep & (np.arange(nrConIneq) != i)]
        
        # Ensure H and d retain 2D structure even if only one row is selected
        if H.ndim == 1:
            H = H.reshape(1, -1)
        if d.ndim == 0:
            d = d.reshape(1, 1)

        # Compute support function in direction of i-th constraint
        val, extreme_point = priv_supportFunc(H, d, Ae, be, A[i, :], 'upper')
        
        # If support function returns -Inf, the reduced polytope is empty
        # which means the original polytope is empty too
        if val == -np.inf:
            empty = True
            return np.array([]), np.array([]), empty
        
        # Check if constraint is redundant
        if val < b[i] or withinTol(val, b[i], tol):
            # Constraint is redundant
            idxKeep[i] = False
            
            # Mark constraints that are active at the extreme point as irredundant
            if extreme_point is not None:
                cons_active = withinTol(A @ extreme_point, b, tol)
                irredundantIneq = irredundantIneq | cons_active
    
    # Remove redundant constraints
    A = A[idxKeep, :]
    b = b[idxKeep]
    
    return A, b, empty


def priv_supportFunc(A, b, Ae, be, direction, bound_type='upper'):
    """
    Compute support function for polytope A*x <= b, Ae*x = be in given direction
    
    Args:
        A: inequality constraint matrix
        b: inequality constraint offset  
        Ae: equality constraint matrix
        be: equality constraint offset
        direction: direction vector
        bound_type: 'upper' for max, 'lower' for min
        
    Returns:
        val: support function value
        extreme_point: point where maximum/minimum is achieved
    """
    try:
        # Set up the linear program
        if bound_type == 'upper':
            c = -direction.flatten()  # Minimize -direction^T * x (= maximize direction^T * x)
        else:
            c = direction.flatten()   # Minimize direction^T * x
        
        # Inequality constraints
        A_ub = A if A is not None and A.size > 0 else np.array([[]]).reshape(0, n) # Ensure empty 2D array
        b_ub = b.flatten() if b is not None and b.size > 0 else np.array([[]]).reshape(0, 1).flatten() # Ensure empty 1D array
        
        # Equality constraints
        A_eq = Ae if Ae is not None and Ae.size > 0 else np.array([[]]).reshape(0, n) # Ensure empty 2D array
        b_eq = be.flatten() if be is not None and be.size > 0 else np.array([[]]).reshape(0, 1).flatten() # Ensure empty 1D array
        
        # Set bounds to allow negative values (scipy defaults to x >= 0!)
        n = len(direction)
        bounds = [(None, None)] * n
        
        # Solve the LP
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                     bounds=bounds, method='highs', options={'presolve': True})
        
        if res.success:
            if bound_type == 'upper':
                val = -res.fun  # Convert back from minimization
            else:
                val = res.fun
            extreme_point = res.x
            return val, extreme_point
        elif res.status == 2:  # Infeasible
            if bound_type == 'upper':
                return -np.inf, None  # Cannot maximize over empty set
            else:
                return np.inf, None   # Cannot minimize over empty set
        elif res.status == 3:  # Unbounded
            if bound_type == 'upper':
                return np.inf, None   # Maximization is unbounded
            else:
                return -np.inf, None  # Minimization is unbounded
        else:
            # Other solver issues - be conservative
            if bound_type == 'upper':
                return np.inf, None   # Assume unbounded for safety
            else:
                return -np.inf, None
            
    except Exception:
        # If LP solver fails, be conservative
        if bound_type == 'upper':
            return np.inf, None
        else:
            return -np.inf, None 