"""
priv_compact_zeros - removes all constraints where the vector is all-zero

Description:
    1) removes all all-zero inequality constraints, i.e., 0*x <= a
    also, if a < 0, then the constraint is infeasible and the set is empty
    2) removes all all-zero equality constraints, i.e., 0*x = a, a = 0
    also, if a ~= 0, then the constraint is infeasible and the set is empty

Syntax:
    A, b, Ae, be, empty = priv_compact_zeros(A, b, Ae, be, tol)

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
    empty - true/false whether polytope is empty

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       03-October-2024 (MATLAB)
Python translation: 2025
"""

import numpy as np
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol


def priv_compact_zeros(A, b, Ae, be, tol):
    """
    Removes all constraints where the vector is all-zero
    
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
        empty: true/false whether polytope is empty
    """
    empty = False
    
    # Handle inequality constraints
    if A is not None and A.size > 0:
        # Find rows with 0*x <= b
        A_zero_rows = np.where(np.all(withinTol(A, 0), axis=1))[0]
        if len(A_zero_rows) > 0:
            # Check whether any of those constraints is infeasible
            if np.any((b[A_zero_rows] < 0) & ~withinTol(b[A_zero_rows], 0, tol)):
                # constraint 0*x <= b with b < 0  =>  infeasible, i.e., empty set
                empty = True
                A = np.array([]).reshape(0, A.shape[1])
                b = np.array([]).reshape(0, 1)
                Ae = np.array([]).reshape(0, A.shape[1]) if A.shape[1] > 0 else np.array([]).reshape(0, 0)
                be = np.array([]).reshape(0, 1)
                return A, b, Ae, be, empty
            else:
                # all constraints are 0*x <= b where all b_i > 0
                A = np.delete(A, A_zero_rows, axis=0)
                b = np.delete(b, A_zero_rows, axis=0)
    
    # Handle equality constraints
    if Ae is not None and Ae.size > 0:
        # Find rows with 0*x = 0
        Ae_zero_rows = np.where(np.all(withinTol(Ae, 0), axis=1))[0]
        if len(Ae_zero_rows) > 0:
            # Check whether any of those constraints is infeasible
            if np.any(~withinTol(be[Ae_zero_rows], 0, tol)):
                # constraint 0*x = be with be ~= 0  =>  infeasible, i.e., empty set
                empty = True
                A = np.array([]).reshape(0, Ae.shape[1]) if Ae.shape[1] > 0 else np.array([]).reshape(0, 0)
                b = np.array([]).reshape(0, 1)
                Ae = np.array([]).reshape(0, Ae.shape[1])
                be = np.array([]).reshape(0, 1)
                return A, b, Ae, be, empty
            else:
                # all constraints are 0*x = 0
                Ae = np.delete(Ae, Ae_zero_rows, axis=0)
                be = np.delete(be, Ae_zero_rows, axis=0)
    
    return A, b, Ae, be, empty 