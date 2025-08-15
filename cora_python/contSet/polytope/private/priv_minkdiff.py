"""
priv_minkdiff - computes the Minkowski difference of a polytope and
   another set or point using support functions

Syntax:
    [A,b,Ae,be,empty] = priv_minkdiff(A,b,Ae,be,S)

Inputs:
    A - inequality constraint matrix
    b - inequality constraint offset
    Ae - equality constraint matrix
    be - equality constraint offset
    S - contSet object

Outputs:
    A - inequality constraint matrix
    b - inequality constraint offset
    Ae - equality constraint matrix
    be - equality constraint offset
    empty - true/false whether result is the empty set

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       03-October-2024 (MATLAB)
Last update:   ---
Python translation: 2025
"""

import numpy as np
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol

def priv_minkdiff(A, b, Ae, be, S):
    """
    Computes the Minkowski difference of a polytope and another set or point using support functions
    
    Args:
        A: inequality constraint matrix
        b: inequality constraint offset
        Ae: equality constraint matrix
        be: equality constraint offset
        S: contSet object
        
    Returns:
        A: inequality constraint matrix
        b: inequality constraint offset
        Ae: equality constraint matrix
        be: equality constraint offset
        empty: true/false whether result is the empty set
    """
    # Assume non-empty result
    empty = False
    
    # Shift entry in offset vector by support function value of subtrahend
    for i in range(A.shape[0]):
        direction = A[i,:].reshape(-1, 1) if A[i,:].ndim == 1 else A[i,:]
        result_upper = S.supportFunc_(direction, 'upper')
        print(f"DEBUG (priv_minkdiff): supportFunc_('upper') result type: {type(result_upper)}, value: {result_upper}")
        l_val, _ = result_upper # Corrected unpacking
        
        if np.isinf(l_val):
            # subtrahend is unbounded in a direction where the minuend is
            # bounded -> result is empty
            A = np.array([]).reshape(0, n) if n > 0 else np.array([]).reshape(0,0)
            b = np.array([]).reshape(0, 1)
            Ae = np.array([]).reshape(0, n) if n > 0 else np.array([]).reshape(0,0)
            be = np.array([]).reshape(0, 1)
            empty = True
            return A, b, Ae, be, empty
        b[i, 0] = b[i, 0] - l_val

    # for equality constraints, we can easily detect if the resulting set
    # becomes empty, as the lower and upper bounds for the support function
    # in those directions must be equal
    Ae_rows = Ae.shape[0]
    for i in range(Ae_rows):
        # I = supportFunc_(S,Ae(i,:)','range');
        direction = Ae[i,:].reshape(-1, 1) if Ae[i,:].ndim == 1 else Ae[i,:]
        result_range = S.supportFunc_(direction, 'range')
        print(f"DEBUG (priv_minkdiff): supportFunc_('range') result type: {type(result_range)}, value: {result_range}")
        I_interval, _ = result_range # Corrected unpacking
        
        if not withinTol(I_interval.inf, I_interval.sup):
            A = np.array([]).reshape(0, n) if n > 0 else np.array([]).reshape(0,0)
            b = np.array([]).reshape(0, 1)
            Ae = np.array([]).reshape(0, n) if n > 0 else np.array([]).reshape(0,0)
            be = np.array([]).reshape(0, 1)
            empty = True
            return A, b, Ae, be, empty
        be[i, 0] = be[i, 0] - I_interval.inf

    print(f"DEBUG (priv_minkdiff): Final empty value: {empty}")
    return A, b, Ae, be, empty # Final return statement
