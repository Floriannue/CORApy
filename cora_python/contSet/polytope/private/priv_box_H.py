"""
priv_box_H - computes the halfspace representation of the box enclosure
    given a halfspace representation

Syntax:
    A, b, empty, fullDim, bounded = priv_box_H(A, b, Ae, be, n)

Inputs:
    A - inequality constraint matrix
    b - inequality constraint offset
    Ae - equality constraint matrix
    be - equality constraint offset
    n - dimension of polytope

Outputs:
    A - inequality constraint matrix
    b - inequality constraint offset
    empty - true/false whether result is the empty set
    fullDim - true/false on degeneracy
    bounded - true/false on boundedness

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 03-October-2024 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import Tuple, Optional
from cora_python.g.functions.matlab.init import unitvector
from .priv_supportFunc import priv_supportFunc
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol


def priv_box_H(A: np.ndarray, b: np.ndarray, Ae: np.ndarray, be: np.ndarray, 
               n: int) -> Tuple[np.ndarray, np.ndarray, bool, bool, bool]:
    """
    priv_box_H - computes the halfspace representation of the box enclosure
    given a halfspace representation
    
    Syntax:
        A, b, empty, fullDim, bounded = priv_box_H(A, b, Ae, be, n)
    
    Inputs:
        A - inequality constraint matrix
        b - inequality constraint offset
        Ae - equality constraint matrix
        be - equality constraint offset
        n - dimension of polytope
    
    Outputs:
        A - inequality constraint matrix
        b - inequality constraint offset
        empty - true/false whether result is the empty set
        fullDim - true/false on degeneracy
        bounded - true/false on boundedness
    
    Authors: Mark Wetzlinger (MATLAB)
             Automatic python translation: Florian Nüssel BA 2025
    Written: 03-October-2024 (MATLAB)
    Python translation: 2025
    """
    
    # init bounds
    ub = np.full((n, 1), np.inf)
    lb = np.full((n, 1), -np.inf)
    
    # Initialize outputs for early return
    empty_out = False
    fullDim_out = False
    bounded_out = False
    A_out = np.array([]).reshape(0, n) # Default empty A with correct dimension
    b_out = np.array([]).reshape(0, 1) # Default empty b with correct dimension

    # loop over all 2n positive/negative basis vectors
    for i in range(n):
        # i-th basis vector (using 0-based indexing)
        e_i = unitvector(i + 1, n)  # unitvector uses 1-based indexing
        
        # maximize
        ub_val, _ = priv_supportFunc(A, b, Ae, be, e_i, 'upper')
        ub[i, 0] = ub_val
        if ub[i, 0] == -np.inf: # If upper bound is -inf, set is empty
            empty_out = True
            bounded_out = True
            fullDim_out = False # If empty, not full dim
            return A_out, b_out, empty_out, fullDim_out, bounded_out
        
        # minimize
        lb_val, _ = priv_supportFunc(A, b, Ae, be, e_i, 'lower')
        lb[i, 0] = lb_val
        if lb[i, 0] == np.inf: # If lower bound is +inf, set is empty
            empty_out = True
            bounded_out = True
            fullDim_out = False # If empty, not full dim
            return A_out, b_out, empty_out, fullDim_out, bounded_out
    
    # construct output arguments
    A_out = np.vstack([np.eye(n), -np.eye(n)])
    b_out = np.vstack([ub, -lb])
    
    # emptiness, boundedness, and degeneracy
    empty_out = False # If we reach here, it's not empty based on bounds
    bounded_out = ~np.any(np.isinf(b_out)) # Check if any bound is infinity
    fullDim_out = ~np.any(withinTol(lb, ub, 1e-10)) # Check if any dimension is degenerate (lb == ub)

    return A_out, b_out, empty_out, fullDim_out, bounded_out 