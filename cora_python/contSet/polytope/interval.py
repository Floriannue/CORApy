"""
interval - encloses a polytope by an interval

Syntax:
    I = interval(P)

Inputs:
    P - polytope object

Outputs:
    I - interval 

Example:
    A = np.array([[1, 2], [-1, 1], [-1, -3], [2, -1]])
    b = np.ones((4, 1))
    P = Polytope(A, b)
    
    I = interval(P)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors: Matthias Althoff, Viktor Kotsev, Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 01-February-2011 (MATLAB)
Last update: 30-July-2016 (MATLAB)
             31-May-2022 (MATLAB)
             14-December-2022 (MW, simplification) (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import Tuple


def interval(P):
    """
    Encloses a polytope by an interval
    
    Args:
        P: Polytope object
        
    Returns:
        Interval object enclosing the polytope
    """
    from cora_python.contSet.interval.interval import Interval
    from .private.priv_box_V import priv_box_V
    from .private.priv_box_H import priv_box_H
    
    # dimension
    n = P.dim()
    
    # obtain bounding box in halfspace representation
    if hasattr(P, 'isVRep') and P.isVRep:
        # vertex representation
        A, b, empty = priv_box_V(P.V, n)
    else:
        # halfspace representation
        Ae = getattr(P, 'Ae', np.array([]).reshape(0, n))
        be = getattr(P, 'be', np.array([]).reshape(0, 1))
        A, b, empty = priv_box_H(P.A, P.b, Ae, be, n)
    
    # exit if already empty
    if empty:
        return Interval.empty(n)
    
    # init lower and upper bounds of resulting interval with Inf values
    lb = np.full((n, 1), -np.inf)
    ub = np.full((n, 1), np.inf)
    
    # indices of constraints for upper bounds (indices of constraints for lower
    # bounds are given by the logical opposite)
    idx_ub = np.any(A > 0, axis=1)
    nnz_ub = np.sum(idx_ub)
    
    # upper bounds that are non-Inf
    idx_nonInf = np.any(A[idx_ub, :], axis=0)
    
    # overwrite bounds using b
    if nnz_ub > 0:
        ub[idx_nonInf] = b[:nnz_ub].reshape(-1, 1)
    
    # lower bounds that are non-(-Inf)
    idx_nonInf_lb = np.any(A[~idx_ub, :], axis=0)
    
    # overwrite bounds using b
    if np.sum(~idx_ub) > 0:
        lb[idx_nonInf_lb] = -b[nnz_ub:].reshape(-1, 1)
    
    # instantiate resulting interval
    return Interval(lb, ub) 