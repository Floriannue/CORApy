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
from typing import TYPE_CHECKING
from cora_python.contSet.interval.interval import Interval

if TYPE_CHECKING:
    from .polytope import Polytope

from .private.priv_box_V import priv_box_V
from .private.priv_box_H import priv_box_H


def interval(P: 'Polytope') -> Interval:
    """
    Encloses a polytope by an interval
    
    Args:
        P: Polytope object
        
    Returns:
        Interval object enclosing the polytope
    """
    
    # dimension
    n = P.dim()
    
    # obtain bounding box (MATLAB priv_box_* logic)
    if P.isVRep:
        # vertex representation
        A, b, empty = priv_box_V(P.V, n)
    else:
        # halfspace representation
        Ae = P.Ae if hasattr(P, 'Ae') else np.array([]).reshape(0, n)
        be = P.be if hasattr(P, 'be') else np.array([]).reshape(0, 1)
        A, b, empty = priv_box_H(P.A, P.b, Ae, be, n)
    
    # exit if already empty
    if empty:
        return Interval.empty(n)
    
    # A from priv_box is axis-aligned +/- ei normals stacked; b has corresponding bounds
    # Reconstruct lb/ub directly to avoid mis-indexing
    if A.shape[0] != 2*n:
        # Fallback: compute via support functions
        lb = np.zeros((n, 1)); ub = np.zeros((n, 1))
        I = np.eye(n)
        for i in range(n):
            ui = I[:, [i]]
            ub[i, 0] = P.supportFunc_(ui, 'upper')[0]
            lb[i, 0] = -P.supportFunc_(ui, 'lower')[0]
        return Interval(lb, ub)
    ub = b[:n].reshape(n, 1)
    lb = -b[n:].reshape(n, 1)
    return Interval(lb, ub)