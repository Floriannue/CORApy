"""
conversionStarSetConZono - convert a star set to a constrained zonotope
   zonotope in the given dimension

This is a Python translation of the MATLAB CORA implementation.

Authors: MATLAB: Niklas Kochdumper, Tobias Ladner
         Python: AI Assistant
"""

import numpy as np
from typing import Tuple


def conversionStarSetConZono(c: np.ndarray, G: np.ndarray, C: np.ndarray, 
                             d: np.ndarray, l_: np.ndarray, u_: np.ndarray):
    """
    Convert a star set to a constrained zonotope zonotope in the given dimension.
    
    Args:
        c: center of star set
        G: generator matrix of star set
        C: constraint matrix of star set
        d: constraint vector of star set
        l_: lower bounds of star set
        u_: upper bounds of star set
        
    Returns:
        cZ: constrained zonotope
        
    See also: -
    """
    # normalize halfspace normal vector length
    temp = np.sqrt(np.sum(C**2, axis=1, keepdims=True))
    C = np.diag(temp.flatten()) @ C
    d = temp * d
    
    numGens = G.shape[1]
    
    # find hypercube constraints
    u = np.zeros((numGens, 1))
    ind = np.where(np.any(C.T == 1, axis=0))[0]
    ind_ = np.setdiff1d(np.arange(C.shape[0]), ind)
    C_ = C[ind, :]
    d_ = d[ind]
    
    # compute upper bounds
    for i in range(numGens):
        ind_i = np.where(C_[:, i] == 1)[0]
        if len(ind_i) == 0:
            u[i] = u_[i]
        else:
            u[i] = np.min(d_[ind_i])
    
    # select constraints
    C = C[ind_, :]
    d = d[ind_]
    
    l = np.zeros((numGens, 1))
    ind = np.where(np.any(C.T == -1, axis=0))[0]
    ind_ = np.setdiff1d(np.arange(C.shape[0]), ind)
    C_ = C[ind, :]
    d_ = d[ind]
    
    # compute lower bounds
    for i in range(numGens):
        ind_i = np.where(C_[:, i] == -1)[0]
        if len(ind_i) == 0:
            l[i] = l_[i]
        else:
            l[i] = np.max(-d_[ind_i])
    
    C = C[ind_, :]
    d = d[ind_]
    
    # scale constraints according to hypercube dimensions
    from cora_python.contSet.interval import Interval
    from cora_python.contSet.zonotope import Zonotope
    from cora_python.contSet.conZonotope import ConZonotope
    
    int_val = Interval(l, u)
    cen = int_val.center()
    R = np.diag(int_val.rad())
    d = d - C @ cen
    C = C @ R
    c = c + G @ cen
    G = G @ R
    
    # represent inequality constraints as equivalent equality constraints
    if C.size > 0:
        Z = Zonotope(Interval(-np.ones((numGens, 1)), np.ones((numGens, 1))))
        l_int = Z.interval().inf
        int_val = Interval(l_int, d)
        cen = int_val.center()
        r = int_val.rad()
        A = np.hstack([C, np.diag(r.flatten())])
        b = cen
        G = np.hstack([G, np.zeros((G.shape[0], len(r)))])
    else:
        A = np.array([])
        b = np.array([])
    
    # construct resulting constrained zonotope
    cZ = ConZonotope(c, G, A, b)
    
    return cZ
