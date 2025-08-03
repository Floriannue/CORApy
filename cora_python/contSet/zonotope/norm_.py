"""
norm_ - computes maximum norm value

Syntax:
    val = norm_(Z, type, mode)

Inputs:
    Z - zonotope object
    type - (optional) which kind of norm (default: 2)
    mode - (optional) 'exact', 'ub' (upper bound),'ub_convex' (more
           precise upper bound computed from a convex program)

Outputs:
    val - norm value
    x - vertex attaining maximum norm

Example: 
    Z = Zonotope(np.array([[1], [0]]), np.array([[1, 3, -2], [2, -1, 0]]))
    norm_(Z)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: contSet/norm, minnorm

Authors:       Victor Gassmann (MATLAB)
               Python translation by AI Assistant
Written:       31-July-2020 (MATLAB)
Last update:   27-March-2023 (MW, rename norm_) (MATLAB)
                2025 (Tiange Yang, Florian Nüssel, Python translation by AI Assistant)
"""

import numpy as np

from .private.priv_norm_exact import priv_norm_exact
from .private.priv_norm_ub import priv_norm_ub

def norm_(Z, norm_type: int = 2, mode: str = 'ub', return_vertex: bool = False):
    """
    Computes maximum norm value of a zonotope.
    
    Args:
        Z: zonotope object
        norm_type: which kind of norm (default: 2)
        mode: 'exact', 'ub' (upper bound), 'ub_convex' (more precise upper bound)
        return_vertex: if True, also return the vertex attaining maximum norm (only for 'exact' mode)
        
    Returns:
        float or tuple: norm value, optionally with vertex (only for 'exact' mode)
    """
    
    # Handle empty zonotope
    if hasattr(Z, 'isemptyobject') and Z.isemptyobject():
        if return_vertex:
            return -np.inf, np.array([])
        else:
            return -np.inf
    
    if norm_type != 2:
        raise ValueError("Only Euclidean norm (type=2) is currently supported")
    
    # Match MATLAB structure exactly
    if mode == 'exact':
        result = priv_norm_exact(Z, norm_type)
        if return_vertex:
            return result  # Return (val, x) tuple
        else:
            return result[0]  # Return only the value
    elif mode == 'ub':
        # Use interval overapproximation - match MATLAB: norm_(interval(Z),type)
        I = Z.interval()
        return I.norm_(norm_type)
    elif mode == 'ub_convex':
        return priv_norm_ub(Z, norm_type)
    else:
        raise ValueError(f"Unknown mode: {mode}")





 