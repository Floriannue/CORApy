"""
norm_ - computes maximum norm value

Syntax:
    val = norm_(Z, type, mode)
    val, x = norm_(Z, type, mode)

Inputs:
    Z - zonotope object
    type - (optional) which kind of norm (default: 2)
    mode - (optional) 'exact', 'ub' (upper bound), 'ub_convex' (more
            precise upper bound computed from a convex program)

Outputs:
    val - norm value
    x - vertex attaining maximum norm (only for 'exact' mode)

Example: 
    Z = zonotope([1;0],[1 3 -2; 2 -1 0])
    norm_(Z)

Authors:       Victor Gassmann (MATLAB)
               Python translation by AI Assistant
Written:       31-July-2020 (MATLAB)
Last update:   ---
Last revision: 27-March-2023 (MW, rename norm_)
"""

import numpy as np
from typing import Union, Tuple, Optional
import warnings

try:
    from scipy.optimize import linprog
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from .private.priv_norm_exact import priv_norm_exact
from .private.priv_norm_ub import priv_norm_ub

def norm_(Z, norm_type: int = 2, mode: str = 'ub', return_vertex: bool = False):
    """
    Computes maximum norm value of a zonotope.
    
    Args:
        Z: zonotope object
        norm_type: which kind of norm (default: 2)
        mode: 'exact', 'ub' (upper bound), 'ub_convex' (more precise upper bound)
        return_vertex: if True, also return the vertex attaining maximum norm
        
    Returns:
        float or tuple: norm value, optionally with vertex
    """
    
    if norm_type != 2:
        raise ValueError("Only Euclidean norm (type=2) is currently supported")
    
    if mode == 'exact':
        if return_vertex:
            return _norm_exact(Z, norm_type)
        else:
            val, _ = _norm_exact(Z, norm_type)
            return val
    elif mode == 'ub':
        # Use interval overapproximation
        I = Z.interval()
        return _interval_norm(I, norm_type)
    elif mode == 'ub_convex':
        return _norm_ub_convex(Z, norm_type)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def _norm_exact(Z, norm_type: int):
    """
    Computes the exact maximum norm using private function.
    
    Args:
        Z: zonotope object
        norm_type: norm type (only 2 supported)
        
    Returns:
        tuple: (val, x) where val is norm value and x is vertex
    """
    
    return priv_norm_exact(Z, norm_type)





def _norm_ub_convex(Z, norm_type: int):
    """
    Compute upper bound using convex optimization via private function.
    """

    return priv_norm_ub(Z, norm_type)


def _interval_norm(I, norm_type: int):
    """
    Compute norm of an interval.
    """
    if norm_type != 2:
        raise ValueError("Only Euclidean norm supported")
    
    # For interval [a, b], the maximum norm is achieved at one of the corners
    # For 2-norm, we need to check which corner gives maximum distance from origin
    
    inf_vals = I.inf.flatten()
    sup_vals = I.sup.flatten()
    
    # Check all corners (2^n combinations)
    n = len(inf_vals)
    max_norm = 0
    
    if n <= 20:  # Avoid exponential explosion
        for i in range(2**n):
            corner = np.zeros(n)
            temp = i
            for j in range(n):
                corner[j] = sup_vals[j] if (temp % 2) == 1 else inf_vals[j]
                temp //= 2
            
            norm_val = np.linalg.norm(corner)
            max_norm = max(max_norm, norm_val)
    else:
        # For high dimensions, use heuristic
        # Maximum norm is likely at corner where each coordinate has maximum absolute value
        corner = np.where(np.abs(inf_vals) > np.abs(sup_vals), inf_vals, sup_vals)
        max_norm = np.linalg.norm(corner)
    
    return max_norm 