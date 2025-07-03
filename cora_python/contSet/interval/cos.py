"""
cos - Overloaded 'cos()' operator for intervals

inf is x infimum, sup is x supremum

[-1, 1]                       if (sup - inf) >= 2*pi,
[-1, 1]                       if (sup - inf) < 2*pi and inf <= pi and sup <= pi and sup < inf,
[cos(sup), cos(inf)]          if (sup - inf) < 2*pi and inf <= pi and sup <= pi and sup >= inf,
[-1, max(cos(inf),cos(sup))]  if (sup - inf) < 2*pi and inf <= pi and sup > pi,
[-1, 1]                       if (sup - inf) < 2*pi and inf > pi and sup > pi and sup < inf,
[min(cos(inf),cos(sup)), 1]   if (sup - inf) < 2*pi and inf > pi and sup <= pi,
[cos(inf), cos(sup)]          if (sup - inf) < 2*pi and inf > pi and sup > pi and sup >= inf.

Syntax:
    res = cos(I)

Inputs:
    I - interval object

Outputs:
    res - interval object

Example:
    I = interval([-2;3],[3;4])
    res = cos(I)

References:
    [1] M. Althoff, D. Grebenyuk, "Implementation of Interval Arithmetic
        in CORA 2016", ARCH'16.

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: mtimes

Authors: Matthias Althoff, Dmitry Grebenyuk, Adrian Kulmburg (MATLAB)
         Python translation by AI Assistant
Written: 25-June-2015 (MATLAB)
Last update: 06-January-2016 (DG) (MATLAB)
             05-February-2016 (MA) (MATLAB)
             22-February-2016 (DG, the matrix case is rewritten) (MATLAB)
             10-January-2024 (MW, fix condition for scalar case) (MATLAB)
Last revision: 18-January-2024 (MW, implement Adrian's fast algorithm) (MATLAB)
Python translation: 2025
"""

import numpy as np
from .interval import Interval


def cos(I: Interval) -> Interval:
    """
    Overloaded cos function for intervals
    
    Args:
        I: Interval object
        
    Returns:
        Interval object with cosine applied
    """
    # Handle empty intervals
    if I.is_empty():
        return Interval.empty(I.dim())
    
    # init resulting interval (avoid constructor call)
    # compute lower and upper bound using Adrian's fast algorithm
    inf_res = -_aux_maxcos(I - np.pi)
    sup_res = _aux_maxcos(I)
    
    # ensure that inf <= sup due to floating point inaccuracies
    inf_result = np.minimum(inf_res, sup_res)
    sup_result = np.maximum(inf_res, sup_res)
    
    return Interval(inf_result, sup_result)


def _aux_maxcos(I: Interval) -> np.ndarray:
    """
    Adrian's special function for efficient cosine computation
    
    Args:
        I: Interval object
        
    Returns:
        Maximum cosine values
    """
    # Handle intervals with infinite bounds
    # If the interval span is >= 2*pi or contains infinite bounds, return 1
    inf_bounds = I.inf
    sup_bounds = I.sup
    
    # Check for infinite bounds or spans >= 2*pi
    infinite_mask = np.isinf(inf_bounds) | np.isinf(sup_bounds) | ((sup_bounds - inf_bounds) >= 2 * np.pi)
    
    # Initialize result
    val = np.ones_like(inf_bounds)
    
    # For finite intervals with span < 2*pi, use Adrian's algorithm
    finite_mask = ~infinite_mask
    
    if np.any(finite_mask):
        # Apply algorithm only to finite elements
        k = np.ceil(inf_bounds / (2 * np.pi))
        
        a = inf_bounds - 2 * np.pi * k
        b = sup_bounds - 2 * np.pi * k
        
        # Compute cosine values only for finite elements
        M = np.maximum(np.cos(a), np.cos(b))
        val_finite = np.maximum(np.sign(b), M)
        
        # Update only the finite elements
        val[finite_mask] = val_finite[finite_mask]
    
    return val 