"""
withinTol - checks whether two numeric values (scalars, vectors, arrays)
    are within a given tolerance

Syntax:
    res = withinTol(a,b)
    res = withinTol(a,b,tol)

Inputs:
    a,b - double (scalar, vector, matrix, n-d arrays)
    tol - (optional) tolerance

Outputs:
    res - true/false for each comparison

Example: 
    res = withinTol(1,1+1e-12)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Victor Gassmann (MATLAB)
               Python translation by AI Assistant
Written:       19-July-2021 (MATLAB)
Last update:   03-December-2023 (MW, handling of Inf values, MATLAB)
               22-April-2024 (LK, isnumeric check, MATLAB)
               18-October-2024 (TL, allow n-d arrays, MATLAB)
Python translation: 2025
"""

import numpy as np


def withinTol(a, b, tol=1e-8):
    """
    Checks whether two numeric values are within a given tolerance.
    
    Args:
        a: First value/array
        b: Second value/array  
        tol: (optional) tolerance (default: 1e-8)
        
    Returns:
        ndarray: Boolean array indicating where values are within tolerance
    """
    # Handle None values
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    
    # Convert to numpy arrays
    a = np.asarray(a)
    b = np.asarray(b)
    
    # Check inputs
    if not np.issubdtype(a.dtype, np.number):
        raise ValueError("First argument must be numeric")
    if not np.issubdtype(b.dtype, np.number):
        raise ValueError("Second argument must be numeric")
    if not np.isscalar(tol) or tol < 0:
        raise ValueError("Tolerance must be a non-negative scalar")
    
    # Absolute tolerance
    res_abs = np.abs(a - b) <= tol
    
    # Relative tolerance (avoid division by zero)
    min_abs = np.minimum(np.abs(a), np.abs(b))
    with np.errstate(divide='ignore', invalid='ignore'):
        res_rel = np.abs(a - b) / min_abs <= tol
    
    # Handle cases where min_abs is zero
    res_rel = np.where(min_abs == 0, False, res_rel)
    
    # Handling of Inf values
    res_inf = np.isinf(a) & np.isinf(b) & (np.sign(a) == np.sign(b))
    
    # Joint result
    res = res_abs | res_rel | res_inf
    
    return res 