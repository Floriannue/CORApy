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
from typing import Union
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def withinTol(a: Union[float, int, np.ndarray], b: Union[float, int, np.ndarray], tol: float = 1e-8) -> Union[bool, np.ndarray]:
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

    Authors:       Victor Gassmann
    Written:       19-July-2021
    Last update:   03-December-2023 (MW, handling of Inf values)
                   22-April-2024 (LK, isnumeric check)
                   18-October-2024 (TL, allow n-d arrays)
    Last revision: 20-July-2023 (TL, speed up input parsing)
                   Automatic python translation: Florian Nüssel BA 2025
    """
    is_scalar_input = np.isscalar(a) and np.isscalar(b)

    # direct check for speed reasons - handle numpy scalars and arrays
    def is_numeric(x):
        return (isinstance(x, (int, float, np.integer, np.floating)) or 
                (isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.number)))
    
    if not is_numeric(a):
        raise CORAerror('CORA:wrongValue', 'first', 'double')
    elif not is_numeric(b):
        raise CORAerror('CORA:wrongValue', 'second', 'double')
    elif not np.isscalar(tol) or tol < 0:
        raise CORAerror('CORA:wrongValue', 'third', 'nonnegative scalar')
    
    a = np.asarray(a)
    b = np.asarray(b)

    # Check dimension
    aux_checkDims(a, b)

    # absolute tolerance
    res_abs = np.abs(a - b) <= tol

    # relative tolerance
    # Avoid division by zero for values close to zero
    with np.errstate(divide='ignore', invalid='ignore'):
        min_abs = np.minimum(np.abs(a), np.abs(b))
        res_rel = np.abs(a - b) / min_abs <= tol
        
        # Handle cases where min_abs is zero or result in NaN/Inf from division
        res_rel = np.where(min_abs == 0, False, res_rel) # If min_abs is 0, they are only equal if a==b
        res_rel[np.isnan(res_rel)] = False # Handle NaN for 0/0 cases, should already be False by np.where
        res_rel[np.isinf(res_rel)] = False # Handle Inf for X/0 cases

    # handling of Inf values
    res_inf = np.isinf(a) & np.isinf(b) & (np.sign(a) == np.sign(b))

    # joint result
    res = res_abs | res_rel | res_inf

    return res.item() if is_scalar_input else res

def aux_checkDims(a, b):
    # check scalar
    if np.isscalar(a) or np.isscalar(b):
        return

    size_a = np.array(a).shape
    size_b = np.array(b).shape

    # Normalize 0D (scalar) and 1D arrays to 2D for consistent comparison,
    # treating 1D arrays as row vectors like MATLAB often does for broadcasting.
    if len(size_a) == 0: # scalar
        size_a = (1, 1)
    elif len(size_a) == 1: # 1D array, treat as row vector
        size_a = (1, size_a[0])

    if len(size_b) == 0: # scalar
        size_b = (1, 1)
    elif len(size_b) == 1: # 1D array, treat as row vector
        size_b = (1, size_b[0])

    n = max(len(size_a), len(size_b)) # This n will now be at least 2 for non-scalars

    # Extend to match number of dimensions (already handled for 0D/1D above, this is for >2D)
    size_a_padded = list(size_a) + [1] * (n - len(size_a))
    size_b_padded = list(size_b) + [1] * (n - len(size_b))

    size_a_padded = np.array(size_a_padded)
    size_b_padded = np.array(size_b_padded)

    # mismatching dimensions must be scalar in either array
    idxMiss = (size_a_padded != size_b_padded)
    if not np.all((size_a_padded[idxMiss] == 1) | (size_b_padded[idxMiss] == 1)):
        raise CORAerror('CORA:dimensionMismatch', a, b) 