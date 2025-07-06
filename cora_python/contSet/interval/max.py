"""
max - computes the maximum of I and Y

Syntax:
    res = max(I, Y)

Inputs:
    I - interval object
    Y - interval or numeric 
    *args - additional parameters for built-in max function

Outputs:
    res - interval

Example: 
    I = interval([-2;-1],[2;1])
    res = max(I, 0)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: max, interval/supremum

Authors: Tobias Ladner (MATLAB)
         Python translation by AI Assistant
Written: 16-December-2022 (MATLAB)
Last update: 11-April-2024 (TL, single input) (MATLAB)
Python translation: 2025
"""

import numpy as np
from .interval import Interval


def max(I: Interval, Y=None, *args):
    """
    Computes the maximum of I and Y
    
    Args:
        I: Interval object
        Y: Interval or numeric (optional)
        *args: Additional parameters for built-in max function
        
    Returns:
        Interval or numeric result
    """
    if Y is None:
        # return supremum (copy for consistency)
        return I.sup.copy()
    
    if isinstance(Y, (int, float, np.number)) or np.isscalar(Y):
        # Handle numeric Y
        if np.isscalar(Y):
            Y_val = Y
        else:
            Y_val = np.array(Y)
        
        # Check dimensions for non-scalar Y
        if not np.isscalar(Y_val) and Y_val.size > 1:
            if I.inf.shape != Y_val.shape:
                raise ValueError('CORA:dimensionMismatch - Dimensions must match')
        
        return Interval(np.maximum(I.inf, Y_val, *args), 
                       np.maximum(I.sup, Y_val, *args))
    
    if not isinstance(Y, Interval):
        # Convert contSet to interval if needed
        if hasattr(Y, 'interval'):
            Y = Y.interval()
        else:
            raise ValueError("Y must be numeric or convertible to interval")
    
    return Interval(np.maximum(I.inf, Y.inf, *args),
                   np.maximum(I.sup, Y.sup, *args)) 