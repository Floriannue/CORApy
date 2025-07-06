"""
min - computes the minimum of I and Y

Syntax:
    res = min(I, Y)

Inputs:
    I - interval object
    Y - interval or numeric 
    *args - additional parameters for built-in min function

Outputs:
    res - interval or numeric

Example: 
    I = interval([-2;-1],[2;1])
    res = min(I, 0)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: min, interval/infimum, interval/max

Authors: Tobias Ladner (MATLAB)
         Python translation by AI Assistant
Written: 16-December-2022 (MATLAB)
Last update: 11-April-2024 (TL, single input) (MATLAB)
Python translation: 2025
"""

import numpy as np
from .interval import Interval


def min(I: Interval, Y=None, *args):
    """
    Computes the minimum of I and Y
    
    Args:
        I: Interval object
        Y: Interval or numeric (optional)
        *args: Additional parameters for built-in min function
        
    Returns:
        Interval or numeric result
    """
    if Y is None:
        # return infimum (copy for consistency)
        return I.inf.copy()
    
    # Import max here to avoid circular imports
    from .max import max as max_func
    
    # Use the max function with negated inputs: min(a,b) = -max(-a,-b)
    if isinstance(Y, Interval):
        neg_Y = Interval(-Y.sup, -Y.inf)
    else:
        neg_Y = -Y
    
    neg_I = Interval(-I.sup, -I.inf)
    result = max_func(neg_I, neg_Y, *args)
    
    if isinstance(result, Interval):
        return Interval(-result.sup, -result.inf)
    else:
        return -result 