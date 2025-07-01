"""
reshape - Overloaded 'reshape()' operator for intervals

Syntax:
    res = reshape(I, m, n)
    res = reshape(I, [m, n])

Inputs:
    I - interval object
    m - number of rows in reshaped interval
    n - number of columns in reshaped interval

Outputs:
    res - interval object

Example: 
    I = Interval(ones(4,1), 2*ones(4,1))
    I.reshape(2, 2)

Authors: Matthias Althoff (MATLAB)
         Python translation by AI Assistant
Written: 21-June-2015 (MATLAB)
Python translation: 2025
"""

import numpy as np
from .interval import Interval


def reshape(I, *args):
    """
    Overloaded reshape operator for intervals
    
    Args:
        I: Interval object
        *args: Integers defining the reshaping
        
    Returns:
        Interval object with reshaped bounds
    """
    
    # Parse arguments
    if len(args) == 1:
        if isinstance(args[0], (list, tuple, np.ndarray)):
            new_shape = tuple(args[0])
        else:
            new_shape = (args[0],)
    else:
        new_shape = args
    
    # Reshape the bounds
    res_inf = I.inf.reshape(new_shape)
    res_sup = I.sup.reshape(new_shape)
    
    return Interval(res_inf, res_sup) 