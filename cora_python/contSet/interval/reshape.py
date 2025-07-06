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
    
    # Handle empty intervals
    if I.isemptyobject():
        # Empty intervals have 0 elements, so they can only be reshaped to other shapes with 0 elements
        new_size = np.prod(new_shape)
        if new_size != 0:
            raise ValueError(f"Cannot reshape empty interval (0 elements) to shape {new_shape} ({new_size} elements)")
        
        # Create new empty interval with the requested shape that has 0 elements
        # This means at least one dimension must be 0
        if len(new_shape) == 1:
            res_inf = np.empty((new_shape[0], 0))
            res_sup = np.empty((new_shape[0], 0))
        else:
            # For multi-dimensional shapes, make the last dimension 0
            shape_with_zero = new_shape[:-1] + (0,)
            res_inf = np.empty(shape_with_zero)
            res_sup = np.empty(shape_with_zero)
        
        return Interval(res_inf, res_sup)
    
    # Reshape the bounds using default (row-major) ordering for consistency with rest of codebase
    res_inf = I.inf.reshape(new_shape)
    res_sup = I.sup.reshape(new_shape)
    
    return Interval(res_inf, res_sup) 