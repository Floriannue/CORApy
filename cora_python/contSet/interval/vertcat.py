"""
vertcat - Overloads the operator for vertical concatenation

Syntax:
    I = vertcat(*args)

Inputs:
    *args - list of interval objects 

Outputs:
    I - interval object 

Example: 
    I1 = interval(-1, 1)
    I2 = interval(1, 2)
    I = vertcat(I1, I2)  # equivalent to [I1; I2] in MATLAB

Authors: Matthias Althoff (MATLAB)
         Python translation by AI Assistant
"""

import numpy as np
from .interval import Interval


def vertcat(*args):
    """
    Overloads the operator for vertical concatenation
    
    Args:
        *args: list of interval objects or numeric values
        
    Returns:
        I: interval object
    """
    
    if len(args) == 0:
        raise ValueError("At least one argument required for concatenation")
    
    # Process the first argument
    first_arg = args[0]
    if isinstance(first_arg, Interval):
        # Start with a copy of the first interval
        res_inf = np.copy(first_arg.inf)
        res_sup = np.copy(first_arg.sup)
    else:
        # If the first argument is numeric, create a new interval
        res_inf = np.asarray(first_arg)
        res_sup = np.asarray(first_arg)

    # Ensure starting arrays are at least 2D for concatenation
    if res_inf.ndim < 2:
        res_inf = res_inf.reshape(-1, 1)
        res_sup = res_sup.reshape(-1, 1)

    # Concatenate with remaining arguments
    for i in range(1, len(args)):
        arg = args[i]
        
        if isinstance(arg, Interval):
            inf_to_add = arg.inf
            sup_to_add = arg.sup
        else:
            inf_to_add = np.asarray(arg)
            sup_to_add = np.asarray(arg)

        # Ensure arrays to be added are at least 2D
        if inf_to_add.ndim < 2:
            inf_to_add = inf_to_add.reshape(-1, 1)
            sup_to_add = sup_to_add.reshape(-1, 1)

        res_inf = np.vstack([res_inf, inf_to_add])
        res_sup = np.vstack([res_sup, sup_to_add])
    
    return Interval(res_inf, res_sup) 