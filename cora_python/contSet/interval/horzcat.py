"""
horzcat - Overloads the operator for horizontal concatenation

Syntax:
    I = horzcat(*args)

Inputs:
    *args - list of interval objects 

Outputs:
    I - interval object 

Example: 
    I1 = interval(-1, 1)
    I2 = interval(1, 2)
    I = horzcat(I1, I2)  # equivalent to [I1, I2] in MATLAB

Authors: Matthias Althoff (MATLAB)
         Python translation by AI Assistant
"""

import numpy as np
from .interval import Interval


def horzcat(*args):
    """
    Overloads the operator for horizontal concatenation
    
    Args:
        *args: list of interval objects or numeric values
        
    Returns:
        I: interval object
    """
    
    if not args:
        return Interval.empty()

    infs = []
    sups = []

    for arg in args:
        if isinstance(arg, Interval):
            if arg.is_empty():
                continue
            inf_val = arg.inf
            sup_val = arg.sup
        else:
            inf_val = np.asarray(arg)
            sup_val = np.asarray(arg)

        # Ensure values are at least 1-D for np.atleast_2d to work as expected on scalars
        inf_val = np.atleast_1d(inf_val)
        sup_val = np.atleast_1d(sup_val)
        
        infs.append(np.atleast_2d(inf_val))
        sups.append(np.atleast_2d(sup_val))

    if not infs:
        return Interval.empty()

    final_inf = np.hstack(infs)
    final_sup = np.hstack(sups)
    
    return Interval(final_inf, final_sup) 