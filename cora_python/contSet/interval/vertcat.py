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
    
    # Start with first argument
    I = args[0]
    
    # If object is not an interval, convert it
    if not isinstance(I, Interval):
        I = Interval(I, I)
    
    # Concatenate with remaining arguments
    for i in range(1, len(args)):
        arg = args[i]
        
        # Check if concatenated variable is an interval
        if isinstance(arg, Interval):
            I.inf = np.concatenate([I.inf, arg.inf], axis=0)
            I.sup = np.concatenate([I.sup, arg.sup], axis=0)
        else:
            # Convert to numpy array for concatenation
            arg_array = np.atleast_1d(arg)
            I.inf = np.concatenate([I.inf, arg_array], axis=0)
            I.sup = np.concatenate([I.sup, arg_array], axis=0)
    
    return I 