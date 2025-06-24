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
    
    if len(args) == 0:
        raise ValueError("At least one argument required for concatenation")
    
    # Start with first argument
    I = args[0]
    
    # If object is not an interval, convert it
    if not isinstance(I, Interval):
        I = Interval(I, I)
    
    # Ensure arrays are at least 2D for horizontal concatenation
    if I.inf.ndim == 1:
        I.inf = I.inf.reshape(-1, 1)
        I.sup = I.sup.reshape(-1, 1)
    
    # Concatenate with remaining arguments
    for i in range(1, len(args)):
        arg = args[i]
        
        # Check if concatenated variable is an interval
        if isinstance(arg, Interval):
            arg_inf = arg.inf
            arg_sup = arg.sup
            
            # Ensure arg arrays are at least 2D
            if arg_inf.ndim == 1:
                arg_inf = arg_inf.reshape(-1, 1)
                arg_sup = arg_sup.reshape(-1, 1)
                
            I.inf = np.concatenate([I.inf, arg_inf], axis=1)
            I.sup = np.concatenate([I.sup, arg_sup], axis=1)
        else:
            # Convert to numpy array for concatenation
            arg_array = np.atleast_1d(arg)
            if arg_array.ndim == 1:
                arg_array = arg_array.reshape(-1, 1)
            I.inf = np.concatenate([I.inf, arg_array], axis=1)
            I.sup = np.concatenate([I.sup, arg_array], axis=1)
    
    return I 