"""
length - Overloads the operator that returns the length
    of the longest array dimension

Syntax:
    l = length(I)

Inputs:
    I - interval object 

Outputs:
    l - length of the largest array dimension

Example: 
    I = interval([[-1, 1]], [[1, 2]])
    length(I)

Authors: Matthias Althoff (MATLAB)
         Python translation by AI Assistant
"""

import numpy as np


def length(I):
    """
    Overloads the operator that returns the length
    of the longest array dimension
    
    Args:
        I: interval object
        
    Returns:
        l: length of the largest array dimension
    """
    
    # Length of infimum and supremum are equal
    # In Python, we use np.max(shape) to get longest dimension
    inf_shape = np.array(I.inf.shape)
    return int(np.max(inf_shape)) if inf_shape.size > 0 else 0 