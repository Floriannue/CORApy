"""
size - Overloads the operator that returns the size of the object

Syntax:
    s = size(I)
    s = size(I, dim)

Inputs:
    I - interval object
    dim - dimension index (optional)

Outputs:
    s - size of the interval (tuple or int)

Example: 
    I = interval([[-1, 1]], [[1, 2]])
    size(I)

Authors: Matthias Althoff (MATLAB)
         Python translation by AI Assistant
"""

import numpy as np


def size(I, dim=None):
    """
    Overloads the operator that returns the size of the object
    
    Args:
        I: interval object
        dim: dimension index (optional)
        
    Returns:
        s: size of the interval (tuple or int if dim specified)
    """
    
    # Return size of infimum (supremum has same size)
    if dim is not None:
        return I.inf.shape[dim] if dim < len(I.inf.shape) else 1
    else:
        return I.inf.shape 