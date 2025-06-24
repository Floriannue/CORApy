"""
le - Overloads the <= operator, checks whether one interval is a subset
    or equal to another interval

Syntax:
    res = le(I1, I2)

Inputs:
    I1 - interval object
    I2 - interval object

Outputs:
    res - Boolean variable: True if I1 is subset or equal to I2

Example: 
    I1 = interval([1, -1], [2, 1])
    I2 = interval([1, -2], [2, 2])
    I1 <= I2

Authors: Matthias Althoff (MATLAB)
         Python translation by AI Assistant
"""

import numpy as np
from .interval import Interval


def le(I1, I2):
    """
    Overloads the <= operator, checks whether one interval is a subset
    or equal to another interval
    
    Args:
        I1: interval object
        I2: interval object
        
    Returns:
        res: Boolean - True if I1 is subset or equal to I2
    """
    
    # Convert to interval if needed
    if not hasattr(I2, 'inf'):
        I2 = Interval(I2, I2)
    
    # Handle empty intervals
    if I1.representsa_('emptySet', 1e-14):
        return True  # Empty set is subset of any set
    if I2.representsa_('emptySet', 1e-14):
        return I1.representsa_('emptySet', 1e-14)  # Only empty is subset of empty
    
    # Check if arrays are compatible for broadcasting
    if I1.inf.shape != I2.inf.shape:
        try:
            # Try to broadcast
            np.broadcast_arrays(I1.inf, I2.inf)
        except ValueError:
            return False
    
    # All left borders of I1 bigger than or equal to I2?
    leftResult = np.all(I1.inf >= I2.inf)
    
    # All right borders of I1 smaller than or equal to I2?
    rightResult = np.all(I1.sup <= I2.sup)
    
    # Both left and right interval tests must be true
    res = leftResult and rightResult
    
    return res 