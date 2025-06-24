"""
lt - Overloads the '<'-operator, checks whether one interval is the
    subset of another interval

Syntax:
    res = lt(I1, I2)

Inputs:
    I1 - interval object
    I2 - interval object

Outputs:
    res - true/false

Example: 
    I1 = interval([1, -1], [2, 1])
    I2 = interval([1, -2], [2, 2])
    I1 < I2

Authors: Matthias Althoff (MATLAB)
         Python translation by AI Assistant
"""

import numpy as np
from .interval import Interval


def lt(I1, I2):
    """
    Overloads the '<'-operator, checks whether one interval is the
    subset of another interval
    
    Args:
        I1: interval object
        I2: interval object
        
    Returns:
        res: true/false
    """
    
    # Convert to interval if needed
    if not hasattr(I2, 'inf'):
        I2 = Interval(I2, I2)
    
    # Handle empty intervals
    if I1.representsa_('emptySet', 1e-14):
        return True  # Empty set is strict subset of any non-empty set
    if I2.representsa_('emptySet', 1e-14):
        return False  # Nothing can be strict subset of empty set
    
    # Check if arrays are compatible for broadcasting
    if I1.inf.shape != I2.inf.shape:
        try:
            # Try to broadcast
            np.broadcast_arrays(I1.inf, I2.inf)
        except ValueError:
            return False
    
    # All infima of I1 bigger than I2? (strict inequality)
    leftResult = np.all(I1.inf > I2.inf)
    
    # All suprema of I1 smaller than I2? (strict inequality)
    rightResult = np.all(I1.sup < I2.sup)
    
    # Both tests must be true
    res = leftResult and rightResult
    
    return res 