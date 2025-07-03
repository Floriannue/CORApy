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
    Overloads the '<'-operator.
    
    If both arguments are intervals, it checks whether I1 is a strict subset
    of I2.
    
    If one argument is an interval and the other is a numeric value, it
    checks if the interval is strictly to the left of the value.
    
    Args:
        I1: interval object or numeric
        I2: interval object or numeric
        
    Returns:
        res: true/false
    """
    
    # Case 1: Interval < Interval (Strict Subset)
    if isinstance(I1, Interval) and isinstance(I2, Interval):
        # Handle empty intervals
        if I1.is_empty():
            return not I2.is_empty() # Empty set is strict subset of any non-empty set
        if I2.is_empty():
            return False  # Nothing can be strict subset of empty set
        
        # Check if arrays are compatible for broadcasting
        try:
            np.broadcast_arrays(I1.inf, I2.inf)
        except ValueError:
            return False # Incompatible shapes
        
        # All infima of I1 bigger than I2? (strict inequality)
        leftResult = np.all(I1.inf > I2.inf)
        # All suprema of I1 smaller than I2? (strict inequality)
        rightResult = np.all(I1.sup < I2.sup)
        
        return leftResult and rightResult

    # Case 2: Interval < Scalar
    elif isinstance(I1, Interval):
        return np.all(I1.sup < I2)

    # Case 3: Scalar < Interval
    else: # I1 is numeric, I2 is Interval
        return np.all(I1 < I2.inf) 