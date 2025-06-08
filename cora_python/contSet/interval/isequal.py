"""
isequal - check if two intervals are equal

Syntax:
    res = isequal(I1, I2)

Inputs:
    I1 - interval object
    I2 - interval object

Outputs:
    res - true if intervals are equal, false otherwise

Authors: Matthias Althoff (MATLAB)
         Python translation by AI Assistant
Written: 19-June-2015 (MATLAB)
Python translation: 2025
"""

import numpy as np


def isequal(obj1, obj2) -> bool:
    """
    Check if two intervals are equal
    
    Args:
        obj1: First interval object
        obj2: Second interval object
        
    Returns:
        True if intervals are equal, False otherwise
    """
    # Import here to avoid circular imports
    from .interval import Interval
    
    if not isinstance(obj2, Interval) :
        return False
    
    # Check if both are empty
    if obj1.inf.size == 0 and obj2.inf.size == 0:
        return True
    
    # Check if one is empty and the other is not
    if obj1.inf.size == 0 or obj2.inf.size == 0:
        return False
    
    # Check if bounds are equal
    return (np.allclose(obj1.inf, obj2.inf, rtol=1e-12, atol=1e-12) and
            np.allclose(obj1.sup, obj2.sup, rtol=1e-12, atol=1e-12)) 
