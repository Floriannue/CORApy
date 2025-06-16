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


def isequal(obj1, obj2, tol: float = 1e-12) -> bool:
    """
    Check if two intervals are equal
    
    Args:
        obj1: First interval object
        obj2: Second interval object
        tol: Tolerance for comparison (default: 1e-12)
        
    Returns:
        True if intervals are equal, False otherwise
    """
    # Import here to avoid circular imports
    from .interval import Interval
    
    if not isinstance(obj2, Interval) :
        return False
    
    # Check if both are empty
    if obj1.inf.size == 0 and obj2.inf.size == 0:
        # Empty intervals are equal only if they have the same dimensions
        return obj1.inf.shape == obj2.inf.shape
    
    # Check if one is empty and the other is not
    if obj1.inf.size == 0 or obj2.inf.size == 0:
        return False
    
    # Check if shapes are different
    if obj1.inf.shape != obj2.inf.shape or obj1.sup.shape != obj2.sup.shape:
        return False
    
    # Check if bounds are equal with given tolerance
    return (np.allclose(obj1.inf, obj2.inf, rtol=tol, atol=tol) and
            np.allclose(obj1.sup, obj2.sup, rtol=tol, atol=tol)) 
