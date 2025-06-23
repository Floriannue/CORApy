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
from .interval import Interval


def isequal(I1:Interval, I2:Interval, tol: float = 1e-12) -> bool:
    """
    Check if two intervals are equal
    
    Args:
        I1: First interval object
        I2: Second interval object
        tol: Tolerance for comparison (default: 1e-12)
        
    Returns:
        True if intervals are equal, False otherwise
    """
    
    if not isinstance(I2, Interval) :
        return False
    
    # Check if both are empty
    if I1.inf.size == 0 and I2.inf.size == 0:
        # Empty intervals are equal only if they have the same dimensions
        return I1.inf.shape == I2.inf.shape
    
    # Check if one is empty and the other is not
    if I1.inf.size == 0 or I2.inf.size == 0:
        return False
    
    # Check if shapes are different
    if I1.inf.shape != I2.inf.shape or I1.sup.shape != I2.sup.shape:
        return False
    
    # Check if bounds are equal with given tolerance
    return (np.allclose(I1.inf, I2.inf, rtol=tol, atol=tol) and
            np.allclose(I1.sup, I2.sup, rtol=tol, atol=tol)) 
