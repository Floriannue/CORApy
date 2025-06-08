"""
rad - returns the radius of an interval

Syntax:
    r = rad(I)

Inputs:
    I - interval object

Outputs:
    r - numerical value

Authors: Matthias Althoff (MATLAB)
         Python translation by AI Assistant
Written: 26-June-2015 (MATLAB)
Python translation: 2025
"""

import numpy as np
from .interval import Interval


def rad(I: Interval) -> np.ndarray:
    """
    Returns the radius of an interval
    
    Args:
        I: Interval object
        
    Returns:
        r: radius of interval (vector)
    """
    # Empty set check
    if I.representsa_('emptySet', 1e-9):
        return np.zeros((I.dim(), 0))
    
    r = 0.5 * (I.sup - I.inf)
    return r 
