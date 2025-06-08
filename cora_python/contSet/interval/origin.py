"""
origin - instantiate an interval representing the origin

Syntax:
    I = interval.origin(n)

Inputs:
    n - dimension of the origin interval

Outputs:
    I - interval object representing the origin in R^n

Authors: Matthias Althoff (MATLAB)
         Python translation by AI Assistant
Written: 19-June-2015 (MATLAB)
Python translation: 2025
"""

import numpy as np


def origin(n: int):
    """
    Instantiate an interval representing the origin in R^n
    
    Args:
        n: Dimension of the origin interval
        
    Returns:
        Interval object representing the origin
    """
    # Import here to avoid circular imports
    from .interval import interval
    
    zeros = np.zeros(n)
    return interval(zeros, zeros) 