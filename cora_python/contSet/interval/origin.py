"""
origin - instantiate an interval representing the origin

Syntax:
    I = interval.origin(n)

Inputs:
    n - dimension of the origin interval

Outputs:
    I - interval object representing the origin in R^n

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 21-September-2024 (MATLAB)
Python translation: 2025
"""

import numpy as np


def origin(n: int):
    """
    Instantiate an interval representing the origin in R^n
    
    Args:
        n: Dimension of the origin interval (must be positive integer)
        
    Returns:
        Interval object representing the origin
        
    Raises:
        ValueError: If n is not a positive integer
    """
    # Input validation
    if not isinstance(n, (int, np.integer)) or n <= 0:
        raise ValueError("Dimension must be a positive integer")
    
    # Import here to avoid circular imports
    from .interval import Interval
    
    # Create 1D array of zeros for better Python/NumPy compatibility
    zeros = np.zeros(n)
    return Interval(zeros, zeros) 
