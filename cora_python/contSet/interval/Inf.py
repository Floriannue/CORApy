"""
Inf - instantiate a fullspace interval

Syntax:
    I = interval.Inf(n)

Inputs:
    n - dimension of the fullspace interval

Outputs:
    I - fullspace interval object

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 09-January-2024 (MATLAB)
Python translation: 2025
"""

import numpy as np
from .interval import Interval

def Inf(n: int = 0) -> Interval:
    """
    Instantiate a fullspace interval
    
    Args:
        n: Dimension of the fullspace interval (must be non-negative integer)
        
    Returns:
        Fullspace interval object
        
    Raises:
        ValueError: If n is not a non-negative integer
    """
    # Input validation
    if not isinstance(n, (int, np.integer)) or n < 0:
        raise ValueError("Dimension must be a non-negative integer")
    
    # Create 1D arrays for better Python/NumPy compatibility
    if n == 0:
        inf = np.array([])
        sup = np.array([])
    else:
        inf = np.full(n, -np.inf)
        sup = np.full(n, np.inf)
    
    return Interval(inf, sup) 
