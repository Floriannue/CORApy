"""
Inf - instantiate a fullspace interval

Syntax:
    I = interval.Inf(n)

Inputs:
    n - dimension of the fullspace interval

Outputs:
    I - fullspace interval object

Authors: Matthias Althoff (MATLAB)
         Python translation by AI Assistant
Written: 19-June-2015 (MATLAB)
Python translation: 2025
"""

import numpy as np


def Inf(n: int):
    """
    Instantiate a fullspace interval
    
    Args:
        n: Dimension of the fullspace interval
        
    Returns:
        Fullspace interval object
    """
    # Import here to avoid circular imports
    from .interval import Interval
    
    inf = np.full(n, -np.inf)
    sup = np.full(n, np.inf)
    
    return Interval(inf, sup) 
