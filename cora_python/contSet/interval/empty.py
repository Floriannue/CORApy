"""
empty - instantiate an empty interval

Syntax:
    I = Interval.empty(n)

Inputs:
    n - dimension of the empty interval

Outputs:
    I - empty interval object

Authors: Matthias Althoff (MATLAB)
         Python translation by AI Assistant
Written: 19-June-2015 (MATLAB)
Python translation: 2025
"""

import numpy as np


def empty(n: int = 0):
    """
    Instantiate an empty interval
    
    Args:
        n: Dimension of the empty interval
        
    Returns:
        Empty interval object
    """
    # Import here to avoid circular imports
    from .interval import Interval
    
    # Create empty interval with proper dimensions
    # In MATLAB: Interval(zeros(n,0))
    empty_array = np.zeros((n, 0))
    return Interval(empty_array) 
