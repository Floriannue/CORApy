"""
contains_ - check if an interval contains given points

Syntax:
    res = contains_(I, points)

Inputs:
    I - interval object
    points - points to check (numpy array)

Outputs:
    res - true if all points are contained, false otherwise

Authors: Matthias Althoff (MATLAB)
         Python translation by AI Assistant
Written: 19-June-2015 (MATLAB)
Python translation: 2025
"""

import numpy as np


def contains_(obj, point: np.ndarray) -> bool:
    """
    Check if interval contains given point(s)
    
    Args:
        obj: Interval object
        point: Point(s) to check
        
    Returns:
        True if all points are contained, False otherwise
    """
    point = np.asarray(point)
    
    if obj.inf.size == 0:
        return False
    
    # Check if point is within bounds
    return np.all(point >= obj.inf) and np.all(point <= obj.sup) 
