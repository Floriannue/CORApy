"""
isemptyobject - check if an interval is empty

Syntax:
    res = isemptyobject(I)

Inputs:
    I - interval object

Outputs:
    res - true if interval is empty, false otherwise

Authors: Matthias Althoff (MATLAB)
         Python translation by AI Assistant
Written: 19-June-2015 (MATLAB)
Python translation: 2025
"""

import numpy as np


def isemptyobject(obj) -> bool:
    """
    Check if interval is empty
    
    Args:
        obj: Interval object
        
    Returns:
        True if interval is empty, False otherwise
    """
    return obj.inf.size == 0 
