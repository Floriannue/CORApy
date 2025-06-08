"""
representsa_ - check if an interval represents a specific set type

Syntax:
    res = representsa_(I, type, tol)

Inputs:
    I - interval object
    type - string specifying the set type
    tol - tolerance for comparison

Outputs:
    res - true if interval represents the specified type, false otherwise

Authors: Matthias Althoff (MATLAB)
         Python translation by AI Assistant
Written: 19-June-2015 (MATLAB)
Python translation: 2025
"""

import numpy as np


from .aux_functions import _within_tol


def representsa_(obj, set_type: str, tol: float = 1e-9) -> bool:
    """
    Check if interval represents a specific set type
    
    Args:
        obj: Interval object
        set_type: Type of set to check ('emptySet', 'origin', 'point')
        tol: Tolerance for comparison
        
    Returns:
        True if interval represents the specified type
    """
    if set_type == 'emptySet':
        return obj.inf.size == 0
    elif set_type == 'origin':
        if obj.inf.size == 0:
            return False
        return (np.allclose(obj.inf, 0, atol=tol) and 
                np.allclose(obj.sup, 0, atol=tol))
    elif set_type == 'point':
        if obj.inf.size == 0:
            return False
        return np.allclose(obj.inf, obj.sup, atol=tol)
    else:
        return False 
