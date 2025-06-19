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


from cora_python.g.functions.matlab.validate.check import withinTol


def representsa_(obj, set_type: str, tol: float = 1e-9, **kwargs):
    """
    Check if interval represents a specific set type
    
    Args:
        obj: Interval object
        set_type: Type of set to check ('emptySet', 'origin', 'point', 'fullspace', etc.)
        tol: Tolerance for comparison
        **kwargs: Additional arguments (e.g., return_set=True)
        
    Returns:
        bool or tuple: True if interval represents the specified type, 
                      or (bool, converted_set) if return_set=True
    """
    return_set = kwargs.get('return_set', False)
    
    if set_type == 'emptySet':
        res = obj.inf.size == 0
        if return_set:
            return res, obj if res else None
        return res
        
    elif set_type == 'origin':
        if obj.inf.size == 0:
            res = False
        else:
            res = (np.allclose(obj.inf, 0, atol=tol) and 
                   np.allclose(obj.sup, 0, atol=tol))
        if return_set:
            return res, np.zeros_like(obj.inf) if res else None
        return res
        
    elif set_type == 'point':
        if obj.inf.size == 0:
            res = False
        else:
            res = np.allclose(obj.inf, obj.sup, atol=tol)
        if return_set:
            return res, (obj.inf + obj.sup) / 2 if res else None
        return res
        
    elif set_type == 'fullspace':
        if obj.inf.size == 0:
            res = False
        else:
            res = np.all(obj.inf == -np.inf) and np.all(obj.sup == np.inf)
        if return_set:
            # Import here to avoid circular imports
            from cora_python.contSet.fullspace import Fullspace
            return res, Fullspace(obj.inf.shape[0]) if res else None
        return res
        
    elif set_type == 'interval':
        # Obviously true
        res = True
        if return_set:
            return res, obj
        return res
        
    else:
        res = False
        if return_set:
            return res, None
        return res 
