"""
and_ - intersection of two intervals

Syntax:
    I_res = and_(I1, I2)
    I_res = and_(I1, I2, method)

Inputs:
    I1 - interval object
    I2 - interval object or other contSet
    method - method for intersection ('exact' by default)

Outputs:
    I_res - intersection of the intervals

Authors: Matthias Althoff (MATLAB)
         Python translation by AI Assistant
Written: 19-June-2015 (MATLAB)
Python translation: 2025
"""

import numpy as np

from .interval import Interval


def and_(I1: Interval, I2: Interval, method: str = 'exact') -> Interval:
    """
    Compute intersection of two intervals
    
    Args:
        I1: First interval object
        I2: Second interval object or other contSet
        method: Method for intersection (default: 'exact')
        
    Returns:
        Intersection of the intervals
    """
    
    # Convert I2 to interval if it's not already
    if not isinstance(I2, Interval) :
        if hasattr(I2, 'interval'):
            I2 = I2.Interval()
        else:
            raise ValueError("Cannot compute intersection: second argument is not convertible to interval")
    
    # Handle empty intervals
    if I1.inf.size == 0 or I2.inf.size == 0:
        return Interval.empty(max(I1.dim() if I1.inf.size > 0 else 0, 
                                 I2.dim() if I2.inf.size > 0 else 0))
    
    # Check dimension compatibility
    if I1.dim() != I2.dim():
        raise ValueError(f"Interval dimensions must match: {I1.dim()} vs {I2.dim()}")
    
    # Compute intersection bounds
    inf_result = np.maximum(I1.inf, I2.inf)
    sup_result = np.minimum(I1.sup, I2.sup)
    
    # Check if intersection is empty (any dimension has inf > sup)
    if np.any(inf_result > sup_result):
        return Interval.empty(I1.dim())
    
    return Interval(inf_result, sup_result) 
