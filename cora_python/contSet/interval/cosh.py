"""
cosh - Overloaded 'cosh()' operator for intervals

Syntax:
    res = cosh(I)

Inputs:
    I - interval object

Outputs:
    res - interval object

Example: 
    I = Interval(-0.5, 1.2)
    I.cosh()

Authors: Dmitry Grebenyuk (MATLAB)
         Python translation by AI Assistant
Written: 06-February-2016 (MATLAB)
Python translation: 2025
"""

import numpy as np

from .interval import Interval


def cosh(I):
    """
    Overloaded hyperbolic cosine operator for intervals
    
    Args:
        I: Interval object
        
    Returns:
        Interval object result of hyperbolic cosine operation
    """
    
    # cosh has a minimum at x=0
    # If interval contains 0, minimum is cosh(0) = 1
    if np.any((I.inf <= 0) & (I.sup >= 0)):
        # Interval contains zero
        res_inf = np.ones_like(I.inf)
        # Maximum is at the endpoints
        res_sup = np.maximum(np.cosh(I.inf), np.cosh(I.sup))
    else:
        # Interval doesn't contain zero - cosh is monotonic on this interval
        res_inf = np.minimum(np.cosh(I.inf), np.cosh(I.sup))
        res_sup = np.maximum(np.cosh(I.inf), np.cosh(I.sup))
    
    return Interval(res_inf, res_sup) 