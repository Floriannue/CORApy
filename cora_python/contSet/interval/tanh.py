"""
tanh - Overloaded 'tanh()' operator for intervals

Syntax:
    res = tanh(I)

Inputs:
    I - interval object

Outputs:
    res - interval object

Example: 
    I = Interval(-0.5, 1.2)
    I.tanh()

Authors: Dmitry Grebenyuk (MATLAB)
         Python translation by AI Assistant
Written: 06-February-2016 (MATLAB)
Python translation: 2025
"""

import numpy as np
from .interval import Interval


def tanh(I):
    """
    Overloaded hyperbolic tangent operator for intervals
    
    Args:
        I: Interval object
        
    Returns:
        Interval object result of hyperbolic tangent operation
    """
    
    # tanh is monotonically increasing
    res_inf = np.tanh(I.inf)
    res_sup = np.tanh(I.sup)
    
    return Interval(res_inf, res_sup) 