"""
sinh - Overloaded 'sinh()' operator for intervals

Syntax:
    res = sinh(I)

Inputs:
    I - interval object

Outputs:
    res - interval object

Example: 
    I = Interval(-0.5, 1.2)
    I.sinh()

Authors: Dmitry Grebenyuk (MATLAB)
         Python translation by AI Assistant
Written: 06-February-2016 (MATLAB)
Python translation: 2025
"""

import numpy as np

from .interval import Interval


def sinh(I):
    """
    Overloaded hyperbolic sine operator for intervals
    
    Args:
        I: Interval object
        
    Returns:
        Interval object result of hyperbolic sine operation
    """
    
    # sinh is monotonically increasing
    res_inf = np.sinh(I.inf)
    res_sup = np.sinh(I.sup)
    
    return Interval(res_inf, res_sup) 