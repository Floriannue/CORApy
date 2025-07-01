"""
atan - Overloaded 'atan()' operator for intervals

Syntax:
    res = atan(I)

Inputs:
    I - interval object

Outputs:
    res - interval object

Example: 
    I = Interval(-1.5, 2.3)
    I.atan()

Authors: Dmitry Grebenyuk (MATLAB)
         Python translation by AI Assistant
Written: 06-February-2016 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import TYPE_CHECKING

from .interval import Interval


def atan(I):
    """
    Overloaded inverse tangent operator for intervals
    
    Args:
        I: Interval object
        
    Returns:
        Interval object result of inverse tangent operation
    """
    
    # atan is monotonically increasing
    res_inf = np.arctan(I.inf)
    res_sup = np.arctan(I.sup)
    
    return Interval(res_inf, res_sup) 