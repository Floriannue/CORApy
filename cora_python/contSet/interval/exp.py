"""
exp - Overloaded 'exp()' operator for intervals

x_ is x infimum, x-- is x supremum

[exp(x_), exp(x--)].

Syntax:
    I = exp(I)

Inputs:
    I - interval object

Outputs:
    I - interval object

Example:
    I = interval(-2,4)
    exp(I)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: interval

Authors: Matthias Althoff (MATLAB)
         Python translation by AI Assistant
Written: 25-June-2015 (MATLAB)
Last update: 18-January-2024 (MW, avoid constructor call) (MATLAB)
Python translation: 2025
"""

import numpy as np
from .interval import Interval


def exp(I: Interval) -> Interval:
    """
    Overloaded exp operator for intervals
    
    Args:
        I: Interval object
        
    Returns:
        Interval object with exponential applied
    """
    # Handle empty intervals
    if I.is_empty():
        return Interval.empty(I.dim())
    
    # Exponential function is monotonic -> apply exp to infima/suprema
    inf_exp = np.exp(I.inf)
    sup_exp = np.exp(I.sup)
    
    return Interval(inf_exp, sup_exp) 