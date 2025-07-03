from __future__ import annotations

"""asinh - Overloaded 'asinh()' operator for intervals"""

import numpy as np
from .interval import Interval

def asinh(i: Interval) -> Interval:
    """
    Overloaded 'asinh()' operator for intervals
    
    x_ is x infimum, x-- is x supremum
    
    [asinh(x_), asinh(x--)]
    
    Syntax:
        I = asinh(I)
    
    Inputs:
        I - interval object
    
    Outputs:
        I - interval object
    
    Example:
        i = Interval(np.array([-1]), np.array([1]))
        res = asinh(i)
    
    """
    
    # an empty interval remains empty
    if i.is_empty():
        return Interval.empty()
    
    inf = np.arcsinh(i.inf)
    sup = np.arcsinh(i.sup)
    
    return Interval(inf, sup) 