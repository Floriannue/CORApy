from __future__ import annotations
"""atanh - Overloaded 'atanh()' operator for intervals"""

from typing import TYPE_CHECKING
import numpy as np
from .interval import Interval
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

if TYPE_CHECKING:
    from .interval import Interval

def atanh(i: "Interval") -> "Interval":
    """
    Overloaded 'atanh()' operator for intervals.
    
    The domain of atanh is [-1, 1]. This function will raise an error if any
    part of the interval lies outside this domain.
    
    Syntax:
        I = atanh(I)
    
    Inputs:
        i - interval object
    
    Outputs:
        Interval object representing the atanh of the input.
    
    Example:
        i = Interval(np.array([-0.5]), np.array([0.3]))
        res = atanh(i)
    """
    
    # an empty interval remains empty
    if i.is_empty():
        return Interval.empty()
    
    # Check if the interval is within the valid domain of atanh: [-1, 1]
    if np.any(i.inf < -1) or np.any(i.sup > 1):
        raise CORAerror('CORA:outOfDomain', 'The interval is outside the valid domain of atanh, which is [-1, 1].')
    
    # If execution reaches here, the interval is within the valid domain
    inf = np.arctanh(i.inf)
    sup = np.arctanh(i.sup)
    
    return Interval(inf, sup) 