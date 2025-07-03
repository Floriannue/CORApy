from __future__ import annotations
"""
sign - Overloaded 'sign()' operator for intervals.

Syntax:
    res = sign(I)

Inputs:
    I - interval object

Outputs:
    res - interval object
"""

from typing import TYPE_CHECKING
import numpy as np
from .interval import Interval

if TYPE_CHECKING:
    from .interval import Interval

def sign(I: "Interval") -> "Interval":
    """
    overloaded built-in signum function; an exact evaluation would
    only contain the values -1, 0, and 1, thus, our result represents an
    outer approximation
    
    Args:
        I: Interval object
        
    Returns:
        Interval object
    """
    
    # an empty interval remains empty
    if I.is_empty():
        return Interval.empty(I.dim())

    res = Interval(np.sign(I.inf), np.sign(I.sup))
    
    return res 