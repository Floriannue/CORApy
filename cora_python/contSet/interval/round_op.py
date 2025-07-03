from __future__ import annotations
"""
round - rounds each element of the interval to the given precision
"""

from typing import TYPE_CHECKING, Optional
import numpy as np
from .interval import Interval

if TYPE_CHECKING:
    from .interval import Interval

def round_op(I: "Interval", N: Optional[int] = 0) -> "Interval":
    """
    rounds each element of the interval to the given precision

    Args:
        I: Interval object
        N: number of digits
        
    Returns:
        I: rounded interval object
    """
    if not isinstance(N, int) or N < 0:
        raise ValueError("Number of digits N must be a non-negative integer.")

    if I.is_empty():
        return Interval.empty(I.dim())
        
    inf = np.round(I.inf, N)
    sup = np.round(I.sup, N)
    
    return Interval(inf, sup) 