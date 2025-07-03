from __future__ import annotations
"""
sum - Overloaded 'sum()' operator for intervals
"""

from typing import TYPE_CHECKING, Optional
import numpy as np
from .interval import Interval

if TYPE_CHECKING:
    from .interval import Interval

def sum_op(I: "Interval", axis: Optional[int] = None) -> "Interval":
    """
    Overloaded 'sum()' operator for intervals

    Args:
        I: interval object
        axis: dimension along which the sum should be computed
    
    Returns:
        res: interval object
    """
    
    if I.is_empty():
        return Interval.empty()

    if axis is None:
        # MATLAB's default sum is along the first non-singleton dimension
        shape = I.inf.shape
        if shape[0] == 1:
            axis = 1
        else:
            axis = 0

    inf = np.sum(I.inf, axis=axis)
    sup = np.sum(I.sup, axis=axis)
    
    return Interval(inf, sup) 