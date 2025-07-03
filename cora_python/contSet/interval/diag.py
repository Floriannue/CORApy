from __future__ import annotations
"""
diag - Create diagonal matrix or get diagonal elements of matrix
"""

from typing import TYPE_CHECKING, Optional
import numpy as np
from .interval import Interval

if TYPE_CHECKING:
    from .interval import Interval

def diag(I: "Interval", k: Optional[int] = 0) -> "Interval":
    """
    Create diagonal matrix or get diagonal elements of matrix

    Args:
        I: interval object
        k: (optional) diagonal number
    
    Returns:
        res: diagonal matrix or diagonal elements of matrix
    """
    
    if I.is_empty():
        return Interval.empty()

    if I.inf.ndim > 2:
        raise ValueError("Interval must not be an n-d array with n > 2.")

    inf = np.diag(I.inf, k=k)
    sup = np.diag(I.sup, k=k)
    
    return Interval(inf, sup) 