from __future__ import annotations
"""
tril - gets lower triangular part of I
"""

from typing import TYPE_CHECKING, Optional
import numpy as np
from .interval import Interval

if TYPE_CHECKING:
    from .interval import Interval

def tril(I: "Interval", k: Optional[int] = 0) -> "Interval":
    """
    gets lower triangular part of I

    Args:
        I: interval object
        k: (see built-in tril for matrices)
    
    Returns:
        res: lower triangular interval object
    """
    
    res = I.copy()
    res.inf = np.tril(res.inf, k=k)
    res.sup = np.tril(res.sup, k=k)
    
    return res 