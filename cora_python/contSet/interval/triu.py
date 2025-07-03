from __future__ import annotations
"""
triu - gets upper triangular part of I
"""

from typing import TYPE_CHECKING, Optional
import numpy as np
from .interval import Interval

if TYPE_CHECKING:
    from .interval import Interval

def triu(I: "Interval", k: Optional[int] = 0) -> "Interval":
    """
    gets upper triangular part of I

    Args:
        I: interval object
        k: (see built-in triu for matrices)
    
    Returns:
        res: upper triangular interval object
    """
    
    res = I.copy()
    res.inf = np.triu(res.inf, k=k)
    res.sup = np.triu(res.sup, k=k)
    
    return res 