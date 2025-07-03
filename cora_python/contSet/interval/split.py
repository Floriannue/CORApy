from __future__ import annotations
"""
split - split an interval in one dimension
"""

from typing import TYPE_CHECKING, List
import numpy as np
from .interval import Interval

if TYPE_CHECKING:
    from .interval import Interval

def split(I: "Interval", n: int) -> List["Interval"]:
    """
    split an interval in one dimension

    Args:
        I: interval object
        n: index of the dimension that is splitted
    
    Returns:
        res: list containing the splitted intervals
    """
    
    d = I.dim()
    if isinstance(d, list):
        raise ValueError("Interval must not be an n-d array with n > 1.")

    if n < 0 or n >= d:
        raise ValueError("Dimension index out of bounds.")

    m = I.center()

    sup = I.sup.copy()
    inf = I.inf.copy()

    sup[n] = m[n]
    inf[n] = m[n]

    return [Interval(I.inf, sup), Interval(inf, I.sup)] 