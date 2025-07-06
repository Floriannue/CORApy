from __future__ import annotations
"""
sum - Overloaded 'sum()' operator for intervals
"""

from typing import TYPE_CHECKING, Optional
import numpy as np
from .interval import Interval
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

if TYPE_CHECKING:
    from .interval import Interval

def sum_op(I: "Interval", dim: Optional[int] = None) -> "Interval":
    """
    Overloaded 'sum()' operator for intervals

    Args:
        I: interval object
        dim: dimension along which the sum should be computed

    Returns:
        res: interval object
    """

    if dim is None:
        if I.inf.shape[0] == 1:
            dim = 1
        else:
            dim = 0
    elif not isinstance(dim, int) or dim not in [0, 1]:
        raise CORAerror('wrongValue', 'second', 'either 0 or 1')


    inf = np.sum(I.inf, axis=dim)
    sup = np.sum(I.sup, axis=dim)
    
    return Interval(inf, sup) 