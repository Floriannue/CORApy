from __future__ import annotations
"""
prod - product of array elements
"""

from typing import TYPE_CHECKING, Optional
import numpy as np
from .interval import Interval
import itertools

if TYPE_CHECKING:
    from .interval import Interval

def prod_op(I: "Interval", axis: Optional[int] = None) -> "Interval":
    """
    product of array elements

    Args:
        I: interval object
        axis: dimension along which the product should be computed
    
    Returns:
        res: interval object
    """
    
    if I.is_empty():
        return Interval.empty()

    shape = I.inf.shape

    if axis is None:
        if len(shape) > 1 and shape[0] == 1:
            axis = 1
        else:
            axis = 0
            
    if I.shape[axis] == 0:
        return Interval.empty()

    if len(shape) == 1:
        bounds = list(zip(I.inf, I.sup))
        products = [np.prod(p) for p in itertools.product(*bounds)]
        return Interval(min(products), max(products))
        
    if axis == 0:
        res_inf = []
        res_sup = []
        for col in range(I.shape[1]):
            bounds = list(zip(I.inf[:, col], I.sup[:, col]))
            products = [np.prod(p) for p in itertools.product(*bounds)]
            res_inf.append(min(products))
            res_sup.append(max(products))
        return Interval(res_inf, res_sup)

    if axis == 1:
        res_inf = []
        res_sup = []
        for row in range(I.shape[0]):
            bounds = list(zip(I.inf[row, :], I.sup[row, :]))
            products = [np.prod(p) for p in itertools.product(*bounds)]
            res_inf.append(min(products))
            res_sup.append(max(products))
        return Interval(res_inf, res_sup) 