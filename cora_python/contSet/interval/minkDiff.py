from __future__ import annotations
"""
minkDiff - compute the Minkowski difference of two intervals
"""

from typing import TYPE_CHECKING, Union
import numpy as np
from cora_python.contSet import interval
from cora_python.contSet.contSet.contSet import ContSet
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

if TYPE_CHECKING:
    from cora_python.contSet.interval.interval import Interval

def minkDiff(I: "Interval", S: Union["Interval", "ContSet", np.ndarray], type: str = "approx") -> "Interval":
    """
    compute the Minkowski difference of two intervals

    Args:
        I: interval object
        S: interval object, contSet object, or numerical vector
        type: type of computation ('exact' or 'inner')
    
    Returns:
        res: interval object after Minkowski difference
    """
    
    if isinstance(S, np.ndarray):
        return I + (-S)
    elif isinstance(S, interval.Interval):
        try:
            res = interval.Interval(I.inf - S.inf, I.sup - S.sup)
            return res
        except CORAerror:
            return interval.Interval.empty(I.dim())
    else: # S is a contSet
        if type == 'exact':
            raise CORAerror('exact Minkowski difference only implemented for intervals')

        # over-approximate Minkowski difference
        inf = I.inf.copy()
        sup = I.sup.copy()
        n = I.dim()

        for i in range(n):
            # create a vector for the i-th dimension
            e_i = np.zeros((n, 1))
            e_i[i] = 1

            # calculate the new bounds
            sup[i] = sup[i] - S.supportFunc(e_i, 'upper')
            inf[i] = inf[i] + S.supportFunc(-e_i, 'upper')

        res = interval.Interval(inf, sup)

    return res 