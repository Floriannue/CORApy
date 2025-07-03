from __future__ import annotations
"""
lift_ - lifts an interval to a higher-dimensional space
"""

from typing import TYPE_CHECKING
import numpy as np
from .interval import Interval
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

if TYPE_CHECKING:
    from .interval import Interval

def lift_(I: "Interval", N: int, proj: np.ndarray) -> "Interval":
    """
    lifts an interval to a higher-dimensional space

    Args:
        I: interval object
        N: dimension of the higher-dimensional space
        proj: states of the high-dimensional space that correspond to the
              states of the low-dimensional interval object
    
    Returns:
        I: interval object in the higher-dimensional space
    """
    
    if I.is_empty():
        raise CORAerror('CORA:notSupported', 'Operation lift is not supported for empty intervals.')
    
    lb = -np.inf * np.ones((N, 1))
    ub = np.inf * np.ones((N, 1))
    
    lb[proj] = I.inf.reshape(-1, 1)
    ub[proj] = I.sup.reshape(-1, 1)
    
    return Interval(lb, ub) 