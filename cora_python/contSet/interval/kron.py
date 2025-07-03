from __future__ import annotations
"""
kron - Kronecker product of two intervals
"""

from typing import TYPE_CHECKING, Union
import numpy as np
from .interval import Interval

if TYPE_CHECKING:
    from .interval import Interval

def kron(I1: Union[np.ndarray, "Interval"], I2: Union[np.ndarray, "Interval"]) -> "Interval":
    """
    Kronecker product of two intervals

    Args:
        I1: interval or numerical matrix
        I2: interval or numerical matrix
    
    Returns:
        res: interval matrix
    """
    
    if not isinstance(I1, Interval):
        vals = np.vstack([
            np.kron(I1, I2.inf).flatten(),
            np.kron(I1, I2.sup).flatten()
        ]).T
        sz = (np.array(I1.shape) * np.array(I2.inf.shape)).tolist()
    elif not isinstance(I2, Interval):
        vals = np.vstack([
            np.kron(I1.inf, I2).flatten(),
            np.kron(I1.sup, I2).flatten()
        ]).T
        sz = (np.array(I1.inf.shape) * np.array(I2.shape)).tolist()
    else:
        vals = np.vstack([
            np.kron(I1.inf, I2.inf).flatten(),
            np.kron(I1.inf, I2.sup).flatten(),
            np.kron(I1.sup, I2.inf).flatten(),
            np.kron(I1.sup, I2.sup).flatten()
        ]).T
        sz = (np.array(I1.inf.shape) * np.array(I2.inf.shape)).tolist()

    inf_mat = np.min(vals, axis=1).reshape(sz)
    sup_mat = np.max(vals, axis=1).reshape(sz)
    
    return Interval(inf_mat, sup_mat) 