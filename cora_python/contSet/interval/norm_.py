from __future__ import annotations
"""
norm_ - computes the exact maximum norm value of specified norm
"""

from typing import TYPE_CHECKING, Optional, Union
import numpy as np
from .interval import Interval

if TYPE_CHECKING:
    from .interval import Interval

def norm_(I: "Interval", p: Optional[Union[int, str]] = 2) -> float:
    """
    computes the exact maximum norm value of specified norm

    Args:
        I: interval object
        p: (optional) additional arguments of builtin/norm
    
    Returns:
        val: norm value
    """
    
    if I.is_empty():
        return -np.inf

    if p == 2:
        # For the 2-norm, we compute the norm of the element-wise maximum
        # of the absolute values of the interval's bounds.
        m = np.maximum(np.abs(I.inf), np.abs(I.sup))
        # The Frobenius norm is the default for matrices
        return np.linalg.norm(m)
    else:
        # For other norms, we would need to implement intervalMatrix logic.
        # This is not yet implemented.
        raise NotImplementedError("Only the 2-norm is currently implemented for intervals.") 