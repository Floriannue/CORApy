from __future__ import annotations
"""
issparse - checks if an interval is sparse
"""

from typing import TYPE_CHECKING
from scipy.sparse import spmatrix
from .interval import Interval

if TYPE_CHECKING:
    from .interval import Interval

def issparse(I: "Interval") -> bool:
    """
    checks if an interval is sparse

    Args:
        I: interval object
    
    Returns:
        res: true/false
    """
    
    return isinstance(I.inf, spmatrix) or isinstance(I.sup, spmatrix) 