from __future__ import annotations
"""
isscalar - check if an interval is one-dimensional
"""

from typing import TYPE_CHECKING
import numpy as np
from .interval import Interval

if TYPE_CHECKING:
    from .interval import Interval

def isscalar(I: "Interval") -> bool:
    """
    check if an interval is one-dimensional

    Args:
        I: interval object
    
    Returns:
        res: true/false
    """
    
    return I.inf.size == 1 