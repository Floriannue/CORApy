from __future__ import annotations
"""
isnan - checks if any value in the interval is NaN
"""

from typing import TYPE_CHECKING
from .interval import Interval

if TYPE_CHECKING:
    from .interval import Interval

def isnan(I: "Interval") -> bool:
    """
    checks if any value in the interval is NaN

    Args:
        I: interval object
    
    Returns:
        res: false
    """
    
    # NaN values are not possible by constructor
    return False 