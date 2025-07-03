from __future__ import annotations
"""
copy - returns a copy of the interval object
"""

from typing import TYPE_CHECKING
from .interval import Interval

if TYPE_CHECKING:
    from .interval import Interval

def copy(I: "Interval") -> "Interval":
    """
    returns a copy of the interval object

    Args:
        I: interval object
    
    Returns:
        res: copy of the interval object
    """
    
    return Interval(I) 