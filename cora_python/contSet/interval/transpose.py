"""
transpose - Overloaded '.T' operator for single operand

Syntax:
   I = transpose(I)

Inputs:
   I - interval object

Outputs:
   I - interval object

Example: 
   I = Interval([[-2], [-3], [-4]], [[5], [6], [7]])
   I.T

Authors:       Dmitry Grebenyuk
Written:       07-February-2016
Last update:   ---
Last revision: ---
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .interval import Interval


def transpose(I: 'Interval') -> 'Interval':
    """
    Overloaded transpose operator for intervals.
    
    Args:
        I: Interval object
        
    Returns:
        Interval object with transposed bounds
    """
    from .interval import Interval
    
    # Transpose both inf and sup
    return Interval(I.inf.T, I.sup.T) 