"""
uminus - Overloaded '-' operator for single operand

Syntax:
   I = uminus(I)

Inputs:
   I - interval object

Outputs:
   I - interval object

Example: 
   I = Interval([-2, -3], [5, 6])
   -I

Authors:       Matthias Althoff
Written:       25-June-2015
Last update:   21-May-2022 (MW, simpler computation)
Last revision: ---
"""

import numpy as np

from .interval import Interval


def uminus(I: Interval) -> Interval:
    """
    Overloaded unary minus operator for intervals.
    
    Args:
        I: Interval object
        
    Returns:
        Interval object with negated bounds
    """
    
    # Negate and swap bounds: -[a,b] = [-b,-a]
    return Interval(-I.sup, -I.inf) 