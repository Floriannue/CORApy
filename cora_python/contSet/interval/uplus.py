"""
uplus - Overloaded '+' operator for single operand (dummy function)

Syntax:
    I = uplus(I)

Inputs:
    I - interval object

Outputs:
    I - interval object

Example: 
    I = interval([1;2],[3;4])
    +I

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: mtimes

Authors: Matthias Althoff (MATLAB)
         Python translation by AI Assistant
Written: 25-June-2015 (MATLAB)
Python translation: 2025
"""

from .interval import Interval


def uplus(I: Interval) -> Interval:
    """
    Overloaded unary plus operator for intervals (dummy function)
    
    Args:
        I: Interval object
        
    Returns:
        I: Same interval object (+I = I)
    """
    # +I = I (unary plus returns the same interval)
    return I 