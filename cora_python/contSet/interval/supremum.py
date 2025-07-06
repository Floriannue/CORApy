"""
supremum - returns the supremum of an interval

Syntax:
    res = supremum(I)

Inputs:
    I - interval object

Outputs:
    res - numerical value

Example: 
    I = interval([-1;1],[1;2])
    res = supremum(I)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors: Matthias Althoff (MATLAB)
         Python translation by AI Assistant
Written: 25-June-2015 (MATLAB)
Python translation: 2025
"""

from .interval import Interval


def supremum(I: Interval):
    """
    Returns the supremum of an interval
    
    Args:
        I: Interval object
        
    Returns:
        Supremum of the interval (copy to ensure independence)
    """
    return I.sup.copy() 