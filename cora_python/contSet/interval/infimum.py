"""
infimum - returns the infimum of an interval

Syntax:
    res = infimum(I)

Inputs:
    I - interval object

Outputs:
    res - infimum of interval

Example: 
    I = interval([-1 1], [1 2])
    res = infimum(I)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: ---

Authors: Matthias Althoff (MATLAB)
         Python translation by AI Assistant
Written: 25-June-2015 (MATLAB)
Python translation: 2025
"""

from .interval import Interval


def infimum(I: Interval):
    """
    Returns the infimum of an interval
    
    Args:
        I: Interval object
        
    Returns:
        Infimum of the interval (copy to ensure independence)
    """
    return I.inf.copy() 