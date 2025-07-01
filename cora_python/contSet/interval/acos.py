"""
acos - Overloaded 'acos()' operator for intervals

Syntax:
    res = acos(I)

Inputs:
    I - interval object

Outputs:
    res - interval object

Example: 
    I = Interval(-0.5, 0.8)
    I.acos()

Authors: Dmitry Grebenyuk (MATLAB)
         Python translation by AI Assistant
Written: 06-February-2016 (MATLAB)
Python translation: 2025
"""

import numpy as np

from .interval import Interval
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def acos(I):
    """
    Overloaded inverse cosine operator for intervals
    
    Args:
        I: Interval object
        
    Returns:
        Interval object result of inverse cosine operation
    """
    
    # acos is only defined for values in [-1, 1]
    # Check domain validity
    if np.any(I.inf < -1) or np.any(I.sup > 1):
        raise CORAerror('CORA:outOfDomain', 'validDomain', '[-1,1]')
    
    # acos is monotonically decreasing, so acos(inf) becomes sup and acos(sup) becomes inf
    res_inf = np.arccos(I.sup)
    res_sup = np.arccos(I.inf)
    
    return Interval(res_inf, res_sup) 