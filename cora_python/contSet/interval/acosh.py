"""
acosh - Overloaded 'acosh()' operator for intervals

x_ is x infimum, x-- is x supremum

[NaN, NaN] if (x-- < 1),
[NaN, acosh(x--)] if (x_ < 1) and (x-- >= 1),
[acosh(x_), acosh(x--)] if (x_ >= 1).

Syntax:
    res = acosh(I)

Inputs:
    I - interval object

Outputs:
    res - interval object

Example: 
    I = Interval([2, 4])
    res = I.acosh()

Authors: Matthias Althoff (MATLAB)
         Python translation by AI Assistant
Written: 12-February-2016 (MATLAB)
Python translation: 2025
"""

import numpy as np

from .interval import Interval
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def acosh(I):
    """
    Overloaded inverse hyperbolic cosine operator for intervals
    
    Args:
        I: Interval object
        
    Returns:
        Interval object result of inverse hyperbolic cosine operation
    """
    
    # to preserve the shape
    lb = I.inf.copy()
    ub = I.sup.copy()
    
    # Initialize result arrays
    res_inf = np.full_like(lb, np.nan)
    res_sup = np.full_like(ub, np.nan)
    
    # find indices
    # Case 1: [acosh(x_), acosh(x--)] if (x_ >= 1)
    ind1 = lb >= 1
    res_inf[ind1] = np.arccosh(lb[ind1])
    res_sup[ind1] = np.arccosh(ub[ind1])
    
    # Case 2: [NaN, acosh(x--)] if (x_ < 1) and (x-- >= 1)
    ind2 = (lb < 1) & (ub >= 1)
    res_inf[ind2] = np.nan
    res_sup[ind2] = np.arccosh(ub[ind2])
    
    # Case 3: [NaN, NaN] if (x-- < 1)
    ind3 = ub < 1
    res_inf[ind3] = np.nan
    res_sup[ind3] = np.nan
    
    # return error if NaN occurs
    if np.any(np.isnan(res_inf)) or np.any(np.isnan(res_sup)):
        raise CORAerror('CORA:outOfDomain', 'validDomain', '>= 1')
    
    return Interval(res_inf, res_sup) 