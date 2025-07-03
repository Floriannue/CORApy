"""
acos - Overloaded 'acos()' operator for intervals

x_ is x infimum, x-- is x supremum

[NaN, NaN] if (x_ < -1) and (x-- > 1),
[NaN, NaN] if (x_ > 1) or (x-- < -1),
[NaN, pi] if (x_ < -1) and (x-- in [-1, 1]),
[0, NaN] if (x_ in [-1, 1]) and (x-- > 1),
[acos(x--), acos(x_)] if (x >= -1) and (x <= 1).

Syntax:
    res = acos(I)

Inputs:
    I - interval object

Outputs:
    res - interval object

Example: 
    I = Interval([-0.5, 0.3])
    res = I.acos()

Authors: Matthias Althoff (MATLAB)
         Python translation by AI Assistant
Written: 05-February-2016 (MATLAB)
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
    
    # to preserve the shape
    lb = I.inf.copy()
    ub = I.sup.copy()
    
    # Initialize result arrays
    res_inf = np.full_like(lb, np.nan)
    res_sup = np.full_like(ub, np.nan)
    
    # find indices
    # Case 1: [NaN, NaN] if (x_ < -1) and (x-- > 1), or (x_ > 1) or (x-- < -1)
    ind1 = ((lb < -1) & (ub > 1)) | (lb > 1) | (ub < -1)
    res_inf[ind1] = np.nan
    res_sup[ind1] = np.nan
    
    # Case 2: [NaN, pi] if (x_ < -1) and (x-- in [-1, 1])
    ind2 = (lb < -1) & (ub >= -1) & (ub <= 1)
    res_inf[ind2] = np.nan
    res_sup[ind2] = np.pi
    
    # Case 3: [0, NaN] if (x_ in [-1, 1]) and (x-- > 1)
    ind3 = (lb >= -1) & (lb <= 1) & (ub > 1)
    res_inf[ind3] = 0
    res_sup[ind3] = np.nan
    
    # Case 4: [acos(x--), acos(x_)] if (x >= -1) and (x <= 1)
    ind4 = (lb >= -1) & (ub <= 1)
    res_inf[ind4] = np.arccos(ub[ind4])  # acos is decreasing
    res_sup[ind4] = np.arccos(lb[ind4])
    
    # return error if NaN occurs
    if np.any(np.isnan(res_inf)) or np.any(np.isnan(res_sup)):
        raise CORAerror('CORA:outOfDomain', 'validDomain', '>= -1 && <= 1')
    
    return Interval(res_inf, res_sup) 