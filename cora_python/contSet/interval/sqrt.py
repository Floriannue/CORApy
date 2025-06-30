"""
sqrt - Overloaded 'sqrt'-function for intervals, computes the square root
   of the interval

x_ is x infimum, x-- is x supremum

[NaN, NaN] if (x-- < 0),
[NaN, sqrt(x--)] if (x_ < 0) and (x-- >= 0),
[sqrt(x_), sqrt(x--)] if (x_ >= 0).

Syntax:
    res = sqrt(I)

Inputs:
    I - interval object

Outputs:
    res - interval object

Example:
    I = interval(9,16)
    sqrt(I)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors: Matthias Althoff (MATLAB)
         Python translation by AI Assistant
Written: 20-January-2016 (MATLAB)
Last update: 21-February-2016 (DG, the matrix case is rewritten) (MATLAB)
             05-May-2020 (MW, standardized error message) (MATLAB)
             18-January-2024 (MW, simplify) (MATLAB)
Python translation: 2025
"""

import numpy as np
from .interval import Interval


def sqrt(I: Interval) -> Interval:
    """
    Overloaded sqrt function for intervals
    
    Args:
        I: Interval object
        
    Returns:
        Interval object with square root applied
        
    Raises:
        ValueError: If any interval contains negative values
    """
    # Handle empty intervals
    if I.is_empty():
        return Interval.empty(I.dim())
    
    # to preserve the shape
    lb = I.inf.copy()
    ub = I.sup.copy()
    
    # Initialize result arrays
    inf_result = np.full_like(lb, np.nan)
    sup_result = np.full_like(ub, np.nan)
    
    # find indices
    ind1 = lb >= 0
    inf_result[ind1] = np.sqrt(lb[ind1])
    sup_result[ind1] = np.sqrt(ub[ind1])
    
    ind2 = (lb < 0) & (ub >= 0)
    inf_result[ind2] = np.nan
    sup_result[ind2] = np.sqrt(ub[ind2])
    
    ind3 = ub < 0
    inf_result[ind3] = np.nan
    sup_result[ind3] = np.nan
    
    # return error if NaN occurs
    if np.any(np.isnan(inf_result)) or np.any(np.isnan(sup_result)):
        raise ValueError('CORA:outOfDomain - validDomain: >= 0')
    
    return Interval(inf_result, sup_result) 