"""
log - Overloaded log (natural logarithm) function for intervals 

x_ is x infimum, x-- is x supremum

[NaN, NaN] if (x-- < 0),
[NaN, log(x--)] if (x_ < 0) and (x-- >= 0)
[log(x_), log(x--)] if (x_ >= 0).

Syntax:
    I = log(I)

Inputs:
    I - interval object

Outputs:
    I - interval object

Example: 
    I = interval([3;9])
    res = log(I)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: mtimes

Authors: Dmitry Grebenyuk (MATLAB)
         Python translation by AI Assistant
Written: 07-February-2016 (MATLAB)
Last update: 21-February-2016 (DG, the matrix case is rewritten) (MATLAB)
             05-May-2020 (MW, addition of error message) (MATLAB)
             21-May-2022 (MW, remove new instantiation) (MATLAB)
             18-January-2024 (MW, simplify) (MATLAB)
Python translation: 2025
"""

import numpy as np
from .interval import Interval


def log(I: Interval) -> Interval:
    """
    Overloaded log (natural logarithm) function for intervals
    
    Args:
        I: Interval object
        
    Returns:
        Interval object with natural logarithm applied
        
    Raises:
        ValueError: If any interval contains non-positive values
    """
    # Handle empty intervals
    if I.is_empty():
        return Interval.empty(I.dim())
    
    # compute logarithm
    inf_result = np.log(I.inf)
    sup_result = np.log(I.sup)
    
    # set entries with non-zero imaginary parts to NaN
    # Handle both scalar and array cases
    if np.isscalar(inf_result):
        if np.imag(inf_result) != 0:
            inf_result = np.nan
    else:
        inf_result[np.imag(inf_result) != 0] = np.nan
        
    if np.isscalar(sup_result):
        if np.imag(sup_result) != 0:
            sup_result = np.nan
    else:
        sup_result[np.imag(sup_result) != 0] = np.nan
    
    # return error if NaN occurs
    if np.any(np.isnan(inf_result)) or np.any(np.isnan(sup_result)):
        raise ValueError('CORA:outOfDomain - validDomain: > 0')
    
    return Interval(inf_result, sup_result) 