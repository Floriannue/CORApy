"""
times - Overloaded '.*' operator for intervals

Syntax:
    res = times(factor1,factor2)

Inputs:
    factor1 - interval object or numeric
    factor2 - interval object or numeric

Outputs:
    res - interval object

Example:
    factor1 = interval([[-2, 3], [1, 2]])
    factor2 = interval([[-1, 1], [-2, 2]])
    result = times(factor1, factor2)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: mtimes

Authors:       Matthias Althoff (MATLAB)
               Python translation by AI Assistant
Written:       19-June-2015
Last update:   13-January-2016 (DG)
               04-April-2023 (TL, minor optimizations)
               04-October-2023 (TL, fix for 0*[inf,-inf])
Last revision: ---
"""

import numpy as np
from typing import Union
from .interval import Interval

def times(factor1: Union[Interval, np.ndarray, float, int], factor2: Union[Interval, np.ndarray, float, int]) -> Interval:
    """
    Element-wise multiplication for intervals.
    
    Implements the .* operator for intervals.
    
    Args:
        factor1: interval object or numeric
        factor2: interval object or numeric
        
    Returns:
        interval: result of element-wise multiplication
    """
    
    # Ensure factor1 is the interval if one of them is
    if not isinstance(factor1, Interval) and isinstance(factor2, Interval):
        factor1, factor2 = factor2, factor1

    # Handle multiplication with a numeric type
    if isinstance(factor1, Interval) and not isinstance(factor2, Interval):
        # Delegate scalar multiplication to mtimes for consistency
        if np.isscalar(factor2):
            return factor1.mtimes(factor2)
        
        # Element-wise multiplication with a matrix or vector
        f2_arr = np.asanyarray(factor2)
        res_inf = factor1.inf * f2_arr
        res_sup = factor1.sup * f2_arr
        
        # fix 0*Inf=NaN cases, which should be 0
        res_inf = np.where(np.isnan(res_inf), 0, res_inf)
        res_sup = np.where(np.isnan(res_sup), 0, res_sup)
        
        # The new bounds are the min and max of the resulting products
        inf = np.minimum(res_inf, res_sup)
        sup = np.maximum(res_inf, res_sup)
        return Interval(inf, sup)
        
    # Both factors are intervals
    elif isinstance(factor1, Interval) and isinstance(factor2, Interval):
        # Possible combinations of bounds
        res1 = factor1.inf * factor2.inf
        res2 = factor1.inf * factor2.sup
        res3 = factor1.sup * factor2.inf
        res4 = factor1.sup * factor2.sup
        
        # fix 0*Inf=NaN cases, which should be 0
        res1 = np.where(np.isnan(res1), 0, res1)
        res2 = np.where(np.isnan(res2), 0, res2)
        res3 = np.where(np.isnan(res3), 0, res3)
        res4 = np.where(np.isnan(res4), 0, res4)
        
        # Find new infimum and supremum
        inf = np.minimum.reduce([res1, res2, res3, res4])
        sup = np.maximum.reduce([res1, res2, res3, res4])
        return Interval(inf, sup)

    # Both factors are numeric
    else:
        # This case should ideally not be handled here, but for completeness:
        return Interval(factor1 * factor2) 