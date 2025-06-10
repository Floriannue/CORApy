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
from .mtimes import mtimes

def times(factor1, factor2):
    """
    Element-wise multiplication for intervals.
    
    Implements the .* operator for intervals.
    
    Args:
        factor1: interval object or numeric
        factor2: interval object or numeric
        
    Returns:
        interval: result of element-wise multiplication
    """
    from .interval import Interval
    
    # If one factor is an interval and the other is a scalar, use mtimes
    if isinstance(factor1, Interval) and not isinstance(factor2, Interval):
        if np.isscalar(factor2):
            # use mtimes (considers scalar case explicitly)
            return mtimes(factor1, factor2)
        
        # Element-wise multiplication with non-scalar numeric
        res_inf = factor1.inf * factor2
        res_sup = factor1.sup * factor2
        
        # fix 0*Inf=NaN cases
        res_inf = np.where(np.isnan(res_inf), 0, res_inf)
        res_sup = np.where(np.isnan(res_sup), 0, res_sup)
        
        # Create result interval
        result = Interval.__new__(Interval)
        result.inf = np.minimum(res_inf, res_sup)
        result.sup = np.maximum(res_inf, res_sup)
        return result
        
    # If one factor is numeric and the other is interval
    elif not isinstance(factor1, Interval) and isinstance(factor2, Interval):
        if np.isscalar(factor1):
            # use mtimes (considers scalar case explicitly)
            return mtimes(factor1, factor2)
        
        # Element-wise multiplication with non-scalar numeric
        res_inf = factor2.inf * factor1
        res_sup = factor2.sup * factor1
        
        # fix 0*Inf=NaN cases
        res_inf = np.where(np.isnan(res_inf), 0, res_inf)
        res_sup = np.where(np.isnan(res_sup), 0, res_sup)
        
        # Create result interval
        result = Interval.__new__(Interval)
        result.inf = np.minimum(res_inf, res_sup)
        result.sup = np.maximum(res_inf, res_sup)
        return result
        
    # Both factors are intervals
    else:
        # Possible combinations
        res1 = factor1.inf * factor2.inf
        res2 = factor1.inf * factor2.sup
        res3 = factor1.sup * factor2.inf
        res4 = factor1.sup * factor2.sup
        
        # fix 0*Inf=NaN cases
        res1 = np.where(np.isnan(res1), 0, res1)
        res2 = np.where(np.isnan(res2), 0, res2)
        res3 = np.where(np.isnan(res3), 0, res3)
        res4 = np.where(np.isnan(res4), 0, res4)
        
        # Find minimum and maximum
        result = Interval.__new__(Interval)
        result.inf = np.minimum(np.minimum(res1, res2), np.minimum(res3, res4))
        result.sup = np.maximum(np.maximum(res1, res2), np.maximum(res3, res4))
        return result 