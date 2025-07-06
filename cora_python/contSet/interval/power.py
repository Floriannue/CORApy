"""
power - Overloaded '.^' operator for intervals (power)

For an interval .^ an integer number
[min(base.^exp), max(base.^exp)   if the exp is an integer and odd;
[0, max(base.^exp)                if the exp is an integer and even.

For an interval .^ a real number 
[min(base.^exp), max(base.^exp)   if base.inf >= 0;
[NaN, NaN]                        if otherwise.

For a number .^ an interval
[min(base.^exp), max(base.^exp)   if base.inf >= 0;
[NaN, NaN]                        if otherwise.

For an interval .^ an interval
[min(base.^exp), max(base.^exp)   if base.inf >= 0;
[NaN, NaN]                        if otherwise.

Syntax:
    res = power(base, exponent)

Inputs:
    base - interval object or numerical value
    exponent - interval object or numerical value

Outputs:
    res - interval object

Example:
    base = Interval([[-3], [2]], [[5], [4]])
    exponent = [2, 3]
    base.power(exponent)

Authors: Dmitry Grebenyuk (MATLAB)
         Python translation by AI Assistant
Written: 10-February-2016 (MATLAB)
Python translation: 2025
"""

import numpy as np
from .interval import Interval
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def power(base, exponent):
    """
    Overloaded power operator for intervals
    
    Args:
        base: Interval object or numerical value  
        exponent: Interval object or numerical value
        
    Returns:
        Interval object result of power operation
    """
    
    # Handle base as Interval, exponent as numeric
    if isinstance(base, Interval) and not isinstance(exponent, Interval):
        exponent = np.asarray(exponent)
        res = Interval(base.inf.copy(), base.sup.copy())
        
        if exponent.ndim == 0:  # Scalar exponent
            if abs(np.round(exponent) - exponent) <= np.finfo(float).eps:
                # Integer exponent
                if exponent >= 0:
                    temp1 = base.inf ** exponent
                    temp2 = base.sup ** exponent
                    res.inf = np.minimum(temp1, temp2)
                    res.sup = np.maximum(temp1, temp2)
                    
                    # Special behavior for even exponents
                    if exponent % 2 == 0 and exponent != 0:
                        ind = (base.inf < 0) & (base.sup > 0)
                        res.inf[ind] = 0
                else:
                    # Negative integer exponent
                    res = power(1 / base, -exponent)
            else:
                # Real-valued exponent
                if exponent >= 0:
                    if base.inf.size == 1:
                        if base.inf >= 0:
                            res.inf = base.inf ** exponent
                            res.sup = base.sup ** exponent
                        else:
                            res.inf = np.nan
                            res.sup = np.nan
                    else:
                        res.inf = base.inf ** exponent
                        res.sup = base.sup ** exponent
                        
                        ind = base.inf < 0
                        res.inf[ind] = np.nan
                        res.sup[ind] = np.nan
                else:
                    # Negative real exponent
                    res = power(1 / base, -exponent)
        
        else:  # Matrix exponent with scalar base
            if base.inf.size == 1:
                res.inf = np.zeros_like(exponent)
                res.sup = np.zeros_like(exponent)
                
                # Check if all elements are integers
                if np.all(np.abs(np.round(exponent) - exponent) <= np.finfo(float).eps):
                    # All integer exponents
                    ind_neg = exponent < 0
                    if np.any(ind_neg):
                        oneover = 1 / base
                        temp1 = oneover.inf[0,0] ** (-exponent[ind_neg])
                        temp2 = oneover.sup[0,0] ** (-exponent[ind_neg])
                        res.inf[ind_neg] = np.minimum(temp1, temp2)
                        res.sup[ind_neg] = np.maximum(temp1, temp2)
                    
                    ind_pos = exponent >= 0
                    if np.any(ind_pos):
                        temp1 = base.inf[0,0] ** exponent[ind_pos]
                        temp2 = base.sup[0,0] ** exponent[ind_pos]
                        res.inf[ind_pos] = np.minimum(temp1, temp2)
                        res.sup[ind_pos] = np.maximum(temp1, temp2)
                    
                    if base.inf < 0 and base.sup > 0:
                        ind_even = (exponent % 2 == 0) & (exponent != 0)
                        res.inf[ind_even] = 0
                else:
                    # Mixed real and integer exponents
                    ind_neg = exponent < 0
                    if np.any(ind_neg):
                        oneover = 1 / base
                        temp1 = oneover.inf[0,0] ** (-exponent[ind_neg])
                        temp2 = oneover.sup[0,0] ** (-exponent[ind_neg])
                        res.inf[ind_neg] = np.minimum(temp1, temp2)
                        res.sup[ind_neg] = np.maximum(temp1, temp2)
                    
                    ind_pos = exponent >= 0
                    if np.any(ind_pos):
                        temp1 = base.inf[0,0] ** exponent[ind_pos]
                        temp2 = base.sup[0,0] ** exponent[ind_pos]
                        res.inf[ind_pos] = np.minimum(temp1, temp2)
                        res.sup[ind_pos] = np.maximum(temp1, temp2)
                    
                    if base.inf < 0 and base.sup > 0:
                        ind_even = (exponent % 2 == 0) & (exponent != 0)
                        res.inf[ind_even] = 0
                    
                    if base.inf < 0:
                        ind_real = np.abs(np.round(exponent) - exponent) > np.finfo(float).eps
                        res.inf[ind_real] = np.nan
                        res.sup[ind_real] = np.nan
            
            else:  # Matrix base with matrix exponent
                if not np.array_equal(base.inf.shape, exponent.shape):
                    raise CORAerror('CORA:dimensionMismatch', base, exponent)
                
                res.inf = np.zeros_like(base.inf)
                res.sup = np.zeros_like(base.sup)
                
                if np.all(np.abs(np.round(exponent) - exponent) <= np.finfo(float).eps):
                    # All integer exponents
                    ind_neg = exponent < 0
                    if np.any(ind_neg):
                        oneover = 1 / base
                        temp1 = oneover.inf[ind_neg] ** (-exponent[ind_neg])
                        temp2 = oneover.sup[ind_neg] ** (-exponent[ind_neg])
                        res.inf[ind_neg] = np.minimum(temp1, temp2)
                        res.sup[ind_neg] = np.maximum(temp1, temp2)
                    
                    ind_pos = exponent >= 0
                    temp1 = base.inf[ind_pos] ** exponent[ind_pos]
                    temp2 = base.sup[ind_pos] ** exponent[ind_pos]
                    res.inf[ind_pos] = np.minimum(temp1, temp2)
                    res.sup[ind_pos] = np.maximum(temp1, temp2)
                    
                    # Special case for even exponents with intervals crossing zero
                    ind_even = (base.inf < 0) & (base.sup > 0) & (exponent % 2 == 0) & (exponent != 0)
                    res.inf[ind_even] = 0
                else:
                    # Mixed real and integer exponents
                    ind_neg = exponent < 0
                    if np.any(ind_neg):
                        oneover = 1 / base
                        temp1 = oneover.inf[ind_neg] ** (-exponent[ind_neg])
                        temp2 = oneover.sup[ind_neg] ** (-exponent[ind_neg])
                        res.inf[ind_neg] = np.minimum(temp1, temp2)
                        res.sup[ind_neg] = np.maximum(temp1, temp2)
                    
                    ind_pos = exponent >= 0
                    temp1 = base.inf[ind_pos] ** exponent[ind_pos]
                    temp2 = base.sup[ind_pos] ** exponent[ind_pos]
                    res.inf[ind_pos] = np.minimum(temp1, temp2)
                    res.sup[ind_pos] = np.maximum(temp1, temp2)
                    
                    # Domain issues for negative base with real exponents
                    ind_invalid = (base.inf < 0) & (np.abs(np.round(exponent) - exponent) > np.finfo(float).eps)
                    res.inf[ind_invalid] = np.nan
                    res.sup[ind_invalid] = np.nan
                    
                    # Special case for even exponents with intervals crossing zero
                    ind_even = (np.abs(np.round(exponent) - exponent) <= np.finfo(float).eps) & \
                               (base.inf < 0) & (base.sup > 0) & (exponent % 2 == 0) & (exponent != 0)
                    res.inf[ind_even] = 0
    
    # Handle numeric base, interval exponent
    elif not isinstance(base, Interval) and isinstance(exponent, Interval):
        base = np.asarray(base)
        res = Interval(exponent.inf.copy(), exponent.sup.copy())
        
        temp1 = base ** exponent.inf
        temp2 = base ** exponent.sup
        res.inf = np.minimum(temp1, temp2)
        res.sup = np.maximum(temp1, temp2)
        
        # Domain check for negative base
        ind_negative = base < 0
        res.inf[ind_negative] = np.nan
        res.sup[ind_negative] = np.nan
    
    # Handle interval base, interval exponent  
    elif isinstance(base, Interval) and isinstance(exponent, Interval):
        res = Interval(base.inf.copy(), base.sup.copy())
        
        # Compute all combinations and take min/max
        temp1 = base.inf ** exponent.inf
        temp2 = base.inf ** exponent.sup
        temp3 = base.sup ** exponent.inf
        temp4 = base.sup ** exponent.sup
        
        res.inf = np.minimum.reduce([temp1, temp2, temp3, temp4])
        res.sup = np.maximum.reduce([temp1, temp2, temp3, temp4])
        
        # Domain check for negative base
        ind_negative = base.inf < 0
        res.inf[ind_negative] = np.nan
        res.sup[ind_negative] = np.nan
    
    else:
        # Both numeric - convert to regular power
        return np.power(base, exponent)
    
    # Return error if NaN occurs
    if np.any(np.isnan(res.inf)) or np.any(np.isnan(res.sup)):
        raise CORAerror('CORA:outOfDomain', 'validDomain', 'base >= 0')
    
    return res
