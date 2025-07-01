"""
rdivide - Overloads the ./ operator that provides elementwise division
   of two matrices

For an interval / a number
[-Inf, +Inf]                                      if denominator = 0;
[min(inf / n, sup / n), max(inf / n, sup / n)     if denominator ~= 0.

For a number / an interval
[min(n / inf, n / sup), max(n / inf, n / sup)     if inf > 0 or sup < 0;
[ n / sup, +Inf]                                  if inf = 0;
[ -Inf, n / inf]                                  if sup = 0;
[NaN, NaN]                                        if inf = 0 and sup = 0;
[-Inf, +Inf]                                      if inf < 0 and sup > 0.

Syntax:
    res = rdivide(numerator, denominator)

Inputs:
    numerator - interval object or numerical value
    denominator - interval object or numerical value

Outputs:
    res - interval object after elementwise division

Example: 
    I = Interval([[-4], [2]], [[1], [3]])
    divisor = [3, 2]
    I = I.rdivide(divisor)

Authors: Dmitry Grebenyuk (MATLAB)
         Python translation by AI Assistant
Written: 07-February-2016 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import Union


def rdivide(numerator, denominator):
    """
    Overloaded elementwise division operator for intervals
    
    Args:
        numerator: Interval object or numerical value
        denominator: Interval object or numerical value
        
    Returns:
        Interval object result of elementwise division
    """
    from .interval import Interval
    from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

    
    # An interval / a (matrix of) scalar(s)
    if isinstance(numerator, Interval) and not isinstance(denominator, Interval):
        denominator = np.asarray(denominator)
        
        # Check size compatibility
        if not (np.array_equal(denominator.shape, numerator.inf.shape) or denominator.size == 1):
            raise CORAerror('CORA:specialError', 'The input size is wrong.')
        
        # Preserve the shape of input
        res = Interval(numerator.inf.copy(), numerator.sup.copy())
        
        # Compute division
        temp1 = numerator.inf / denominator
        temp2 = numerator.sup / denominator
        res.inf = np.minimum(temp1, temp2)
        res.sup = np.maximum(temp1, temp2)
        
        # Handle division by zero
        if denominator.size == numerator.inf.size:
            # Matrix case
            zero_mask = (denominator == 0)
            res.inf[zero_mask] = -np.inf
            res.sup[zero_mask] = np.inf
        elif denominator.size == 1:
            # Scalar case
            if denominator == 0:
                res.inf[:] = -np.inf
                res.sup[:] = np.inf
    
    # A (matrix of) scalar(s) / an interval
    elif not isinstance(numerator, Interval) and isinstance(denominator, Interval):
        numerator = np.asarray(numerator)
        
        if numerator.size == 1:
            # Scalar numerator
            res = Interval(denominator.inf.copy(), denominator.sup.copy())
            
            # Compute division avoiding division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                temp1 = numerator / denominator.sup
                temp2 = numerator / denominator.inf
            
            res.inf = np.minimum(temp1, temp2)
            res.sup = np.maximum(temp1, temp2)
            
            # Handle special cases
            # Case: inf = 0 and sup = 0
            zero_interval = (denominator.inf == 0) & (denominator.sup == 0)
            res.inf[zero_interval] = np.nan
            res.sup[zero_interval] = np.nan
            
            # Case: sup = 0 (but inf != 0)
            sup_zero = (denominator.sup == 0) & (denominator.inf != 0)
            res.inf[sup_zero] = -np.inf
            res.sup[sup_zero] = numerator / denominator.inf[sup_zero]
            
            # Case: inf = 0 (but sup != 0)  
            inf_zero = (denominator.inf == 0) & (denominator.sup != 0)
            res.inf[inf_zero] = numerator / denominator.sup[inf_zero]
            res.sup[inf_zero] = np.inf
            
            # Case: inf < 0 and sup > 0
            crosses_zero = (denominator.inf < 0) & (denominator.sup > 0)
            res.inf[crosses_zero] = -np.inf
            res.sup[crosses_zero] = np.inf
            
        elif np.array_equal(numerator.shape, denominator.inf.shape):
            # Matrix numerator with matching dimensions
            res = Interval(denominator.inf.copy(), denominator.sup.copy())
            
            # Compute division avoiding division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                temp1 = numerator / denominator.sup
                temp2 = numerator / denominator.inf
            
            res.inf = np.minimum(temp1, temp2)
            res.sup = np.maximum(temp1, temp2)
            
            # Handle special cases
            # Case: inf = 0 and sup = 0
            zero_interval = (denominator.inf == 0) & (denominator.sup == 0)
            res.inf[zero_interval] = np.nan
            res.sup[zero_interval] = np.nan
            
            # Case: sup = 0 (but inf != 0)
            sup_zero = (denominator.sup == 0) & (denominator.inf != 0)
            res.inf[sup_zero] = -np.inf
            res.sup[sup_zero] = numerator[sup_zero] / denominator.inf[sup_zero]
            
            # Case: inf = 0 (but sup != 0)
            inf_zero = (denominator.inf == 0) & (denominator.sup != 0)
            res.inf[inf_zero] = numerator[inf_zero] / denominator.sup[inf_zero]
            res.sup[inf_zero] = np.inf
            
            # Case: inf < 0 and sup > 0
            crosses_zero = (denominator.inf < 0) & (denominator.sup > 0)
            res.inf[crosses_zero] = -np.inf
            res.sup[crosses_zero] = np.inf
            
        else:
            raise CORAerror('CORA:specialError', 'The input size is wrong.')
    
    # An interval / an interval (x / y)
    elif isinstance(numerator, Interval) and isinstance(denominator, Interval):
        # Check size compatibility
        if not (np.array_equal(numerator.inf.shape, denominator.inf.shape) or 
                numerator.inf.size == 1 or denominator.inf.size == 1):
            raise CORAerror('CORA:specialError', 'The input size is wrong.')
        
        # Use multiplication by reciprocal: x / y = x * (1/y)
        y_reciprocal = 1 / denominator
        res = numerator * y_reciprocal
    
    else:
        # Both are numeric, use regular division
        return numerator / denominator
    
    # Return error if NaN occurs
    if np.any(np.isnan(res.inf)) or np.any(np.isnan(res.sup)):
        raise CORAerror('CORA:outOfDomain', 'validDomain', 'inf ~= 0 && sup ~= 0')
    
    return res 