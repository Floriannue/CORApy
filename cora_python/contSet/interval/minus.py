"""
minus - Overloaded '-' operator for intervals

Evaluates the operation minuend + (-1)*subtrahend for two intervals,
where '+' is the Minkowski sum. Not to be confused with the Minkowski 
difference operation.

Syntax:
    res = minus(minuend, subtrahend)

Inputs:
    minuend - interval or numerical value
    subtrahend - interval or numerical value

Outputs:
    res - interval

Authors: Matthias Althoff (MATLAB)
         Python translation by AI Assistant
Written: 25-June-2015 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import Union
from .interval import interval
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAError


def minus(minuend: Union[interval, np.ndarray, float, int], 
          subtrahend: Union[interval, np.ndarray, float, int]) -> interval:
    """
    Overloaded '-' operator for intervals
    
    Args:
        minuend: interval or numerical value
        subtrahend: interval or numerical value
        
    Returns:
        res: interval result
    """
    # Find an interval object
    # Is minuend an interval?
    if isinstance(minuend, interval):
        # Initialize result with minuend
        res = interval.__new__(interval)
        res.precedence = minuend.precedence
        
        # Is subtrahend an interval?
        if isinstance(subtrahend, interval):
            # Calculate infimum and supremum
            # For interval subtraction: [a,b] - [c,d] = [a-d, b-c]
            res.inf = minuend.inf - subtrahend.sup
            res.sup = minuend.sup - subtrahend.inf
        else:
            # subtrahend is numeric
            subtrahend = np.asarray(subtrahend)
            # Calculate infimum and supremum
            res.inf = minuend.inf - subtrahend
            res.sup = minuend.sup - subtrahend
    else:
        # minuend is numeric, subtrahend must be interval
        if not isinstance(subtrahend, interval):
            raise CORAError('CORA:wrongInput', 
                           'At least one operand must be an interval')
        
        # Initialize result with subtrahend structure
        res = interval.__new__(interval)
        res.precedence = subtrahend.precedence
        
        # minuend must be a particular value
        minuend = np.asarray(minuend)
        # Calculate infimum and supremum
        # For numeric - interval: c - [a,b] = [c-b, c-a]
        res.inf = minuend - subtrahend.sup
        res.sup = minuend - subtrahend.inf
    
    return res 