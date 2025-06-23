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
from .interval import Interval
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def minus(minuend: Union[Interval, np.ndarray, float, int], 
          subtrahend: Union[Interval, np.ndarray, float, int]) -> Interval:
    """
    Overloaded '-' operator for intervals
    
    Args:
        minuend: Interval or numerical value
        subtrahend: Interval or numerical value
        
    Returns:
        res: Interval result
    """
    # Find an interval object
    # Is minuend an interval?
    if isinstance(minuend, Interval):
        # Initialize result with minuend
        res = Interval.__new__(Interval)
        res.precedence = minuend.precedence
        
        # Is subtrahend an interval?
        if isinstance(subtrahend, Interval):
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
        if not isinstance(subtrahend, Interval):
            raise CORAerror('CORA:wrongInput', 
                           'At least one operand must be an interval')
        
        # Initialize result with subtrahend structure
        res = Interval.__new__(Interval)
        res.precedence = subtrahend.precedence
        
        # minuend must be a particular value
        minuend = np.asarray(minuend)
        # Calculate infimum and supremum
        # For numeric - interval: c - [a,b] = [c-b, c-a]
        res.inf = minuend - subtrahend.sup
        res.sup = minuend - subtrahend.inf
    
    return res 
