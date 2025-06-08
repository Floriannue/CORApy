"""
plus - overloaded '+' operator for the Minkowski sum of an interval and
    another set or point

Syntax:
    S_out = I + S
    S_out = plus(I, S)

Inputs:
    I - interval object, numeric
    S - contSet object, numeric

Outputs:
    S_out - Minkowski sum

Authors: Matthias Althoff (MATLAB)
         Python translation by AI Assistant
Written: 19-June-2015 (MATLAB)
Last update: 05-May-2020 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import Union
from .interval import interval
from .aux_functions import _reorder_numeric, _equal_dim_check, _representsa
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAError


def plus(I: interval, S: Union[interval, np.ndarray, float, int]) -> interval:
    """
    Overloaded '+' operator for the Minkowski sum of an interval and another set or point
    
    Args:
        I: interval object or numeric
        S: contSet object or numeric
        
    Returns:
        S_out: Minkowski sum
    """
    # Ensure that numeric is second input argument
    S_out, S = _reorder_numeric(I, S)
    
    # Call function with lower precedence
    if hasattr(S, 'precedence') and S.precedence < S_out.precedence:
        return S + S_out
    
    try:
        # interval-interval case
        if isinstance(S, interval):
            result = interval.__new__(interval)
            result.precedence = S_out.precedence
            result.inf = S_out.inf + S.inf
            result.sup = S_out.sup + S.sup
            return result
        
        # numeric vector/matrix
        if isinstance(S, (int, float, np.ndarray)):
            S = np.asarray(S)
            result = interval.__new__(interval)
            result.precedence = S_out.precedence
            result.inf = S_out.inf + S
            result.sup = S_out.sup + S
            return result
            
    except Exception as e:
        # Check whether different dimension of ambient space
        _equal_dim_check(S_out, S)
        
        # Check for empty sets
        if (_representsa(S_out, 'emptySet', 1e-9) or 
            _representsa(S, 'emptySet', 1e-9)):
            return interval.empty(S_out.dim())
        
        # Re-raise the original error
        raise e
    
    # If we get here, the operation is not supported
    raise CORAError('CORA:noops', f'Operation not supported between {type(S_out)} and {type(S)}') 