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
from .interval import Interval
# Removed static helper imports - use object methods and proper validation instead
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def plus(I: Interval, S: Union[Interval, np.ndarray, float, int]) -> Interval:
    """
    Overloaded '+' operator for the Minkowski sum of an interval and another set or point
    
    Args:
        I: Interval object or numeric
        S: contSet object or numeric
        
    Returns:
        S_out: Minkowski sum
    """
    # Ensure that numeric is second input argument (reorder if needed)
    if not isinstance(I, Interval) and isinstance(S, Interval):
        S_out, S = S, I  # Swap so interval is first
    else:
        S_out, S = I, S
    
    # Since Interval has highest precedence (120), it should handle all operations
    # No need to delegate to lower precedence functions
    
    try:
        # interval-interval case
        if isinstance(S, Interval):
            result = Interval.__new__(Interval)
            result.precedence = S_out.precedence
            # Check if both intervals are vectors (1D or column vectors) vs matrices
            # Only flatten if both are vectors to avoid creating 2D intervals from vector addition
            # Preserve matrix shape if one is a true matrix (shape (m, n) where n > 1)
            S_out_is_vector = (S_out.inf.ndim == 1) or (S_out.inf.ndim == 2 and S_out.inf.shape[1] == 1)
            S_is_vector = (S.inf.ndim == 1) or (S.inf.ndim == 2 and S.inf.shape[1] == 1)
            
            if S_out_is_vector and S_is_vector:
                # Both are vectors - flatten to 1D to avoid creating 2D intervals
                S_out_inf = S_out.inf.flatten() if S_out.inf.ndim > 1 else S_out.inf
                S_out_sup = S_out.sup.flatten() if S_out.sup.ndim > 1 else S_out.sup
                S_inf = S.inf.flatten() if S.inf.ndim > 1 else S.inf
                S_sup = S.sup.flatten() if S.sup.ndim > 1 else S.sup
            else:
                # At least one is a matrix - preserve shapes (they should match)
                S_out_inf = S_out.inf
                S_out_sup = S_out.sup
                S_inf = S.inf
                S_sup = S.sup
            
            result.inf = S_out_inf + S_inf
            result.sup = S_out_sup + S_sup
            return result
        
        # numeric vector/matrix
        if isinstance(S, (int, float, np.ndarray)):
            S = np.asarray(S)
            result = Interval.__new__(Interval)
            result.precedence = S_out.precedence
            # Ensure compatible shapes - if S_out.inf is 1D and S is 2D (column vector), 
            # reshape S_out.inf to match, or vice versa
            inf_shape = S_out.inf.shape
            sup_shape = S_out.sup.shape
            S_shape = S.shape
            
            # If shapes don't match in a way that would cause unwanted broadcasting
            if inf_shape != S_shape and len(inf_shape) == 1 and len(S_shape) == 2 and S_shape[1] == 1:
                # S_out.inf is 1D, S is column vector - reshape S_out.inf to column vector
                inf_reshaped = S_out.inf.reshape(-1, 1)
                sup_reshaped = S_out.sup.reshape(-1, 1)
                result.inf = inf_reshaped + S
                result.sup = sup_reshaped + S
            elif inf_shape != S_shape and len(inf_shape) == 2 and len(S_shape) == 1 and inf_shape[1] == 1:
                # S_out.inf is column vector, S is 1D - reshape S to column vector
                S_reshaped = S.reshape(-1, 1)
                result.inf = S_out.inf + S_reshaped
                result.sup = S_out.sup + S_reshaped
            else:
                # Shapes are compatible or one is scalar
                result.inf = S_out.inf + S
                result.sup = S_out.sup + S
            return result
        
        # Handle zonotope case - convert zonotope to interval first
        if hasattr(S, '__class__') and S.__class__.__name__ == 'Zonotope':
            # Convert zonotope to interval using its interval() method
            S_interval = S.interval()
            result = Interval.__new__(Interval)
            result.precedence = S_out.precedence
            
            # Check if both intervals are vectors (1D or column vectors) vs matrices
            # Only flatten if both are vectors to avoid creating 2D intervals from vector addition
            # Preserve matrix shape if one is a true matrix (shape (m, n) where n > 1)
            S_out_is_vector = (S_out.inf.ndim == 1) or (S_out.inf.ndim == 2 and S_out.inf.shape[1] == 1)
            S_interval_is_vector = (S_interval.inf.ndim == 1) or (S_interval.inf.ndim == 2 and S_interval.inf.shape[1] == 1)
            
            if S_out_is_vector and S_interval_is_vector:
                # Both are vectors - flatten to 1D to avoid creating 2D intervals
                S_out_inf = S_out.inf.flatten() if S_out.inf.ndim > 1 else S_out.inf
                S_out_sup = S_out.sup.flatten() if S_out.sup.ndim > 1 else S_out.sup
                S_inf = S_interval.inf.flatten() if S_interval.inf.ndim > 1 else S_interval.inf
                S_sup = S_interval.sup.flatten() if S_interval.sup.ndim > 1 else S_interval.sup
            else:
                # At least one is a matrix - preserve shapes (they should match)
                S_out_inf = S_out.inf
                S_out_sup = S_out.sup
                S_inf = S_interval.inf
                S_sup = S_interval.sup
            
            result.inf = S_out_inf + S_inf
            result.sup = S_out_sup + S_sup
            return result
            
    except Exception as e:
        # Check whether different dimension of ambient space
        if hasattr(S_out, 'dim') and hasattr(S, 'dim'):
            if S_out.dim() != S.dim():
                raise CORAerror('CORA:dimensionMismatch', 
                               f'Dimension mismatch: {S_out.dim()} vs {S.dim()}')
        
        # Check for empty sets using object methods
        if (S_out.representsa_('emptySet', 1e-9) or 
            (hasattr(S, 'representsa_') and S.representsa_('emptySet', 1e-9))):
            return Interval.empty(S_out.dim())
        
        # Re-raise the original error
        raise e
    
    # If we get here, the operation is not supported
    raise CORAerror('CORA:noops', f'Operation not supported between {type(S_out)} and {type(S)}') 
