"""
and_ - intersection of two intervals

Syntax:
    I_res = and_(I1, I2)
    I_res = and_(I1, I2, method)

Inputs:
    I1 - interval object
    I2 - interval object or other contSet
    method - method for intersection ('exact' by default)

Outputs:
    I_res - intersection of the intervals

Authors: Matthias Althoff (MATLAB)
         Python translation by AI Assistant
Written: 19-June-2015 (MATLAB)
Python translation: 2025
"""

import numpy as np

from .interval import Interval


def and_(I1: Interval, I2: Interval, method: str = 'exact') -> Interval:
    """
    Compute intersection of two intervals
    
    Args:
        I1: First interval object
        I2: Second interval object or other contSet
        method: Method for intersection (default: 'exact')
        
    Returns:
        Intersection of the intervals
    """
    
    # Convert I2 to interval if it's not already
    if not isinstance(I2, Interval) :
        if hasattr(I2, 'interval'):
            I2 = I2.interval()
        else:
            raise ValueError("Cannot compute intersection: second argument is not convertible to interval")
    
    # Handle empty intervals
    if I1.inf.size == 0 or I2.inf.size == 0:
        return Interval.empty(max(I1.dim() if I1.inf.size > 0 else 0, 
                                 I2.dim() if I2.inf.size > 0 else 0))
    
    # Check dimension compatibility
    if I1.dim() != I2.dim():
        raise ValueError(f"Interval dimensions must match: {I1.dim()} vs {I2.dim()}")
    
    # Compute intersection bounds
    # MATLAB: lb = max(I.inf, S.inf); ub = min(I.sup, S.sup);
    inf_result = np.maximum(I1.inf, I2.inf)
    sup_result = np.minimum(I1.sup, I2.sup)
    
    # Check if intersection is empty
    # MATLAB: tmp = lb - ub; if all(tmp <= eps, 'all') then not empty, else empty
    # Note: MATLAB's logic checks if tmp <= eps. For valid intervals, lb <= ub, so tmp <= 0 <= eps.
    # For invalid intervals (lb > ub), tmp > 0, and if tmp > eps, it's empty.
    tmp = inf_result - sup_result
    eps = np.finfo(float).eps
    if np.all(tmp <= eps):
        # Not empty - return intersection using MATLAB's logic
        # MATLAB: res = interval(min([lb,ub],[],2),max([lb,ub],[],2));
        # This takes min/max along dimension 2 (columns) of the matrix [lb,ub]
        # For 1D: [lb, ub] is a row vector, min/max along dim 2 gives min/max of the row
        # For multi-D: need to handle properly
        
        # Handle scalar/1D case
        if inf_result.ndim == 0:
            # Scalar
            lb_val = float(inf_result)
            ub_val = float(sup_result)
            res_inf = min(lb_val, ub_val)
            res_sup = max(lb_val, ub_val)
            return Interval(res_inf, res_sup)
        elif inf_result.ndim == 1:
            # 1D array - MATLAB's min([lb,ub],[],2) for row vector [lb, ub]
            # For row vector, dim 2 is along columns, so min of [lb, ub] is min(lb, ub) element-wise
            res_inf = np.minimum(inf_result, sup_result)
            res_sup = np.maximum(inf_result, sup_result)
            return Interval(res_inf, res_sup)
        else:
            # Multi-dimensional case
            # MATLAB: min([lb,ub],[],2) - concatenate lb and ub along dim 2, then min along dim 2
            # For 2D: [lb, ub] creates a matrix with lb and ub as columns, min along dim 2 gives element-wise min
            res_inf = np.minimum(inf_result, sup_result)
            res_sup = np.maximum(inf_result, sup_result)
            return Interval(res_inf, res_sup)
    else:
        # Empty intersection
        return Interval.empty(I1.dim()) 
