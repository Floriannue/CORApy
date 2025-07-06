from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.matlab.validate.check.equal_dim_check import equal_dim_check
import numpy as np


def boundaryPoint(I, dir_, *varargin):
    #
    # Description:
    #   computes the point on the boundary of an interval along a
    #   given direction, starting from a given start point, or, by default,
    #   from the center of the set; for unbounded intervals, a start point
    #   must be provided; any given start point must be contained in the set;
    #   note that the vector may immediately reach the boundary of degenerate
    #   intervals
    #
    if not np.any(dir_):
        raise CORAerror('wrongValue', 'second', "Direction must be a non-zero vector.")

    # check number of input arguments
    if len(varargin) > 1:
        raise CORAerror('CORA:tooManyInputArgs', 3)

    if len(varargin) >= 1:
        startPoint = varargin[0]
    else:
        startPoint = I.center()

    if np.any(np.isnan(startPoint)):
        raise CORAerror('wrongValue', 'third', 'For unbounded sets, a start point must be provided')

    # check for dimensions
    equal_dim_check(I, dir_)
    equal_dim_check(I, startPoint)

    # read out dimension
    dim_val = I.dim()

    # for empty sets, return empty
    if I.is_empty():
        shape = [dim_val] if isinstance(dim_val, int) else dim_val
        shape.append(0)
        return np.zeros(shape)

    if not I.contains(startPoint):
        raise CORAerror('wrongValue', 'third', 'Start point must be contained in the set.')

    # MATLAB logic: calculate bound and ratio from the UN-SHIFTED interval
    rad = I.rad()
    sign_dir = np.sign(dir_)
    
    # for positive/negative values, extract upper/lower bound
    # bound = infimum(I) + 2*rad(I) .* max(sign_dir,zeros([n,1]));
    max_term = np.maximum(sign_dir, np.zeros_like(sign_dir, dtype=float))
    
    # Masked multiplication to avoid inf*0 warning
    term = np.zeros_like(rad, dtype=float)
    safe_mask = ~np.isinf(rad) | (sign_dir != 0)
    term[safe_mask] = 2 * rad[safe_mask] * max_term[safe_mask]
    
    bound = I.inf + term
    
    # compute factor of limiting dimension
    non_zero_dir_indices = dir_ != 0
    if not np.any(non_zero_dir_indices):
        ratio = 0.0
    else:
        # Subtract startPoint component-wise before division
        bound_shifted = bound - startPoint
        ratios = bound_shifted[non_zero_dir_indices] / dir_[non_zero_dir_indices]
        
        # take minimum positive value (excluding -Inf and negative)
        positive_ratios = ratios[ratios >= 0]
        finite_ratios = positive_ratios[np.isfinite(positive_ratios)]

        if finite_ratios.size > 0:
            ratio = np.min(finite_ratios)
        else:
            # This case can happen if the only path is in an unbounded direction
            ratio = np.inf

    # multiply (normalized) direction with that factor
    x = dir_ * ratio + startPoint

    # dimensions with infinite width (unless direction is zero in that
    # direction!)
    d_inf = np.isinf(rad)
    inf_dirs = d_inf & (dir_ != 0)
    x[inf_dirs] = np.inf * sign_dir[inf_dirs]

    return x 