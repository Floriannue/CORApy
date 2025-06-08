"""
mtimes - Overloaded '*' operator for intervals

Syntax:
    res = factor1 * factor2
    res = mtimes(factor1, factor2)

Inputs:
    factor1 - interval object, numeric
    factor2 - interval object, numeric, contSet object

Outputs:
    res - interval

Authors: Matthias Althoff (MATLAB)
         Python translation by AI Assistant
Written: 19-June-2015 (MATLAB)
Last update: 04-October-2024 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import Union
from .interval import Interval
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAError

# Optional scipy import for sparse matrix support
try:
    import scipy.sparse as sp
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def mtimes(factor1: Union[Interval, np.ndarray, float, int], 
          factor2: Union[Interval, np.ndarray, float, int]) -> Interval:
    """
    Overloaded '*' operator for intervals
    
    Args:
        factor1: Interval object or numeric
        factor2: Interval object, numeric, or contSet object
        
    Returns:
        res: Interval result
    """
    # Handle contSet cases (not implemented yet, but structure for future)
    if hasattr(factor2, '__class__') and factor2.__class__.__name__ in [
        'polyZonotope', 'zonotope', 'conZonotope', 'zonoBundle']:
        raise CORAError('CORA:noops', f'Operation not supported with {type(factor2)}')
    
    # Other contSet cases not supported
    if (hasattr(factor2, 'precedence') and 
        not isinstance(factor2, Interval) and 
        hasattr(factor2, '__class__')):
        raise CORAError('CORA:noops', f'Operation not supported between {type(factor1)} and {type(factor2)}')
    
    # Convert inputs to intervals if needed
    if not isinstance(factor1, Interval):
        factor1 = _numeric_to_Interval(factor1)
    if not isinstance(factor2, Interval):
        factor2 = _numeric_to_Interval(factor2)
    
    # Scalar case
    if _is_scalar(factor1) and _is_scalar(factor2):
        return _mtimes_scalar(factor1, factor2)
    
    if _is_scalar(factor1):  # factor2 is a matrix
        return _mtimes_scalar_matrix(factor1, factor2)
    
    if _is_scalar(factor2):  # factor1 is a matrix
        return _mtimes_matrix_scalar(factor1, factor2)
    
    # Matrix case
    if not _is_sparse(factor1) and not _is_sparse(factor2):
        return _mtimes_nonsparse(factor1, factor2)
    else:
        return _mtimes_sparse(factor1, factor2)


def _numeric_to_Interval(value):
    """Convert numeric value to interval"""
    if isinstance(value, (int, float)):
        return Interval(np.array([value]))
    elif isinstance(value, np.ndarray):
        return Interval(value, value)
    else:
        raise CORAError('CORA:wrongInput', f'Cannot convert {type(value)} to interval')


def _is_scalar(obj):
    """Check if object represents a scalar"""
    if isinstance(obj, Interval):
        return obj.inf.size == 1
    return np.isscalar(obj)


def _is_sparse(obj):
    """Check if object is sparse"""
    if not HAS_SCIPY:
        return False
    
    if isinstance(obj, Interval):
        return sp.issparse(obj.inf) or sp.issparse(obj.sup)
    return sp.issparse(obj)


def _mtimes_scalar(factor1: Interval, factor2: Interval) -> Interval:
    """Multiply two scalar intervals"""
    # Handle zero cases
    if (_is_zero_Interval(factor1) or _is_zero_Interval(factor2)):
        return Interval(np.array([0.0]))
    
    # Get possible values
    possible_values = [
        factor1.inf.item() * factor2.inf.item(),
        factor1.inf.item() * factor2.sup.item(),
        factor1.sup.item() * factor2.inf.item(),
        factor1.sup.item() * factor2.sup.item()
    ]
    
    # Infimum and supremum
    inf_val = np.min(possible_values)
    sup_val = np.max(possible_values)
    
    return Interval(np.array([inf_val]), np.array([sup_val]))


def _mtimes_scalar_matrix(factor1: Interval, factor2: Interval) -> Interval:
    """Multiply scalar interval with matrix interval"""
    scalar_inf = factor1.inf.item()
    scalar_sup = factor1.sup.item()
    
    if scalar_inf == 0 and scalar_sup == 0:
        # 0 * anything = 0
        return Interval(np.zeros_like(factor2.inf))
    elif scalar_inf >= 0 and scalar_sup >= 0:
        # Positive scalar interval
        return Interval(scalar_inf * factor2.inf, scalar_sup * factor2.sup)
    elif scalar_inf <= 0 and scalar_sup <= 0:
        # Negative scalar interval - when multiplying by negative values, bounds swap
        # scalar_inf * factor2.sup gives the minimum (most negative)
        # scalar_sup * factor2.inf gives the maximum (least negative)
        return Interval(scalar_inf * factor2.sup, scalar_sup * factor2.inf)
    else:
        # Scalar interval contains zero - need to consider all combinations
        products = [
            scalar_inf * factor2.inf,
            scalar_inf * factor2.sup,
            scalar_sup * factor2.inf,
            scalar_sup * factor2.sup
        ]
        
        # Stack all products and find min/max element-wise
        all_products = np.stack(products, axis=-1)
        inf_result = np.min(all_products, axis=-1)
        sup_result = np.max(all_products, axis=-1)
        
        return Interval(inf_result, sup_result)


def _mtimes_matrix_scalar(factor1: Interval, factor2: Interval) -> Interval:
    """Multiply matrix interval with scalar interval"""
    # Check if factor2 is effectively numeric (same inf and sup)
    if np.allclose(factor2.inf, factor2.sup):
        # factor2 is effectively a numeric scalar, use the scalar logic
        scalar_val = factor2.inf.item()
        if scalar_val < 0:
            return Interval(scalar_val * factor1.sup, scalar_val * factor1.inf)
        elif scalar_val > 0:
            return Interval(scalar_val * factor1.inf, scalar_val * factor1.sup)
        else:  # scalar_val == 0
            return Interval(np.zeros_like(factor1.inf))
    else:
        # factor2 is a true interval, use element-wise multiplication
        # This broadcasts the scalar interval across all elements of the matrix
        return _element_wise_multiply(factor1, factor2)


def _element_wise_multiply(factor1: Interval, factor2: Interval) -> Interval:
    """Element-wise multiplication of intervals with broadcasting"""
    # Get all possible products for each element
    products = [
        factor1.inf * factor2.inf,
        factor1.inf * factor2.sup,
        factor1.sup * factor2.inf,
        factor1.sup * factor2.sup
    ]
    
    # Stack and find min/max element-wise
    all_products = np.stack(products, axis=-1)
    
    # Handle NaN cases (0 * inf = 0)
    nan_mask = np.isnan(all_products)
    all_products[nan_mask] = 0
    
    inf_result = np.min(all_products, axis=-1)
    sup_result = np.max(all_products, axis=-1)
    
    return Interval(inf_result, sup_result)


def _mtimes_nonsparse(factor1: Interval, factor2: Interval) -> Interval:
    """Matrix multiplication for non-sparse intervals"""
    # Always perform matrix multiplication (not element-wise)
    # The mtimes function in MATLAB always does matrix multiplication
    f1_inf = factor1.inf
    f1_sup = factor1.sup
    f2_inf = factor2.inf
    f2_sup = factor2.sup
    
    # Ensure 2D arrays for matrix multiplication
    if f1_inf.ndim == 1:
        f1_inf = f1_inf.reshape(1, -1)
        f1_sup = f1_sup.reshape(1, -1)
    if f2_inf.ndim == 1:
        f2_inf = f2_inf.reshape(-1, 1)
        f2_sup = f2_sup.reshape(-1, 1)
    
    # Check dimension compatibility
    if f1_inf.shape[1] != f2_inf.shape[0]:
        raise CORAError('CORA:wrongInput', 
                       f'Matrix dimensions incompatible: {f1_inf.shape} * {f2_inf.shape}')
    
    # Perform matrix multiplication for all combinations
    products = [
        np.dot(f1_inf, f2_inf),
        np.dot(f1_inf, f2_sup),
        np.dot(f1_sup, f2_inf),
        np.dot(f1_sup, f2_sup)
    ]
    
    # Stack and find min/max element-wise
    all_products = np.stack(products, axis=-1)
    
    # Handle NaN cases (0 * inf = 0)
    nan_mask = np.isnan(all_products)
    all_products[nan_mask] = 0
    
    inf_result = np.min(all_products, axis=-1)
    sup_result = np.max(all_products, axis=-1)
    
    # Reshape back to original dimensions if needed
    if factor1.inf.ndim == 1 and factor2.inf.ndim == 1:
        inf_result = inf_result.flatten()
        sup_result = sup_result.flatten()
    
    return Interval(inf_result, sup_result)


def _mtimes_sparse(factor1: Interval, factor2: Interval) -> Interval:
    """Matrix multiplication for sparse intervals"""
    # For sparse matrices, use the slower but more memory-efficient algorithm
    f1_shape = factor1.inf.shape
    f2_shape = factor2.inf.shape
    
    if len(f1_shape) == 1:
        m, k = 1, f1_shape[0]
    else:
        m, k = f1_shape
        
    if len(f2_shape) == 1:
        k2, n = f2_shape[0], 1
    else:
        k2, n = f2_shape
    
    if k != k2:
        raise CORAError('CORA:wrongInput', 
                       f'Matrix dimensions incompatible: {factor1.inf.shape} * {factor2.inf.shape}')
    
    # Preallocate output bounds
    res_inf = np.zeros((m, n))
    res_sup = np.zeros((m, n))
    
    # Create temporary interval for row operations
    temp_interval = Interval(np.zeros(k), np.zeros(k))
    
    for i in range(m):
        # Get i-th row
        temp_interval.inf = factor1.inf[i, :]
        temp_interval.sup = factor1.sup[i, :]
        
        # Multiply row with matrix
        row_result = _mtimes_vector_matrix(temp_interval, factor2)
        res_inf[i, :] = row_result.inf
        res_sup[i, :] = row_result.sup
    
    return Interval(res_inf, res_sup)


def _mtimes_vector_matrix(vector_interval: Interval, matrix_interval: Interval) -> Interval:
    """Multiply vector interval with matrix interval"""
    k, n = matrix_interval.inf.shape
    
    # [k] .* [k, n] = [k, n]
    v_inf = vector_interval.inf[:, np.newaxis]
    v_sup = vector_interval.sup[:, np.newaxis]
    
    products = [
        v_inf * matrix_interval.inf,
        v_inf * matrix_interval.sup,
        v_sup * matrix_interval.inf,
        v_sup * matrix_interval.sup
    ]
    
    # Stack and handle NaN
    all_products = np.stack(products, axis=-1)
    nan_mask = np.isnan(all_products)
    all_products[nan_mask] = 0
    
    # Sum over k dimension
    sums = np.sum(all_products, axis=0)  # [n, 4]
    
    inf_result = np.min(sums, axis=-1)
    sup_result = np.max(sums, axis=-1)
    
    return Interval(inf_result, sup_result)


def _is_zero_Interval(obj: Interval) -> bool:
    """Check if interval represents zero"""
    return np.allclose(obj.inf, 0) and np.allclose(obj.sup, 0) 
