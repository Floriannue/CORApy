"""
Kahan summation algorithm for improved floating-point precision
"""
import numpy as np
from typing import Union


def kahan_sum(arr: np.ndarray, axis: int = None) -> Union[float, np.ndarray]:
    """
    Kahan summation algorithm for improved floating-point precision
    
    This algorithm reduces rounding errors in summation by tracking lost low-order bits.
    
    Args:
        arr: Input array
        axis: Axis along which to sum (None for all elements)
        
    Returns:
        Sum with improved precision
    """
    if axis is None:
        # Sum all elements
        sum_val = 0.0
        c = 0.0  # Running compensation for lost low-order bits
        for x in arr.flatten():
            y = x - c
            t = sum_val + y
            c = (t - sum_val) - y
            sum_val = t
        return sum_val
    else:
        # Sum along specified axis
        if arr.ndim == 1:
            return kahan_sum(arr, axis=None)
        
        # For 2D arrays, sum along axis
        if axis == 0:
            # Sum along rows (axis 0)
            result = np.zeros(arr.shape[1])
            for j in range(arr.shape[1]):
                result[j] = kahan_sum(arr[:, j], axis=None)
            return result
        elif axis == 1:
            # Sum along columns (axis 1)
            result = np.zeros(arr.shape[0])
            for i in range(arr.shape[0]):
                result[i] = kahan_sum(arr[i, :], axis=None)
            return result
        else:
            # Fallback to numpy sum for other axes
            return np.sum(arr, axis=axis)


def kahan_sum_abs(arr: np.ndarray, axis: int = None) -> Union[float, np.ndarray]:
    """
    Kahan summation of absolute values
    
    Args:
        arr: Input array
        axis: Axis along which to sum (None for all elements)
        
    Returns:
        Sum of absolute values with improved precision
    """
    return kahan_sum(np.abs(arr), axis=axis)
