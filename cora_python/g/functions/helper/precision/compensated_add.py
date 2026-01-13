"""
Compensated addition for improved floating-point precision
"""
import numpy as np
from typing import Union


def compensated_add(a: Union[float, np.ndarray], b: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Compensated addition (Kahan-Babuska algorithm) for improved precision
    
    This reduces rounding errors when adding two numbers or arrays.
    
    Args:
        a: First operand
        b: Second operand
        
    Returns:
        Sum with improved precision
    """
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        # Element-wise compensated addition for arrays
        result = np.zeros_like(a)
        for i in np.ndindex(a.shape):
            # Kahan-Babuska algorithm
            sum_val = a[i] + b[i]
            err = b[i] - (sum_val - a[i])
            result[i] = sum_val + err
        return result
    else:
        # Scalar addition
        sum_val = a + b
        err = b - (sum_val - a)
        return sum_val + err
