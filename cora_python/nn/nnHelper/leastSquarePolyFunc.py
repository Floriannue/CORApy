"""
leastSquarePolyFunc - determine the optimal polynomial function fit that 
   minimizes the squared distance to the data points

Syntax:
    coeffs = nnHelper.leastSquarePolyFunc(x, y, n)

Inputs:
    x - x values
    y - y values
    n - polynomial order

Outputs:
    coeffs - coefficients of resulting polynomial

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: -

Authors:       Niklas Kochdumper, Tobias Ladner
Written:       17-September-2021
Last update:   ---
Last revision: 28-March-2022 (TL)
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import Union

def leastSquarePolyFunc(x: Union[float, np.ndarray], y: Union[float, np.ndarray], n: int) -> np.ndarray:
    """
    Determine the optimal polynomial function fit that minimizes the squared distance to the data points
    
    Args:
        x: x values
        y: y values
        n: polynomial order
        
    Returns:
        coeffs: coefficients of resulting polynomial
    """
    # Match MATLAB exactly: A = x'.^(0:n)
    A = np.column_stack([x ** i for i in range(n + 1)])
    
    # Match MATLAB exactly: coeffs = pinv(A) * y'
    coeffs = np.linalg.pinv(A) @ y
    
    # Match MATLAB exactly: coeffs = fliplr(coeffs')
    coeffs = np.flip(coeffs)
    
    return coeffs
