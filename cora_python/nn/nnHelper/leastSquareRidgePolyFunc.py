"""
leastSquareRidgePolyFunc - determine the optimal polynomial function fit using ridge regression

Syntax:
    coeffs = nnHelper.leastSquareRidgePolyFunc(x, y, n)

Inputs:
    x - x values
    y - y values
    n - polynomial order

Outputs:
    coeffs - coefficients of resulting polynomial

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: leastSquarePolyFunc

Authors:       Niklas Kochdumper, Tobias Ladner
Written:       17-September-2021
Last update:   ---
Last revision: 28-March-2022 (TL)
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import Union

def leastSquareRidgePolyFunc(x: Union[float, np.ndarray], y: Union[float, np.ndarray], n: int) -> np.ndarray:
    """
    Determine the optimal polynomial function fit using ridge regression
    
    Args:
        x: x values
        y: y values
        n: polynomial order
        
    Returns:
        coeffs: coefficients of resulting polynomial
    """
    # For now, use regular polynomial fit like MATLAB
    # TODO: Implement actual ridge regression
    from .leastSquarePolyFunc import leastSquarePolyFunc
    return leastSquarePolyFunc(x, y, n)
