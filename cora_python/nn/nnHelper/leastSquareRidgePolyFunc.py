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

def leastSquareRidgePolyFunc(x: Union[float, np.ndarray], y: Union[float, np.ndarray], n: int, lambda_reg: float = 0.001) -> np.ndarray:
    """
    Determine the optimal polynomial function fit using ridge regression
    
    Args:
        x: x values
        y: y values
        n: polynomial order
        lambda_reg: coefficient of Tikhonov regularization, default to 0.001
        
    Returns:
        coeffs: coefficients of resulting polynomial
    """
    # Convert to numpy arrays (no validation like MATLAB)
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Match MATLAB exactly: A = x'.^(0:n)
    A = np.column_stack([x ** i for i in range(n + 1)])
    
    # Match MATLAB exactly: coeffs = (A' * A + lambda * eye(n+1)) \ A' * y'
    # Ridge regression: (A^T A + λI)^(-1) A^T y
    ATA = A.T @ A
    ATy = A.T @ y
    regularization = lambda_reg * np.eye(n + 1)
    
    coeffs = np.linalg.solve(ATA + regularization, ATy)
    
    # Match MATLAB exactly: coeffs = fliplr(coeffs')
    coeffs = np.flip(coeffs)
    
    return coeffs
