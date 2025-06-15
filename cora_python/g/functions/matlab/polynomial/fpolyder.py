"""
fpolyder - computes the derivative of a polynomial (faster than polyder)

Syntax:
    dp = fpolyder(p)

Inputs:
    p - polynomial coefficients (highest degree first)

Outputs:
    dp - derivative polynomial coefficients

Example:
    p = [1, 2, 1]  # x^2 + 2x + 1
    dp = fpolyder(p)  # 2x + 2

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: polyder

Authors: Tobias Ladner
         Python translation by AI Assistant
Written: 26-May-2023
Last update: ---
Last revision: ---
"""

import numpy as np


def fpolyder(p: np.ndarray) -> np.ndarray:
    """
    Computes the derivative of a polynomial (faster than polyder).
    
    Args:
        p: Polynomial coefficients (highest degree first)
        
    Returns:
        dp: Derivative polynomial coefficients
    """
    p = np.array(p, dtype=float)
    
    # Handle edge cases
    if len(p) == 0:
        return np.array([0])
    
    if len(p) == 1:
        return np.array([0])
    
    # Compute derivative coefficients
    n = len(p)
    dp = np.zeros(n - 1)
    
    for i in range(n - 1):
        dp[i] = p[i] * (n - 1 - i)
    
    # Remove leading zeros
    if len(dp) > 0:
        # Find first non-zero coefficient
        first_nonzero = 0
        while first_nonzero < len(dp) and abs(dp[first_nonzero]) < 1e-14:
            first_nonzero += 1
        
        if first_nonzero == len(dp):
            # All coefficients are zero
            dp = np.array([0])
        else:
            # Remove leading zeros
            dp = dp[first_nonzero:]
    
    return dp 