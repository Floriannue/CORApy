"""
fpolyder - differentiate polynomial

Syntax:
    dp = nnHelper.fpolyder(p)

Inputs:
    p - coefficients of polynomial

Outputs:
    dp - coefficients of derivative polynomial

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: -

Authors:       Tobias Ladner
Written:       28-March-2022
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import Union

def fpolyder(p: Union[list, np.ndarray]) -> np.ndarray:
    """
    Differentiate polynomial - same as polyder but faster
    
    Args:
        p: coefficients of polynomial (highest degree first)
        
    Returns:
        dp: coefficients of derivative polynomial
    """
    if p is None:
        return np.array([0])
    
    p = np.array(p, dtype=float)
    
    # Handle edge cases
    if len(p) == 0:
        return np.array([0])
    
    if len(p) == 1:
        return np.array([0])
    
    # MATLAB: order = length(p)-1; dp = (order:-1:1) .* p(1:end-1);
    # p = [a_n, a_{n-1}, ..., a_1, a_0] (highest degree first)
    # dp = [n*a_n, (n-1)*a_{n-1}, ..., 1*a_1]
    order = len(p) - 1
    dp = np.array([(order - i) * p[i] for i in range(order)])
    
    return dp