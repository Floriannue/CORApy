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
    Differentiate polynomial
    
    Args:
        p: coefficients of polynomial
        
    Returns:
        dp: coefficients of derivative polynomial
    """
    if len(p) <= 1:
        return np.array([0])
    
    # p = [a_n, a_{n-1}, ..., a_1, a_0]
    # dp = [n*a_n, (n-1)*a_{n-1}, ..., 1*a_1]
    n = len(p) - 1
    dp = np.array([(n - i) * p[i] for i in range(n)])
    
    return dp
