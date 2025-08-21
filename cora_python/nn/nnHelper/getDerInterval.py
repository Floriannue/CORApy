"""
getDerInterval - compute bounds for the derivative of a polynomial

Syntax:
    [derl,deru] = nnHelper.getDerInterval(coeffs, l, u)

Inputs:
    coeffs - coefficients of polynomial
    l - lower bound of input domain
    u - upper bound of input domain

Outputs:
    [derl,deru] - interval bounding the derivative

Other m-files required: none
Subfunctions: fpolyder
MAT-files required: none

See also: fpolyder

Authors:       Tobias Ladner
Written:       28-March-2022
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import Union
from .fpolyder import fpolyder

def getDerInterval(coeffs: np.ndarray, l: Union[float, np.ndarray], u: Union[float, np.ndarray]) -> tuple:
    """
    Compute bounds for the derivative of a polynomial
    
    Args:
        coeffs: coefficients of polynomial
        l: lower bound of input domain
        u: upper bound of input domain
        
    Returns:
        Tuple of (derl, deru) interval bounding the derivative
    """
    # compute derivative
    dp = fpolyder(coeffs)
    
    # find roots of derivative
    roots = np.roots(dp)
    
    # filter real roots
    real_roots = roots[np.abs(roots.imag) < 1e-10].real
    
    # add boundary points
    points = np.concatenate([[l], real_roots, [u]])
    
    # evaluate derivative at all points
    values = np.polyval(dp[::-1], points)  # Reverse for np.polyval
    
    # find bounds
    derl = np.min(values)
    deru = np.max(values)
    
    return derl, deru
