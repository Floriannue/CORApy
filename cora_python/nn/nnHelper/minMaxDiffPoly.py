"""
minMaxDiffPoly - compute the maximum and the minimum difference between two polynomials

Syntax:
    [diffl,diffu] = nnHelper.minMaxDiffPoly(coeffs1, coeffs2, l, u)

Inputs:
    coeffs1 - coefficients of first polynomial
    coeffs2 - coefficients of second polynomial
    l - lower bound of input domain
    u - upper bound of input domain

Outputs:
    [diffl,diffu] - interval bounding the lower and upper error

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

def minMaxDiffPoly(coeffs1: np.ndarray, coeffs2: np.ndarray, 
                   l: Union[float, np.ndarray], u: Union[float, np.ndarray]) -> tuple:
    """
    Compute the maximum and the minimum difference between two polynomials
    
    Args:
        coeffs1: coefficients of first polynomial
        coeffs2: coefficients of second polynomial
        l: lower bound of input domain
        u: upper bound of input domain
        
    Returns:
        Tuple of (diffl, diffu) interval bounding the lower and upper error
    """
    # compute difference polynomial
    p = coeffs1 - coeffs2
    
    # compute derivative
    dp = fpolyder(p)
    
    # find roots of derivative
    roots = np.roots(dp)
    
    # filter real roots
    real_roots = roots[np.abs(roots.imag) < 1e-10].real
    
    # add boundary points
    points = np.concatenate([[l], real_roots, [u]])
    
    # evaluate difference at all points
    values = np.polyval(p[::-1], points)  # Reverse for np.polyval
    
    # find bounds
    diffl = np.min(values)
    diffu = np.max(values)
    
    return diffl, diffu
