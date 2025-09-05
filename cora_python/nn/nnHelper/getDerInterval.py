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
from cora_python.g.functions.matlab.polynomial.fpolyder import fpolyder


def getDerInterval(coeffs: np.ndarray, l: float, u: float) -> tuple[float, float]:
    """
    Compute the maximum and the minimum derivative of the given polynomial.
    
    Args:
        coeffs: coefficients of polynomial
        l: lower bound of input domain
        u: upper bound of input domain
        
    Returns:
        derl, deru: interval bounding the derivative within [l,u]
        
    See also: -
    """
    # find extreme points of derivative of polynomial
    p = coeffs
    dp = fpolyder(p)
    dp2 = fpolyder(dp)
    
    # Find roots of second derivative (critical points of first derivative)
    dp2_roots = np.roots(dp2)
    # Filter imaginary roots (keep only real roots)
    dp2_roots = dp2_roots[np.imag(dp2_roots) == 0]
    
    # evaluate extreme points of derivative
    points = np.concatenate([[l], dp2_roots, [u]])
    # Filter points within bounds [l, u]
    points = points[(l <= points) & (points <= u)]
    
    # Evaluate derivative at these points
    dp_y = np.polyval(dp, points)
    
    derl = np.min(dp_y)
    deru = np.max(dp_y)
    
    return derl, deru
