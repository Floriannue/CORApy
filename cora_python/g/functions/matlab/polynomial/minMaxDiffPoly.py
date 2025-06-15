"""
minMaxDiffPoly - compute the maximum and the minimum difference between 
    two polynomials on the given domain: min/max_x p_1(x) - p_2(x)

Syntax:
    diffl, diffu = minMaxDiffPoly(coeffs1, coeffs2, l, u)

Inputs:
    coeffs1 - coefficients of first polynomial
    coeffs2 - coefficients of second polynomial
    l - lower bound of domain
    u - upper bound of domain

Outputs:
    diffl - minimum difference
    diffu - maximum difference

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: -

Authors: Tobias Ladner
         Python translation by AI Assistant
Written: 26-May-2023
Last update: ---
Last revision: ---
"""

import numpy as np
from typing import Tuple
from cora_python.g.functions.matlab.polynomial.fpolyder import fpolyder


def minMaxDiffPoly(coeffs1: np.ndarray, coeffs2: np.ndarray, l: float, u: float) -> Tuple[float, float]:
    """
    Compute the maximum and minimum difference between two polynomials on the given domain.
    
    Args:
        coeffs1: Coefficients of first polynomial (highest degree first)
        coeffs2: Coefficients of second polynomial (highest degree first)
        l: Lower bound of domain
        u: Upper bound of domain
        
    Returns:
        diffl: Minimum difference p_1(x) - p_2(x) on [l, u]
        diffu: Maximum difference p_1(x) - p_2(x) on [l, u]
    """
    # Convert inputs to numpy arrays
    coeffs1 = np.array(coeffs1)
    coeffs2 = np.array(coeffs2)
    
    # Compute difference polynomial: p_1(x) - p_2(x)
    max_len = max(len(coeffs1), len(coeffs2))
    p = np.zeros(max_len)
    
    # Pad coefficients to same length and compute difference
    if len(coeffs1) < max_len:
        coeffs1_padded = np.concatenate([np.zeros(max_len - len(coeffs1)), coeffs1])
    else:
        coeffs1_padded = coeffs1
        
    if len(coeffs2) < max_len:
        coeffs2_padded = np.concatenate([np.zeros(max_len - len(coeffs2)), coeffs2])
    else:
        coeffs2_padded = coeffs2
    
    p = coeffs1_padded - coeffs2_padded
    
    # Determine extreme points
    dp = fpolyder(p)
    
    # Find roots of derivative
    if len(dp) > 1:  # Only if derivative is not constant
        dp_roots = np.roots(dp)
        # Filter imaginary roots (keep only real roots)
        dp_roots = dp_roots[np.abs(np.imag(dp_roots)) < 1e-10].real
        # Filter roots within domain (l, u)
        dp_roots = dp_roots[(dp_roots > l) & (dp_roots < u)]
    else:
        dp_roots = np.array([])
    
    # Extrema include boundary points and critical points
    extrema = np.concatenate([[l], dp_roots, [u]])
    
    # Evaluate polynomial at all extrema
    diff = np.polyval(p, extrema)
    
    # Compute final min/max difference
    diffl = np.min(diff)
    diffu = np.max(diff)
    
    return diffl, diffu 