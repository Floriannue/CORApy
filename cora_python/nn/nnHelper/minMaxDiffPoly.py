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
        coeffs1: coefficients of first polynomial (ascending order like MATLAB)
        coeffs2: coefficients of second polynomial (ascending order like MATLAB)
        l: lower bound of input domain
        u: upper bound of input domain
        
    Returns:
        Tuple of (diffl, diffu) interval bounding the lower and upper error
    """
    # MATLAB style: pad to same length, subtract
    # MATLAB: p = zeros(1,max(length(coeffs1),length(coeffs2)));
    #         p(end-length(coeffs1)+1:end) = coeffs1;
    #         p(end-length(coeffs2)+1:end) = p(end-length(coeffs2)+1:end)-coeffs2;
    max_len = max(len(coeffs1), len(coeffs2))
    p = np.zeros(max_len)
    p[-len(coeffs1):] = coeffs1
    p[-len(coeffs2):] = p[-len(coeffs2):] - coeffs2
    
    # compute derivative
    dp = fpolyder(p)
    
    # find roots of derivative
    roots = np.roots(dp)
    
    # filter real roots
    real_roots = roots[np.abs(roots.imag) < 1e-10].real
    
    # filter roots within domain
    real_roots = real_roots[(real_roots > l) & (real_roots < u)]
    
    # add boundary points
    points = np.concatenate([[l], real_roots, [u]])
    
    # evaluate difference at all points
    # p is in ascending order (like MATLAB), and np.polyval handles this correctly
    values = np.polyval(p, points)
    
    # find bounds
    diffl = np.min(values)
    diffu = np.max(values)
    
    return diffl, diffu
