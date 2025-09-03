"""
minMaxDiffOrder - compute the maximum and the minimum difference between the activation
function and a polynomial fit

Syntax:
    [diffl,diffu] = nnHelper.minMaxDiffOrder(coeffs, l, u, f, der1)

Inputs:
    coeffs - coefficients of polynomial
    l - lower bound of input domain
    u - upper bound of input domain
    f - function handle of activation function
    der1 - bounds for derivative of activation functions

Outputs:
    [diffl,diffu] - interval bounding the lower and upper error

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: -

Authors:       Tobias Ladner
Written:       28-March-2022
Last update:   31-August-2022 (adjust tol)
               30-May-2023 (output bounds)
               02-May-2025 (added maxPoints)
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import Union, Callable
from .getDerInterval import getDerInterval

def minMaxDiffOrder(coeffs: np.ndarray, l: Union[float, np.ndarray], u: Union[float, np.ndarray], 
                   f: Callable, der1l: Union[float, np.ndarray], der1u: Union[float, np.ndarray]) -> tuple:
    """
    Compute the maximum and the minimum difference between the activation
    function and a polynomial fit
    
    Args:
        coeffs: coefficients of polynomial
        l: lower bound of input domain
        u: upper bound of input domain
        f: function handle of activation function
        der1l, der1u: bounds for derivative of activation functions
        
    Returns:
        Tuple of (diffl, diffu) interval bounding the lower and upper error
    """
    tol = 1e-4
    minPoints = 1e4
    maxPoints = 5e9  # requires 40GB
    
    if l == u:
        # compute exact result directly
        diff = f(l)
        yp = np.polyval(coeffs, l)  # coeffs already in descending order
        diff = diff - yp
        diffl = diff
        diffu = diff
        return diffl, diffu
    
    # calculate bounds for derivative of polynomial
    der2l, der2u = getDerInterval(coeffs, l, u)
    
    # der = der1 - -der2; % '-' as we calculate f(x) - p(x)
    der = np.max(np.abs([
        der1l - -der2l,
        der1l - -der2u,
        der1u - -der2l,
        der1u - -der2u
    ]))
    
    # determine number of points to sample
    dx = tol / der
    reqPoints = int(np.ceil((u - l) / dx))
    numPoints = min(max(reqPoints, minPoints), maxPoints)
    
    # re-calculate tolerance with number of used points
    dx = (u - l) / numPoints
    tol = der * dx
    
    # sample points
    x = np.linspace(l, u, numPoints)
    x = np.concatenate([[l], x, [u]])  # add l, u in case x is empty (der = 0)
    diff = f(x) - np.polyval(coeffs, x)  # coeffs already in descending order
    
    # find bounds
    diffl = np.min(diff) - tol
    diffu = np.max(diff) + tol
    
    return diffl, diffu
