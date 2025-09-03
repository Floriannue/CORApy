"""
findBernsteinPoly - finds a polynomial approximating f on [l,u] using
   bernstein polynomials

This is a Python translation of the MATLAB CORA implementation.

Authors: MATLAB: Tobias Ladner
         Python: AI Assistant
"""

import numpy as np
from typing import Callable
from scipy.special import comb


def findBernsteinPoly(f: Callable, l: float, u: float, n: int) -> np.ndarray:
    """
    Find a polynomial approximating f on [l,u] using Bernstein polynomials.
    
    Args:
        f: function handle to approximate
        l, u: bounds of domain
        n: order of the polynomial
        
    Returns:
        coeffs: coefficients of the polynomial
        
    See also: -
    """
    # Input validation
    if l >= u:
        raise ValueError("l must be < u")
    if n < 0:
        raise ValueError("n must be non-negative")
    if not isinstance(n, int):
        raise ValueError("n must be an integer")
    
    # Handle edge case n=0
    if n == 0:
        # For order 0, return constant function value at midpoint
        midpoint = (l + u) / 2
        return np.array([f(midpoint)])
    
    # scale domain to [0,1] and compute polynomial according to
    # https://en.wikipedia.org/wiki/Bernstein_polynomial#Approximating_continuous_functions
    
    # scale domain
    def f_norm(x):
        return f(x * (u - l) + l)
    
    # compute bernstein polynomial - match MATLAB exactly
    coeffs_norm = np.zeros(n + 1)
    for v in range(n + 1):
        coeffs_norm = coeffs_norm + f_norm(v / n) * aux_b(v, n)
    
    # transform polynomial back to [l,u] domain - match MATLAB exactly
    coeffs = np.zeros(n + 1)
    P = np.array([1])  # Pascal's triangle
    for i in range(n + 1):
        # MATLAB: coeffs(end-i:end) = coeffs(end-i:end) + ...
        #         coeffs_norm(end-i) .* P .* (l/(u-l)).^(0:i) .* (1/(u-l)).^(i:-1:0);
        coeffs[-(i + 1):] = coeffs[-(i + 1):] + \
            coeffs_norm[-(i + 1)] * P * (l / (u - l)) ** np.arange(i + 1) * (1 / (u - l)) ** np.arange(i, -1, -1)
        
        # prepare for next iteration - match MATLAB exactly
        # MATLAB: P = [1 P(2:end)+P(1:end-1) 1];
        if i < n:
            P = np.concatenate([[1], P[1:] + P[:-1], [1]])
    
    return coeffs


def aux_b(v: int, n: int) -> np.ndarray:
    """
    Returns the bernstein polynomial b_{v,n}(x) in normal form.
    
    Args:
        v: parameter
        n: order
        
    Returns:
        coeffs: coefficients
    """
    # Match MATLAB exactly: coeffs(end-l) = nchoosek(n,l)*nchoosek(l,v)*(-1)^(l-v);
    coeffs = np.zeros(n + 1)
    for l in range(v, n + 1):
        coeffs[-(l + 1)] = comb(n, l) * comb(l, v) * (-1) ** (l - v)
    
    return coeffs
