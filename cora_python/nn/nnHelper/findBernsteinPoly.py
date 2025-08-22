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
    # scale domain to [0,1] and compute polynomial according to
    # https://en.wikipedia.org/wiki/Bernstein_polynomial#Approximating_continuous_functions
    
    # scale domain
    def f_norm(x):
        return f(x * (u - l) + l)
    
    # compute bernstein polynomial
    coeffs_norm = np.zeros(n + 1)
    for v in range(n + 1):
        coeffs_norm = coeffs_norm + f_norm(v / n) * aux_b(v, n)
    
    # transform polynomial back to [l,u] domain
    coeffs = np.zeros(n + 1)
    P = np.array([1])  # Pascal's triangle
    for i in range(n + 1):
        coeffs[-(i + 1):] = coeffs[-(i + 1):] + \
            coeffs_norm[-(i + 1)] * P * (l / (u - l)) ** np.arange(i + 1) * (1 / (u - l)) ** np.arange(i, -1, -1)
        
        # prepare for next iteration
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
    coeffs = np.zeros(n + 1)
    for l in range(v, n + 1):
        coeffs[-(l + 1)] = comb(n, l) * comb(l, v) * (-1) ** (l - v)
    
    return coeffs
