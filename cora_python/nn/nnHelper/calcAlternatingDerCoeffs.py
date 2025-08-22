"""
calcAlternatingDerCoeffs - calculate coefficients for polynomial using
   'throw-catch'

This is a Python translation of the MATLAB CORA implementation.

Authors: MATLAB: Tobias Ladner
         Python: AI Assistant
"""

import numpy as np
from typing import Any


def calcAlternatingDerCoeffs(l: float, u: float, order: int, layer: Any) -> np.ndarray:
    """
    Calculate coefficients for polynomial using 'throw-catch'.
    
    Args:
        l: lower bound of input domain
        u: upper bound of input domain
        order: order of the resulting polynomial
        layer: nnSShapeLayer
        
    Returns:
        coeffs: output of the neural network
        
    See also: -
    """
    if layer.df(l) > layer.df(u):
        coeffs = calcAlternatingDerCoeffs(u, l, order, layer)
        return coeffs
    
    # init
    X = []
    Y = []
    
    # get function value
    exponents = np.flip(np.arange(order + 1))
    coeffs_full = np.ones_like(exponents)
    coeffs = np.ones_like(exponents)
    
    xi = coeffs_full * (l ** exponents)
    yi = np.array([layer.f(l)])
    
    X.append(xi)
    Y.append(yi)
    
    # get first derivative
    exponents = np.maximum(0, exponents - 1)
    coeffs = np.polyder(coeffs)
    coeffs_full = np.concatenate([coeffs, np.zeros(1 + order - len(coeffs))])
    
    xi = coeffs_full * (l ** exponents)
    yi = np.array([layer.df(l)])
    
    X.append(xi)
    Y.append(yi)
    
    for i in range(2, int(np.ceil((order + 1) / 2)) + 1):
        # add each higher derivative
        xi = coeffs_full * (u ** exponents)
        df_i = layer.getDf(i)
        yi = np.array([df_i(u)])
        
        X.append(xi)
        Y.append(yi)
        
        exponents = np.maximum(0, exponents - 1)
        coeffs = np.polyder(coeffs)
        coeffs_full = np.concatenate([coeffs, np.zeros(1 + order - len(coeffs))])
        
        xi = coeffs_full * (l ** exponents)
        yi = np.array([df_i(l)])
        
        X.append(xi)
        Y.append(yi)
    
    # Convert lists to numpy arrays
    X = np.array(X)
    Y = np.array(Y)
    
    # determine if X is square (and thus also invertible by construction)
    if X.shape[0] == X.shape[1]:
        # X\Y is more efficient than inv(X) * Y
        # https://de.mathworks.com/help/matlab/ref/inv.html#bu6sfy8-1
        coeffs = np.linalg.solve(X, Y)
    else:
        coeffs = np.linalg.pinv(X) @ Y
    
    coeffs = coeffs.T
    return coeffs
