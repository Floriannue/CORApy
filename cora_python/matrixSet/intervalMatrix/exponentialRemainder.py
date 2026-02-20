"""
exponentialRemainder - returns the remainder of the exponential matrix

Syntax:
    E = exponentialRemainder(intMat, maxOrder)

Inputs:
    intMat - intervalMatrix object
    maxOrder - maximum order of Taylor series

Outputs:
    E - remainder of exponential

Authors:       Matthias Althoff (MATLAB)
              Python translation: 2025
"""

import math
import numpy as np
from typing import TYPE_CHECKING
from scipy.linalg import expm as scipy_expm

if TYPE_CHECKING:
    from .intervalMatrix import intervalMatrix


def exponentialRemainder(intMat: 'intervalMatrix', maxOrder: int) -> 'intervalMatrix':
    """
    Returns the remainder of the exponential matrix
    
    Args:
        intMat: intervalMatrix object
        maxOrder: maximum order of Taylor series
        
    Returns:
        E: Remainder interval matrix
    """
    # Compute absolute value bound
    # MATLAB: M = abs(intMat);
    from .abs import abs as intervalMatrix_abs
    M = intervalMatrix_abs(intMat)
    
    # Compute exponential matrix
    # MATLAB: eM = expm(M);
    eM = scipy_expm(M)
    
    # No value is infinity
    # MATLAB: if ~any(any(isnan(eM)))
    if not np.any(np.isnan(eM)):
        # Compute first Taylor terms
        # MATLAB: Mpow = eye(dim(intMat));
        # MATLAB: eMpartial = eye(dim(intMat));
        from .dim import dim
        n = dim(intMat)[0]
        Mpow = np.eye(n)
        eMpartial = np.eye(n)
        
        # MATLAB: for i=1:maxOrder
        for i in range(1, maxOrder + 1):
            # MATLAB: Mpow = M*Mpow;
            Mpow = M @ Mpow
            # MATLAB: eMpartial = eMpartial + Mpow/factorial(i);
            eMpartial = eMpartial + Mpow / math.factorial(i)
        
        # MATLAB: W = eM-eMpartial;
        W = eM - eMpartial
        
        # Instantiate remainder
        # MATLAB: E = intervalMatrix(zeros(dim(intMat)),W);
        from .intervalMatrix import IntervalMatrix
        E = IntervalMatrix(np.zeros((n, n)), W)
    else:
        # Instantiate remainder
        # MATLAB: E = intervalMatrix(zeros(dim(intMat)),Inf(dim(intMat)));
        from .dim import dim
        n = dim(intMat)[0]
        from .intervalMatrix import IntervalMatrix
        E = IntervalMatrix(np.zeros((n, n)), np.full((n, n), np.inf))
    
    return E
