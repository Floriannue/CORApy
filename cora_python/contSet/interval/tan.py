"""
tan - Overloaded 'tan()' operator for intervals

inf is infimum, sup is supremum

[-Inf, Inf]                   if (sup - inf) >= pi,
[-Inf, Inf]                   if (sup - inf) < pi) and inf < pi/2 and (sup < inf or sup > pi/2),
[tan(inf), tan(sup)]          if (sup - inf) < pi and inf < pi/2 and (sup >= inf and sup <= pi/2),
[-Inf, Inf]                   if (sup - inf) < pi) and inf >= pi/2 and (sup < inf and sup > pi/2),
[tan(inf), tan(sup)]          if (sup - inf) < pi and inf >= pi/2 and (sup >= inf or sup <= pi/2).

Syntax:
    res = tan(I)

Inputs:
    I - interval object

Outputs:
    I - interval object

Example: 
    I = Interval(0.1, 0.2)
    I.tan()

References:
    [1] M. Althoff, D. Grebenyuk, "Implementation of Interval Arithmetic
        in CORA 2016", ARCH'16.

Authors: Daniel Althoff, Dmitry Grebenyuk, Matthias Althoff, Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 03-November-2015 (MATLAB)
Python translation: 2025
"""

import numpy as np

from .interval import Interval


def tan(I):
    """
    Overloaded tangent operator for intervals
    
    Args:
        I: Interval object
        
    Returns:
        Interval object result of tangent operation
    """

    # Copy to avoid constructor call
    lb = I.inf.copy()
    ub = I.sup.copy()
    
    # Initialize interval with -Inf/Inf
    dims = I.inf.shape
    res_inf = np.full(dims, -np.inf)
    res_sup = np.full(dims, np.inf)
    
    # Only dimensions with a diameter smaller than pi or where
    # tan(inf) <= tan(sup) have non-Inf values
    taninf = np.tan(lb)
    tansup = np.tan(ub)
    
    # Condition: interval width < pi AND tan(inf) <= tan(sup)
    # This handles cases where the interval doesn't cross discontinuities
    ind = (ub - lb < np.pi) & (taninf <= tansup)
    res_inf[ind] = taninf[ind]
    res_sup[ind] = tansup[ind]
    
    return Interval(res_inf, res_sup) 