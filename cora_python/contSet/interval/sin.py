"""
sin - Overloaded 'sin()' operator for intervals

inf is x infimum, sup is x supremum

[-1, 1]                       if (sup - inf) >= 2*pi,
[-1, 1]                       if (sup - inf) < 2*pi) and inf <= pi/2 and sup < inf),
[sin(inf), sin(sup)]          if (sup - inf) < 2*pi and inf <= pi/2 and sup <= pi/2 and sup >= inf,
[min(sin(inf), sin(sup)), 1]  if (sup - inf) < 2*pi and inf <= pi/2 and sup > pi/2 and sup <= 3/2*pi,
[-1, 1]                       if (sup - inf) < 2*pi and inf <= pi/2 and sup > 3/2*pi),
[-1, 1]                       if (sup - inf) < 2*pi and inf > pi/2 and inf <= 3/2*pi and sup > pi/2 and sup < inf,
[-1, max(sin(inf), sin(sup))] if (sup - inf) < 2*pi and inf > pi/2 and inf <= 3/2*pi and sup <= pi/2,
[sin(sup), sin(inf)]          if (sup - inf) < 2*pi and inf > pi/2 and inf <= 3/2*pi and sup > pi/2 and sup <= 3/2*pi and sup >= inf,
[-1, 1]                       if (sup - inf) < 2*pi and inf > 3/2*pi and inf <= 2*pi and sup > 3/2*pi and sup < inf,
[sin(inf), sin(sup)]          if (sup - inf) < 2*pi and inf > 3/2*pi and inf <= 2*pi and sup <= pi/2,
[min(sin(inf), sin(sup)) , 1] if (sup - inf) < 2*pi and inf > 3/2*pi and inf <= 2*pi and sup > pi/2 and sup <= 3/2*pi,
[sin(inf), sin(sup)]          if (sup - inf) < 2*pi and inf > 3/2*pi and inf <= 2*pi and sup > 3/2*pi and sup >= inf.

Syntax:
    res = sin(I)

Inputs:
    I - interval object

Outputs:
    res - interval object

Example: 
    I = interval([-2;-1],[3;4])
    sin(I)

References:
    [1] M. Althoff, D. Grebenyuk, "Implementation of Interval Arithmetic
        in CORA 2016", ARCH'16.

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: mtimes

Authors: Matthias Althoff, Dmitry Grebenyuk, Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 24-June-2015 (MATLAB)
Last update: 13-January-2016 (DG) (MATLAB)
             05-February-2016 (MA) (MATLAB)
             06-February-2016 (DG) (MATLAB)
             10-February-2016 (MA) (MATLAB)
             22-February-2016 (DG, the matrix case is rewritten) (MATLAB)
             10-January-2024 (MW, fix condition for scalar case) (MATLAB)
Last revision: 18-January-2024 (MW, call fast algorithm in interval/cos) (MATLAB)
Python translation: 2025
"""

import numpy as np
from .interval import Interval


def sin(I: Interval) -> Interval:
    """
    Overloaded sin function for intervals
    
    Uses the identity sin(x) = cos(x - pi/2) for efficient computation
    
    Args:
        I: Interval object
        
    Returns:
        Interval object with sine applied
    """
    # Handle empty intervals
    if I.is_empty():
        return Interval.empty(I.dim())
    
    # Use the fast cosine algorithm with phase shift
    # sin(x) = cos(x - pi/2)
    return I.cos(I - np.pi/2) 