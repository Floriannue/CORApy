"""
expm - operator for the exponential matrix of an interval matrix,
   evaluated dependently

Syntax:
    eI = expm(intMat)
    eI = expm(intMat, maxOrder)
    [eI, iPow, E] = expm(intMat, r, maxOrder)
    [eI, iPow, E] = expm(intMat, r, maxOrder, initialOrder, initialPower)

Inputs:
    intMat - interval matrix
    maxOrder - maximum Taylor series order until remainder is computed
    r - time step size
    initialOrder - starting order (optional)
    initialPower - initial power (optional)

Outputs:
    eI - interval matrix exponential
    iPow - list storing the powers of the matrix
    E - interval matrix for the remainder

Authors:       Matthias Althoff (MATLAB)
              Python translation: 2025
"""

import numpy as np
from typing import Tuple, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .intervalMatrix import IntervalMatrix


def expm(intMat: 'IntervalMatrix', *args) -> 'IntervalMatrix':
    """
    Computes the exponential matrix of an interval matrix (dependent evaluation)
    
    Args:
        intMat: intervalMatrix object
        *args: Variable arguments:
            - No args: maxOrder=10
            - 1 arg: maxOrder
            - 2 args: r, maxOrder
            - 4 args: r, maxOrder, initialOrder, initialPower
            
    Returns:
        eI: Interval matrix exponential (or tuple with iPow, E if 3+ args)
    """
    from .dependentTerms import dependentTerms
    from .powers import powers
    from .exponentialRemainder import exponentialRemainder
    
    nargs = len(args)
    
    if nargs == 0:
        maxOrder = 10
        # Compute exact terms
        # MATLAB: [sq,H] = dependentTerms(intMat,1);
        sq, H = dependentTerms(intMat, 1.0)
        initialOrder = 2
        initialPower = sq
        # Init eI
        eI = H
    elif nargs == 1:
        maxOrder = args[0]
        # Compute exact terms
        sq, H = dependentTerms(intMat, 1.0)
        initialOrder = 2
        initialPower = sq
        # Init eI
        eI = H
    elif nargs == 2:
        r = args[0]
        maxOrder = args[1]
        # Compute exact terms
        # MATLAB: [sq,H] = dependentTerms(intMat*(1/r),r);
        intMat_scaled = intMat * (1.0 / r)
        sq, H = dependentTerms(intMat_scaled, r)
        initialOrder = 2
        initialPower = sq
        # Init eI
        eI = H
    elif nargs == 4:
        r = args[0]
        maxOrder = args[1]
        initialOrder = args[2]
        initialPower = args[3]
        # Init eI
        # MATLAB: eI=initialPower*(1/factorial(initialOrder));
        eI = initialPower * (1.0 / np.math.factorial(initialOrder))
    else:
        raise ValueError(f"expm expects 0, 1, 2, or 4 additional arguments, got {nargs}")
    
    # Compute powers
    # MATLAB: iPow=powers(intMat,maxOrder,initialOrder,initialPower);
    iPow = powers(intMat, maxOrder, initialOrder, initialPower)
    
    # Compute Taylor series
    # MATLAB: for i=(initialOrder+1):maxOrder
    for i in range(initialOrder + 1, maxOrder + 1):
        # MATLAB: eI = eI + iPow{i}*(1/factorial(i));
        # iPow is 0-indexed, so iPow[i-1] corresponds to MATLAB iPow{i}
        if i - 1 < len(iPow):
            eI = eI + iPow[i - 1] * (1.0 / np.math.factorial(i))
    
    # Compute exponential remainder
    # MATLAB: E = exponentialRemainder(intMat,maxOrder);
    E = exponentialRemainder(intMat, maxOrder)
    
    # Final result
    # MATLAB: eI = eI+E;
    eI = eI + E
    
    # Return based on number of arguments
    if nargs >= 2:
        return eI, iPow, E
    else:
        return eI
