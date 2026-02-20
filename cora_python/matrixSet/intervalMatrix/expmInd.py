"""
expmInd - operator for the exponential matrix of an interval matrix,
   evaluated independently

Syntax:
    eI = expmInd(intMat, maxOrder)
    [eI, iPow, E] = expmInd(intMat, maxOrder, initialOrder, initialPower)

Inputs:
    intMat - interval matrix
    maxOrder - maximum Taylor series order until remainder is computed
    initialOrder - starting order (optional)
    initialPower - initial power (optional)

Outputs:
    eI - interval matrix exponential
    iPow - list storing the powers of the matrix
    E - interval matrix for the remainder

Authors:       Matthias Althoff (MATLAB)
              Python translation: 2025
"""

import math
import numpy as np
from typing import Tuple, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .intervalMatrix import IntervalMatrix


def expmInd(intMat: 'IntervalMatrix', *args):
    """
    Computes the exponential matrix of an interval matrix (independent evaluation)
    
    Args:
        intMat: intervalMatrix object
        *args: Variable arguments:
            - 1 arg: maxOrder
            - 3 args: maxOrder, initialOrder, initialPower
            
    Returns:
        eI: Interval matrix exponential (or tuple with iPow, E if 3+ args)
    """
    from .powers import powers
    from .exponentialRemainder import exponentialRemainder
    
    nargs = len(args)
    
    if nargs == 1:
        maxOrder = args[0]
        initialOrder = 0
        # MATLAB: initialPower = intMat^0;
        n = intMat.int.shape[0]
        initialPower = IntervalMatrix(np.eye(n), np.zeros((n, n)))
        
        # Compute powers
        # MATLAB: iPow=powers(intMat,maxOrder);
        iPow = powers(intMat, maxOrder)
    elif nargs == 3:
        maxOrder = args[0]
        initialOrder = args[1]
        initialPower = args[2]
        
        # Compute powers
        # MATLAB: iPow=powers(intMat,maxOrder,initialOrder,initialPower);
        iPow = powers(intMat, maxOrder, initialOrder, initialPower)
    else:
        raise ValueError(f"expmInd expects 1 or 3 additional arguments, got {nargs}")
    
    # Compute finite Taylor series
    # Initialize
    # MATLAB: eI=initialPower*(1/factorial(initialOrder));
    eI = initialPower * (1.0 / math.factorial(initialOrder))
    
    # Compute Taylor series
    # MATLAB: for i=(initialOrder+1):maxOrder
    for i in range(initialOrder + 1, maxOrder + 1):
        # MATLAB: eI = eI + iPow{i}*(1/factorial(i));
        if i - 1 < len(iPow):
            eI = eI + iPow[i - 1] * (1.0 / math.factorial(i))
    
    # Compute exponential remainder
    # MATLAB: E = exponentialRemainder(intMat,maxOrder);
    E = exponentialRemainder(intMat, maxOrder)
    
    # Final result
    # MATLAB: eI = eI+E;
    eI = eI + E
    
    # Return based on number of arguments
    if nargs == 3:
        return eI, iPow, E
    else:
        return eI
