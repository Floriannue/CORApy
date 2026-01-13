"""
powers - computes the powers of an interval matrix

Syntax:
    pow = powers(intMat, maxOrder)
    pow = powers(intMat, maxOrder, initialOrder, initialPower)

Inputs:
    intMat - intervalMatrix object
    maxOrder - maximum Taylor series order until remainder is computed
    initialOrder - first Taylor series order (optional)
    initialPower - initial power for mixed computations (optional)

Outputs:
    pow - list of powers of the interval matrix

Authors:       Matthias Althoff (MATLAB)
              Python translation: 2025
"""

from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .intervalMatrix import IntervalMatrix


def powers(intMat: 'IntervalMatrix', maxOrder: int,
           initialOrder: Optional[int] = None,
           initialPower: Optional['IntervalMatrix'] = None) -> List['IntervalMatrix']:
    """
    Computes the powers of an interval matrix
    
    Args:
        intMat: intervalMatrix object
        maxOrder: maximum Taylor series order
        initialOrder: first Taylor series order (default: 1)
        initialPower: initial power (default: intMat)
        
    Returns:
        pow: List of interval matrix powers
    """
    if initialOrder is None:
        initialOrder = 1
    if initialPower is None:
        initialPower = intMat
    
    # Initialize power
    # MATLAB: pow{initialOrder}=initialPower;
    pow_list = [None] * maxOrder
    pow_list[initialOrder - 1] = initialPower
    
    # Compute powers
    # MATLAB: for i=(initialOrder+1):maxOrder
    for i in range(initialOrder + 1, maxOrder + 1):
        # MATLAB: pow{i} = pow{i-1}*intMat;
        pow_list[i - 1] = pow_list[i - 2] * intMat
    
    # Return list (filter None)
    return [p for p in pow_list if p is not None]
