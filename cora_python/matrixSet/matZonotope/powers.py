"""
powers - computes the powers of a matrix zonotope

Syntax:
    pow = powers(matZ, maxOrder)
    pow = powers(matZ, maxOrder, initialOrder, initialPower)

Inputs:
    matZ - matZonotope object
    maxOrder - maximum Taylor series order until remainder is computed
    initialOrder - starting order (optional)
    initialPower - initial power matrix (optional)

Outputs:
    pow - list of matrix zonotope powers (0-indexed, but MATLAB uses 1-indexed)

Authors:       Matthias Althoff (MATLAB)
              Python translation: 2025
"""

from typing import List, Optional
from .matZonotope import matZonotope


def powers(matZ: matZonotope, maxOrder: int, 
           initialOrder: Optional[int] = None, 
           initialPower: Optional[matZonotope] = None) -> List[matZonotope]:
    """
    Computes the powers of a matrix zonotope
    
    Args:
        matZ: matZonotope object
        maxOrder: maximum Taylor series order
        initialOrder: starting order (default: 1)
        initialPower: initial power matrix (default: matZ)
        
    Returns:
        pow: List of matrix zonotope powers
             Note: MATLAB uses 1-indexed cell arrays, Python uses 0-indexed lists
             So pow[0] corresponds to MATLAB pow{1}, pow[1] to pow{2}, etc.
    """
    # Set default values
    if initialOrder is None:
        initialOrder = 1
    if initialPower is None:
        initialPower = matZ
    
    # Initialize power list
    # MATLAB uses 1-indexed: pow{initialOrder} = initialPower
    # Python uses 0-indexed: pow[initialOrder-1] = initialPower
    pow_list = [None] * maxOrder
    pow_list[initialOrder - 1] = initialPower
    
    # Compute powers
    # MATLAB: for i=(initialOrder+1):maxOrder
    for i in range(initialOrder + 1, maxOrder + 1):
        # MATLAB: pow{i} = pow{i-1}*matZ;
        # In Python: pow[i-1] = pow[i-2] * matZ
        pow_list[i - 1] = pow_list[i - 2] * matZ
    
    # Return list (filter out None entries if any)
    return [p for p in pow_list if p is not None]
