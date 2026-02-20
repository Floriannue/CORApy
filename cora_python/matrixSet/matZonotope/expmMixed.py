"""
expmMixed - operator for the exponential matrix of a matrix zonotope,
   evaluated dependently. Higher order terms are computed via interval
   arithmetic.

Syntax:
    [eZ,eI,zPow,iPow,E] = expmMixed(matZ, r, intermediateOrder, maxOrder)

Inputs:
    matZ - matZonotope object
    r - time step size
    intermediateOrder - Taylor series order until computation is performed with matrix zonotopes
    maxOrder - maximum Taylor series order until remainder is computed

Outputs:
    eZ - matrix zonotope exponential part
    eI - interval matrix exponential part
    zPow - list of matrix zonotope powers
    iPow - list of interval matrix powers
    E - interval matrix for the remainder

Authors:       Matthias Althoff (MATLAB)
              Python translation: 2025
"""

import math
import numpy as np
from typing import Tuple, List, TYPE_CHECKING
from .matZonotope import matZonotope
from .dependentTerms import dependentTerms
from .powers import powers
from cora_python.matrixSet.intervalMatrix import IntervalMatrix
from cora_python.matrixSet.intervalMatrix.expm import expm

if TYPE_CHECKING:
    from .matZonotope import matZonotope


def expmMixed(matZ: 'matZonotope', r: float, intermediateOrder: int, 
              maxOrder: int) -> Tuple['matZonotope', 'IntervalMatrix', 
                                      List['matZonotope'], List['IntervalMatrix'], 
                                      'IntervalMatrix']:
    """
    Computes exponential matrix for matrix zonotope (mixed evaluation)
    
    Args:
        matZ: matZonotope object
        r: time step size
        intermediateOrder: Order until computation uses matrix zonotopes
        maxOrder: Maximum Taylor series order
        
    Returns:
        eZ: Matrix zonotope exponential part
        eI: Interval matrix exponential part
        zPow: List of matrix zonotope powers
        iPow: List of interval matrix powers
        E: Interval matrix remainder
    """
    # Compute exact terms
    # MATLAB: [sq,H] = dependentTerms(matZ*(1/r),r);
    matZ_scaled = matZ * (1.0 / r)
    sq, H = dependentTerms(matZ_scaled, r)
    
    # Init eZ
    # MATLAB: eZ = H;
    eZ = H
    
    # Compute powers
    # MATLAB: zPow=powers(matZ,intermediateOrder,2,sq);
    zPow = powers(matZ, intermediateOrder, 2, sq)
    
    # Add first power for input computations
    # MATLAB: zPow{1}=matZ;
    zPow[0] = matZ  # Python 0-indexed
    
    # Compute finite Taylor sum
    # MATLAB: for i=3:intermediateOrder
    for i in range(3, intermediateOrder + 1):
        # MATLAB: eZ = eZ + zPow{i}*(1/factorial(i));
        # zPow is 0-indexed, so zPow[i-1] corresponds to MATLAB zPow{i}
        if i - 1 < len(zPow):
            eZ = eZ + zPow[i - 1] * (1.0 / math.factorial(i))
    
    # Compute interval part
    # MATLAB: intMat = intervalMatrix(matZ);
    intMat = IntervalMatrix(matZ)
    # MATLAB: [eI,iPow,E] = expm(intMat, r, maxOrder, intermediateOrder+1, ...
    #     intMat*intervalMatrix(zPow{intermediateOrder}));
    zPow_intermediate = zPow[intermediateOrder - 1] if intermediateOrder - 1 < len(zPow) else zPow[-1]
    initialPower = intMat * IntervalMatrix(zPow_intermediate)
    eI, iPow, E = expm(intMat, r, maxOrder, intermediateOrder + 1, initialPower)
    
    return eZ, eI, zPow, iPow, E
