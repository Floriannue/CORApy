"""
expmIndMixed - operator for the exponential matrix of a matrix zonotope,
   evaluated independently. Higher order terms are computed via interval
   arithmetic.

Syntax:
    [eZ,eI,zPow,iPow,E] = expmIndMixed(matZ, intermediateOrder, maxOrder)

Inputs:
    matZ - matZonotope object
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
from .powers import powers
from cora_python.matrixSet.intervalMatrix import IntervalMatrix
from cora_python.matrixSet.intervalMatrix.expmInd import expmInd

if TYPE_CHECKING:
    from .matZonotope import matZonotope


def expmIndMixed(matZ: 'matZonotope', intermediateOrder: int, 
                 maxOrder: int) -> Tuple['matZonotope', 'IntervalMatrix', 
                                         List['matZonotope'], List['IntervalMatrix'], 
                                         'IntervalMatrix']:
    """
    Computes exponential matrix for matrix zonotope (independent evaluation)
    
    Args:
        matZ: matZonotope object
        intermediateOrder: Order until computation uses matrix zonotopes
        maxOrder: Maximum Taylor series order
        
    Returns:
        eZ: Matrix zonotope exponential part
        eI: Interval matrix exponential part
        zPow: List of matrix zonotope powers
        iPow: List of interval matrix powers
        E: Interval matrix remainder
    """
    # Compute powers
    # MATLAB: zPow=powers(matZ,intermediateOrder);
    zPow = powers(matZ, intermediateOrder)
    
    # Compute finite Taylor series
    # Initialize matrix zonotope
    # MATLAB: eZ=matZ^0;
    n = matZ.C.shape[0]
    eZ = matZonotope(np.eye(n), np.zeros((n, n, 0)))
    
    # Compute finite Taylor sum
    # MATLAB: for i=1:intermediateOrder
    for i in range(1, intermediateOrder + 1):
        # MATLAB: eZ = eZ + zPow{i}*(1/factorial(i));
        # zPow is 0-indexed, so zPow[i-1] corresponds to MATLAB zPow{i}
        if i - 1 < len(zPow):
            eZ = eZ + zPow[i - 1] * (1.0 / math.factorial(i))
    
    # Compute interval part
    # MATLAB: intMat = intervalMatrix(matZ);
    intMat = IntervalMatrix(matZ)
    # MATLAB: [eI,iPow,E] = expmInd(intMat, maxOrder, intermediateOrder+1, ...
    #     intMat*intervalMatrix(zPow{intermediateOrder}));
    zPow_intermediate = zPow[intermediateOrder - 1] if intermediateOrder - 1 < len(zPow) else zPow[-1]
    initialPower = intMat * IntervalMatrix(zPow_intermediate)
    eI, iPow, E = expmInd(intMat, maxOrder, intermediateOrder + 1, initialPower)
    
    return eZ, eI, zPow, iPow, E
