"""
exactPlus - adds two zonotopes by adding all generators of common
   generator factors. Caution: The correspondance of generator factors
   has to be ensured before calling the function; this function is not a
   replacement for the Minkowski sum

Syntax:
    Z = exactPlus(Z,Z2)

Inputs:
    Z - zonotope object
    Z2 - zonotope object
    nrOfGens - (optional) limit on the number of generators that can be
               added exactly

Outputs:
    Z - zonotope object

Example: 
    Z1 = zonotope([0;0],[1 2 -1; 1 -1 3]);
    Z2 = zonotope([0;0],[2 4 -1; 2 -2 3]);
    Zexact = exactPlus(Z1,Z2);
    Z = Z1 + Z2;

    figure; hold on;
    plot(Z1);
    plot(Z2);
    plot(Zexact,[1,2],'g');
    plot(Z,[1,2],'r--');

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: zonotope/plus

Authors:       Matthias Althoff (MATLAB)
               Python translation by AI Assistant
Written:       30-August-2013 (MATLAB)
Last update:   06-September-2013 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import Optional
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from .zonotope import Zonotope


def exactPlus(Z: Zonotope, Z2: Zonotope, nrOfGens: Optional[int] = None) -> Zonotope:
    """
    Adds two zonotopes by adding all generators of common generator factors.
    Caution: The correspondance of generator factors has to be ensured before 
    calling the function; this function is not a replacement for the Minkowski sum.
    
    Args:
        Z: First zonotope object
        Z2: Second zonotope object
        nrOfGens: (optional) limit on the number of generators that can be added exactly
        
    Returns:
        Zonotope: Result of exact addition
        
    Raises:
        CORAerror: If inputs are invalid or computation fails
    """
    # Check input arguments
    if not isinstance(Z, Zonotope):
        raise CORAerror('CORA:wrongValue', 'first', 'zonotope')
    
    if not isinstance(Z2, Zonotope):
        raise CORAerror('CORA:wrongValue', 'second', 'zonotope')
    
    if nrOfGens is not None and not isinstance(nrOfGens, (int, float)):
        raise CORAerror('CORA:wrongValue', 'third', 'numeric')
    
    # Number of generators
    nrOfgens1 = Z.generators.shape[1]
    nrOfgems2 = Z2.generators.shape[1]
    
    if nrOfGens is None:
        nrOfGens = min(nrOfgens1, nrOfgems2)
    else:
        nrOfGens = min(nrOfgens1, nrOfgems2, int(nrOfGens))
    
    # Resulting zonotope
    c_new = Z.center + Z2.center
    
    # Combine generators
    G_common = Z.generators[:, :nrOfGens] + Z2.generators[:, :nrOfGens]
    G_remaining1 = Z.generators[:, nrOfGens:] if nrOfGens < nrOfgens1 else np.zeros((Z.center.shape[0], 0))
    G_remaining2 = Z2.generators[:, nrOfGens:] if nrOfGens < nrOfgems2 else np.zeros((Z2.center.shape[0], 0))
    
    G_new = np.hstack([G_common, G_remaining1, G_remaining2])
    
    return Zonotope(c_new, G_new) 