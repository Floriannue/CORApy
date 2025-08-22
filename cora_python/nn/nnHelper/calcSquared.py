"""
calcSquared - computes pZ_1 * pZ_2 with pZ_1, pZ_2 being multiplicatives 
   of a one-dimensional polyZonotope pZ:
   pZ_1 * pZ_2 =
        = (c1 + G1 + GI1)*(c2 + G2 + GI2)
        = c1*c2 + c1*G2 + c1*GI2
          + G1*c2 + G1*G2 + G2*GI2
          + GI1*c2 + GI1*G2 + GI1*GI2
        = (c1*c2 + sum(0.5[GI1*GI2](:, I)))
          % = c
          + (c1*G2 + G1*c2 + G1*G2)
          % = G
          + (c1*GI2 + GI2*c2 + 0.5[GI1*GI2](:, I) + [GI1*GI2](:, ~I) + G1*GI2 + GI2*G2)
          % = GI
          % with I ... indices of generators that need shifting

Note: (Half of) [GI1*GI2](:, I) appears in c & GI:
  Squaring independent Generators need shifting
  as only positive part is used afterwards.
  -> this will be corrected when adding the terms together

This is a Python translation of the MATLAB CORA implementation.

Authors: MATLAB: Tobias Ladner
         Python: AI Assistant
"""

import numpy as np
from typing import Tuple
from .calcSquaredG import calcSquaredG


def calcSquared(c1: np.ndarray, G1: np.ndarray, GI1: np.ndarray, E1: np.ndarray,
                c2: np.ndarray, G2: np.ndarray, GI2: np.ndarray, E2: np.ndarray,
                isEqual: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute pZ_1 * pZ_2 with pZ_1, pZ_2 being multiplicatives of a one-dimensional polyZonotope pZ.
    
    Args:
        c1: center of first polyZonotope
        G1: dependent generators of first polyZonotope
        GI1: independent generators of first polyZonotope
        E1: exponential matrix of first polyZonotope
        c2: center of second polyZonotope
        G2: dependent generators of second polyZonotope
        GI2: independent generators of second polyZonotope
        E2: exponential matrix of second polyZonotope
        isEqual: whether the two polyZonotopes are equal
        
    Returns:
        c: center of pZ^(i1+i2)
        G: dependent generators of pZ^(i1+i2)
        GI: independent generators of pZ^(i1+i2)
        
    See also: polyZonotope/quadMap, nnHelper/calcSquaredG, nnHelper/calcSquaredE
    """
    G_quad = calcSquaredG(G1, G2, isEqual)
    GI_quad = calcSquaredG(GI1, GI2, isEqual)
    
    # construct squared parameters
    c = c1 * c2
    
    # See Note
    if isEqual:
        r = GI1.shape[1]
        GI_quad[:, :r] = 0.5 * GI_quad[:, :r]
        c = c + np.sum(GI_quad[:, :r], axis=1, keepdims=True)
    
    # the same principle applies to G1, G2:
    # if all exponents are even and they become independent generators
    # after multiplying them with GI2, GI1, respectively.
    # except center does not need shifting as
    # G1, G2 only scale the independent generators of GI2, GI1.
    
    # G1 * GI2
    even_indices = np.all(E1 % 2 == 0, axis=0)
    G1_ind = G1.copy()  # copy by value
    G1_ind[:, even_indices] = 0.5 * G1_ind[:, even_indices]
    G1GI2 = calcSquaredG(G1_ind, GI2)
    
    # GI1 * G2
    even_indices = np.all(E2 % 2 == 0, axis=0)
    G2_ind = G2.copy()  # copy by value
    G2_ind[:, even_indices] = 0.5 * G2_ind[:, even_indices]
    GI1G2 = calcSquaredG(GI1, G2_ind)
    
    G = np.hstack([G1 * c2, c1 * G2, G_quad])
    
    if isEqual:
        GI = np.hstack([GI1 * c2, c1 * GI1, GI_quad, 2 * GI1G2])
    else:
        GI = np.hstack([GI1 * c2, c1 * GI2, GI_quad, G1GI2, GI1G2])
    
    return c, G, GI
