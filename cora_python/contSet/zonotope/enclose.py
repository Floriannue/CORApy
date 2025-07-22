"""
enclose - encloses a zonotope and its affine transformation

Description:
    Computes the set
    { a x1 + (1 - a) * x2 | x1 ∈ Z, x2 ∈ Z2, a ∈ [0,1] }
    where Z2 = M*Z + Zplus

Syntax:
    Z = enclose(Z, Z2)
    Z = enclose(Z, M, Zplus)

Inputs:
    Z - zonotope object
    Z2 - zonotope object
    M - matrix for the linear transformation (optional)
    Zplus - zonotope object added to the linear transformation (optional)

Outputs:
    Z - zonotope object

Authors: Matthias Althoff (MATLAB)
         Python translation by AI Assistant
Written: 30-September-2006 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import Union, Optional
from .zonotope import Zonotope

def enclose(Z: Zonotope, Z2_or_M=None, Zplus: Optional[Zonotope] = None) -> Zonotope:
    """
    Encloses a zonotope and its affine transformation
    
    Args:
        Z: Zonotope object
        Z2_or_M: Either a zonotope object Z2 or a transformation matrix M
        Zplus: Zonotope object added to the linear transformation (optional)
        
    Returns:
        Zonotope: Enclosing zonotope object
        
    Example:
        >>> Z1 = Zonotope([1.5, 1.5], [[1, 0], [0, 1]])
        >>> M = np.array([[-1, 0], [0, -1]])
        >>> Z2 = M @ Z1 + [0.5, 0.5]
        >>> Z = enclose(Z1, Z2)
    """
    
    # Parse input arguments
    if Zplus is None:
        # Two argument form: enclose(Z, Z2)
        Z2 = Z2_or_M
    else:
        # Three argument form: enclose(Z, M, Zplus)
        M = Z2_or_M
        Z2 = (M @ Z) + Zplus
    
    # Retrieve number of generators of the zonotopes
    generators1 = Z.G.shape[1]
    generators2 = Z2.G.shape[1]
    
    # If first zonotope has more or equal generators
    if generators2 <= generators1:
        cG = (Z.c - Z2.c) / 2
        Gcut = Z.G[:, :generators2]
        Gadd = Z.G[:, generators2:generators1] if generators2 < generators1 else np.array([]).reshape(Z.G.shape[0], 0)
        Gequal = Z2.G
    else:
        cG = (Z2.c - Z.c) / 2
        Gcut = Z2.G[:, :generators1]
        Gadd = Z2.G[:, generators1:generators2]
        Gequal = Z.G
    
    # Compute enclosing zonotope
    Z.c = (Z.c + Z2.c) / 2
    
    # Construct the new generator matrix 
    G_parts = [(Gcut + Gequal) / 2, cG, (Gcut - Gequal) / 2]
    if Gadd.size > 0:
        G_parts.append(Gadd)
    
    # Concatenate all parts
    if len(G_parts) > 0:
        Z.G = np.hstack(G_parts)
    else:
        Z.G = np.array([]).reshape(Z.c.shape[0], 0)
    
    return Z 