"""
enclose - encloses a zonotope and its affine transformation

Description:
    Computes the set
    { a x1 + (1 - a) * x2 | x1 \in Z, x2 \in Z2, a \in [0,1] }
    where Z2 = M*Z + Zplus

Syntax:
    Z = enclose(Z, Z2)
    Z = enclose(Z, M, Zplus)

Inputs:
    Z - zonotope object
    Z2 - zonotope object
    M - matrix for the linear transformation
    Zplus - zonotope object added to the linear transformation

Outputs:
    Z - zonotope object

Example: 
    Z1 = Zonotope(np.array([[1.5, 1, 0], [1.5, 0, 1]]))
    M = np.array([[-1, 0], [0, -1]])
    Z2 = M @ Z1 + np.array([[0.5], [0.5]])
    Z = enclose(Z1, Z2)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: conZonotope/enclose

Authors: Matthias Althoff (MATLAB)
         Python translation by AI Assistant
Written: 30-September-2006 (MATLAB)
Last update: 22-March-2007 (MATLAB)
         2025 (Tiange Yang, Florian Nüssel, Python translation by AI Assistant)
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

    # MATLAB objects are value types; avoid mutating input zonotopes in Python.
    Z_local = Z.copy()
    Z2_local = Z2.copy() if isinstance(Z2, Zonotope) else Z2
    
    # Retrieve number of generators of the zonotopes
    generators1 = Z_local.G.shape[1]
    generators2 = Z2_local.G.shape[1]
    
    # If first zonotope has more or equal generators
    if generators2 <= generators1:
        cG = (Z_local.c - Z2_local.c) / 2
        Gcut = Z_local.G[:, :generators2]
        Gadd = Z_local.G[:, generators2:generators1] if generators2 < generators1 else np.array([]).reshape(Z_local.G.shape[0], 0)
        Gequal = Z2_local.G
    else:
        cG = (Z2_local.c - Z_local.c) / 2
        Gcut = Z2_local.G[:, :generators1]
        Gadd = Z2_local.G[:, generators1:generators2]
        Gequal = Z_local.G
    
    # Compute enclosing zonotope
    Z_local.c = (Z_local.c + Z2_local.c) / 2
    
    # Construct the new generator matrix 
    G_parts = [(Gcut + Gequal) / 2, cG, (Gcut - Gequal) / 2]
    if Gadd.size > 0:
        G_parts.append(Gadd)
    
    # Concatenate all parts
    if len(G_parts) > 0:
        Z_local.G = np.hstack(G_parts)
    else:
        Z_local.G = np.array([]).reshape(Z_local.c.shape[0], 0)
    
    return Z_local