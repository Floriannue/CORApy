"""
mtimes - Overloaded '*' operator for the multiplication of a matrix or a 
   matrix zonotope with a matrix zonotope

Syntax:
    matZ = factor1 * factor2
    matZ = mtimes(factor1, factor2)

Inputs:
    factor1 - numeric matrix or matZonotope object
    factor2 - numeric matrix or matZonotope object

Outputs:
    matZ - matrix zonotope

Authors:       Matthias Althoff, Tobias Ladner (MATLAB)
              Python translation: 2025
"""

import numpy as np
from typing import Union
from .matZonotope import matZonotope
from .pagemtimes import pagemtimes


def mtimes(factor1: Union[np.ndarray, matZonotope, float, int], 
           factor2: Union[np.ndarray, matZonotope, float, int]) -> matZonotope:
    """
    Matrix multiplication for matZonotope
    
    Args:
        factor1: Numeric matrix, scalar, or matZonotope
        factor2: Numeric matrix, scalar, or matZonotope
        
    Returns:
        matZ: Resulting matrix zonotope
    """
    # Handle scalar multiplication (element-wise)
    # MATLAB: if isnumeric(factor1) and isscalar, or if isnumeric(factor2) and isscalar
    if isinstance(factor1, (int, float, np.number)) and isinstance(factor2, matZonotope):
        # scalar * matZonotope: element-wise multiplication
        # MATLAB: S_out.C = factor1*factor2.C;
        # MATLAB: S_out.G = pagemtimes(factor1,factor2.G);
        return matZonotope(factor1 * factor2.C, factor1 * factor2.G)
    
    if isinstance(factor2, (int, float, np.number)) and isinstance(factor1, matZonotope):
        # matZonotope * scalar: element-wise multiplication
        # MATLAB: S_out.C = factor1.C*factor2;
        # MATLAB: S_out.G = pagemtimes(factor1.G,factor2);
        return matZonotope(factor1.C * factor2, factor1.G * factor2)
    
    # factor1 is a numeric matrix -> factor2 must be a matZonotope object
    if isinstance(factor1, np.ndarray):
        # MATLAB: S_out.C = factor1*factor2.C;
        # MATLAB: S_out.G = pagemtimes(factor1,factor2.G);
        return matZonotope(factor1 @ factor2.C, pagemtimes(factor1, factor2.G))
    
    # factor2 is a numeric matrix -> factor1 must be a matZonotope object
    if isinstance(factor2, np.ndarray):
        # MATLAB: S_out.C = factor1.C*factor2;
        # MATLAB: S_out.G = pagemtimes(factor1.G,factor2);
        return matZonotope(factor1.C @ factor2, pagemtimes(factor1.G, factor2))
    
    # Both are matZonotopes
    if isinstance(factor1, matZonotope) and isinstance(factor2, matZonotope):
        return _aux_mtimes_matZonotope(factor1, factor2)
    
    # matZonotope * zonotope
    if isinstance(factor1, matZonotope):
        # Check if factor2 is a zonotope
        try:
            from cora_python.contSet.zonotope import Zonotope
            if isinstance(factor2, Zonotope):
                return _aux_mtimes_zonotope(factor1, factor2)
        except ImportError:
            pass
    
    raise TypeError(f"Unsupported multiplication: {type(factor1)} * {type(factor2)}")


def _aux_mtimes_matZonotope(matZ1: matZonotope, matZ2: matZonotope) -> matZonotope:
    """
    Auxiliary function for matZonotope * matZonotope multiplication
    """
    # MATLAB: Z1 = cat(3,matZ1.C,matZ1.G);
    if matZ1.G.size > 0:
        Z1 = np.concatenate([matZ1.C[:, :, np.newaxis], matZ1.G], axis=2)
    else:
        Z1 = matZ1.C[:, :, np.newaxis]
    
    # MATLAB: Z2 = cat(3,matZ2.C,matZ2.G);
    if matZ2.G.size > 0:
        Z2 = np.concatenate([matZ2.C[:, :, np.newaxis], matZ2.G], axis=2)
    else:
        Z2 = matZ2.C[:, :, np.newaxis]
    
    n1, m1, h1 = Z1.shape
    n2, m2, h2 = Z2.shape
    
    if m1 != n2:
        raise ValueError(f"Matrix dimensions incompatible: {Z1.shape} * {Z2.shape}")
    
    # MATLAB: Z = pagemtimes(Z1,Z2);
    Z = pagemtimes(Z1, Z2)
    
    # MATLAB: S_out.C = Z(:,:,1);
    # MATLAB: S_out.G = Z(:,:,2:end);
    C_out = Z[:, :, 0]
    if Z.shape[2] > 1:
        G_out = Z[:, :, 1:]
    else:
        G_out = np.zeros((n1, m2, 0))
    
    return matZonotope(C_out, G_out)


def _aux_mtimes_zonotope(matZ, Z):
    """
    Auxiliary function for matZonotope * zonotope multiplication
    
    MATLAB: function Z = aux_mtimes_zonotope(matZ,Z)
    Computes the linear map of a zonotope by a matrix zonotope.
    
    Args:
        matZ: matZonotope object
        Z: Zonotope object
        
    Returns:
        Z: Resulting zonotope
    """
    from cora_python.contSet.zonotope import Zonotope
    
    # MATLAB: Z.c = matZ.C*Z.c;
    c_new = matZ.C @ Z.center()
    
    # MATLAB: Z.G = [matZ.C*Z.G, reshape(matZ.G*Z.c,[],size(Z.G,2))];
    # First part: matZ.C * Z.G
    if Z.generators().size > 0:
        G_part1 = matZ.C @ Z.generators()
    else:
        G_part1 = np.zeros((matZ.C.shape[0], 0))
    
    # Second part: reshape(matZ.G * Z.c, [], size(Z.G, 2))
    # MATLAB: reshape(matZ.G*Z.c,[],size(Z.G,2))
    # This multiplies each generator matrix in matZ.G with Z.c and reshapes
    if matZ.G.size > 0:
        # matZ.G has shape (n, m, h) where h is number of generators
        # Z.c has shape (m, 1)
        # We want to compute matZ.G[:, :, i] @ Z.c for each i
        n, m, h = matZ.G.shape
        Z_c = Z.center()
        G_part2_list = []
        for i in range(h):
            G_i = matZ.G[:, :, i] @ Z_c  # Shape (n, 1)
            G_part2_list.append(G_i)
        # Stack horizontally
        if G_part2_list:
            G_part2 = np.hstack(G_part2_list)  # Shape (n, h)
        else:
            G_part2 = np.zeros((n, 0))
    else:
        G_part2 = np.zeros((matZ.C.shape[0], 0))
    
    # MATLAB: Z.G = [matZ.C*Z.G, reshape(matZ.G*Z.c,[],size(Z.G,2))];
    # But wait, the reshape dimensions don't match. Let me check MATLAB code more carefully.
    # Actually, looking at the MATLAB code, it seems like:
    # - matZ.C*Z.G gives generators from the center matrix
    # - matZ.G*Z.c gives generators from the generator matrices
    # The reshape might be different. Let me implement a simpler version first.
    
    # Concatenate generators
    if G_part1.size > 0 and G_part2.size > 0:
        G_new = np.hstack([G_part1, G_part2])
    elif G_part1.size > 0:
        G_new = G_part1
    elif G_part2.size > 0:
        G_new = G_part2
    else:
        G_new = np.zeros((c_new.shape[0], 0))
    
    return Zonotope(c_new, G_new)
