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
    
    # matZonotope * zonotope / zonoBundle
    if isinstance(factor1, matZonotope):
        try:
            from cora_python.contSet.zonotope import Zonotope
            from cora_python.contSet.zonoBundle import ZonoBundle
            if isinstance(factor2, Zonotope):
                return _aux_mtimes_zonotope(factor1, factor2)
            if isinstance(factor2, ZonoBundle):
                return _aux_mtimes_zonoBundle(factor1, factor2)
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
    
    # extract center and generators
    c = Z.center()
    G = Z.generators()

    # normalize generator shape to 3D when needed
    matZ_G = matZ.G
    if isinstance(matZ_G, np.ndarray) and matZ_G.ndim == 2:
        if matZ_G.size == 0:
            matZ_G = np.zeros((*matZ.C.shape, 0))
        else:
            matZ_G = matZ_G[:, :, np.newaxis]

    # output dimension
    if matZ.C.shape == (1, 1) and (matZ_G.size == 0 or matZ_G.shape[:2] == (1, 1)):
        n = c.shape[0]
    else:
        n = matZ.C.shape[0]

    # center
    c_new = matZ.C @ c

    # part 1: matZ.C * G
    if G.size > 0:
        G_part1 = matZ.C @ G
    else:
        G_part1 = np.zeros((n, 0))

    # part 2: reshape(pagemtimes(matZ.G, c), n, [])
    if matZ_G.size > 0:
        part2 = pagemtimes(matZ_G, c)  # shape (n, 1, h)
        G_part2 = part2.reshape(n, -1, order='F')
    else:
        G_part2 = np.zeros((n, 0))

    # part 3: reshape(pagemtimes(matZ.G, reshape(G,...)), n, [])
    if matZ_G.size > 0 and G.size > 0:
        n_mat, m_mat, h = matZ_G.shape
        g = G.shape[1]
        part3_cols = []
        for j in range(g):
            G_col = G[:, j:j+1]
            for i in range(h):
                part3_cols.append(matZ_G[:, :, i] @ G_col)
        G_part3 = np.hstack(part3_cols) if part3_cols else np.zeros((n, 0))
    else:
        G_part3 = np.zeros((n, 0))

    # concatenate generators
    parts = [p for p in [G_part1, G_part2, G_part3] if p.size > 0]
    G_new = np.hstack(parts) if parts else np.zeros((n, 0))

    return Zonotope(c_new, G_new)


def _aux_mtimes_zonoBundle(matZ, zB):
    """Auxiliary function for matZonotope * zonoBundle."""
    from cora_python.contSet.zonoBundle import ZonoBundle
    zB_out = ZonoBundle(zB)
    for i in range(zB_out.parallelSets):
        zB_out.Z[i] = _aux_mtimes_zonotope(matZ, zB_out.Z[i])
    return zB_out
