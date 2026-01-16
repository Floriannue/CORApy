"""
mtimes - Overloaded '*' operator for interval matrix multiplication

Syntax:
    res = factor1 * factor2
    res = mtimes(factor1, factor2)

Inputs:
    factor1 - numeric matrix or intervalMatrix
    factor2 - numeric matrix or intervalMatrix

Outputs:
    res - intervalMatrix

Authors:       Matthias Althoff (MATLAB)
              Python translation: 2025
"""

import numpy as np
from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .intervalMatrix import IntervalMatrix


def mtimes(factor1: Union[np.ndarray, 'IntervalMatrix'], 
           factor2: Union[np.ndarray, 'IntervalMatrix']) -> 'IntervalMatrix':
    """
    Matrix multiplication for interval matrices
    
    Args:
        factor1: Numeric matrix or intervalMatrix
        factor2: Numeric matrix or intervalMatrix
        
    Returns:
        res: Resulting intervalMatrix
    """
    from .intervalMatrix import IntervalMatrix
    from cora_python.contSet.interval.mtimes import mtimes as interval_mtimes
    
    # factor1 is numeric, factor2 is intervalMatrix
    if isinstance(factor1, np.ndarray):
        # MATLAB: res.int = factor1 * factor2.int;
        res_int = interval_mtimes(factor1, factor2.int)
        center = 0.5 * (res_int.inf + res_int.sup)
        delta = 0.5 * (res_int.sup - res_int.inf)
        return IntervalMatrix(center, delta)
    
    # factor2 is numeric, factor1 is intervalMatrix
    if isinstance(factor2, np.ndarray):
        # MATLAB: res.int = factor1.int * factor2;
        res_int = interval_mtimes(factor1.int, factor2)
        center = 0.5 * (res_int.inf + res_int.sup)
        delta = 0.5 * (res_int.sup - res_int.inf)
        return IntervalMatrix(center, delta)
    
    # Scalar multiplication: intervalMatrix * scalar or scalar * intervalMatrix
    if isinstance(factor1, (int, float, np.number)) and isinstance(factor2, IntervalMatrix):
        # scalar * intervalMatrix: multiply interval by scalar
        res_int = type(factor2.int)(factor2.int.inf * factor1, factor2.int.sup * factor1)
        return IntervalMatrix((res_int.inf + res_int.sup) / 2, (res_int.sup - res_int.inf) / 2)
    
    if isinstance(factor2, (int, float, np.number)) and isinstance(factor1, IntervalMatrix):
        # intervalMatrix * scalar: multiply interval by scalar
        res_int = type(factor1.int)(factor1.int.inf * factor2, factor1.int.sup * factor2)
        return IntervalMatrix((res_int.inf + res_int.sup) / 2, (res_int.sup - res_int.inf) / 2)
    
    # Both are intervalMatrices
    if isinstance(factor1, IntervalMatrix) and isinstance(factor2, IntervalMatrix):
        # MATLAB: res.int = factor1.int * factor2.int;
        res_int = interval_mtimes(factor1.int, factor2.int)
        center = 0.5 * (res_int.inf + res_int.sup)
        delta = 0.5 * (res_int.sup - res_int.inf)
        return IntervalMatrix(center, delta)
    
    # intervalMatrix * zonotope or polyZonotope
    if isinstance(factor1, IntervalMatrix):
        from cora_python.contSet.zonotope import Zonotope
        from cora_python.contSet.polyZonotope import PolyZonotope
        
        if isinstance(factor2, Zonotope):
            return _aux_mtimes_zonotope(factor1, factor2)
        
        if isinstance(factor2, PolyZonotope):
            return _aux_mtimes_polyZonotope(factor1, factor2)
    
    raise TypeError(f"Unsupported multiplication: {type(factor1)} * {type(factor2)}")


def _aux_mtimes_zonotope(intMat, Z):
    """
    Auxiliary function for intervalMatrix * zonotope multiplication
    See Theorem 3.3 in [1]
    
    MATLAB: function Z = aux_mtimes_zonotope(intMat,Z)
    
    Args:
        intMat: IntervalMatrix object
        Z: Zonotope object
        
    Returns:
        Z: Resulting zonotope
    """
    from cora_python.contSet.zonotope import Zonotope
    from cora_python.contSet.interval import Interval
    
    # MATLAB: M_min = infimum(intMat.int);
    # MATLAB: M_max = supremum(intMat.int);
    M_min = intMat.int.inf
    M_max = intMat.int.sup
    
    # MATLAB: T = 0.5*(M_max+M_min);
    # MATLAB: S = 0.5*(M_max-M_min);
    T = 0.5 * (M_max + M_min)
    S = 0.5 * (M_max - M_min)
    
    # MATLAB: Zabssum = sum(abs([Z.c,Z.G]),2);
    Z_c = Z.center()
    Z_G = Z.generators()
    # Ensure Z_c is a column vector (n x 1) where n matches T's column dimension
    if Z_c.ndim == 1:
        Z_c = Z_c.reshape(-1, 1)
    elif Z_c.ndim == 2:
        if Z_c.shape[1] != 1:
            # If it's a row vector or matrix, flatten and reshape to column vector
            Z_c = Z_c.flatten().reshape(-1, 1)
        # Z_c is already a column vector, but check if dimension matches T
        if Z_c.shape[0] != T.shape[1]:
            # Dimension mismatch - this shouldn't happen in correct usage
            raise ValueError(f"Dimension mismatch: IntervalMatrix is {T.shape[0]}x{T.shape[1]}, but zonotope center is {Z_c.shape[0]}D")
    
    if Z_G.size > 0:
        # Ensure Z_G is 2D (n x p) where n matches Z_c
        if Z_G.ndim == 1:
            Z_G = Z_G.reshape(-1, 1)
        elif Z_G.ndim == 2 and Z_G.shape[0] != Z_c.shape[0]:
            # Reshape if needed to match Z_c dimension
            Z_G = Z_G.reshape(Z_c.shape[0], -1)
        Z_combined = np.hstack([Z_c, Z_G])
    else:
        Z_combined = Z_c
    Zabssum = np.sum(np.abs(Z_combined), axis=1, keepdims=True)
    
    # MATLAB: Z.c = T*Z.c;
    # T is (n x m), Z_c is (m x 1), result is (n x 1)
    c_new = T @ Z_c
    
    # MATLAB: Z.G = [T*Z.G,diag(S*Zabssum)];
    G_list = []
    if Z_G.size > 0:
        G_part1 = T @ Z_G
        G_list.append(G_part1)
    
    # diag(S*Zabssum) creates diagonal matrix with S*Zabssum on diagonal
    G_part2 = np.diag((S @ Zabssum).flatten())
    G_list.append(G_part2)
    
    if G_list:
        G_new = np.hstack(G_list)
    else:
        G_new = np.zeros((c_new.shape[0], 0))
    
    return Zonotope(c_new, G_new)


def _aux_mtimes_polyZonotope(intMat, pZ):
    """
    Auxiliary function for intervalMatrix * polyZonotope multiplication
    
    MATLAB: function pZ = aux_mtimes_polyZonotope(intMat,pZ)
    
    Args:
        intMat: IntervalMatrix object
        pZ: PolyZonotope object
        
    Returns:
        pZ: Resulting polyZonotope
    """
    from cora_python.contSet.polyZonotope import PolyZonotope
    
    # MATLAB: M_min = infimum(intMat.int);
    # MATLAB: M_max = supremum(intMat.int);
    M_min = intMat.int.inf
    M_max = intMat.int.sup
    
    # MATLAB: M = 0.5*(M_max+M_min);
    # MATLAB: R = 0.5*(M_max-M_min);
    M = 0.5 * (M_max + M_min)
    R = 0.5 * (M_max - M_min)
    
    # MATLAB: I = interval(pZ);
    I = pZ.interval()
    # MATLAB: S = abs(center(I)) + rad(I);
    S = np.abs(I.center()) + I.rad()
    # Ensure S is a column vector
    if S.ndim == 1:
        S = S.reshape(-1, 1)
    
    # MATLAB: pZ.c = M*pZ.c;
    c_new = M @ pZ.c
    
    # MATLAB: if ~isempty(pZ.G)
    # MATLAB:     pZ.G = M*pZ.G;
    # MATLAB: end
    if pZ.G.size > 0:
        G_new = M @ pZ.G
    else:
        G_new = np.zeros((c_new.shape[0], 0))
    
    # MATLAB: if ~isempty(pZ.GI)
    # MATLAB:     pZ.GI = [M*pZ.GI, diag(R*S)];
    # MATLAB: else
    # MATLAB:     pZ.GI = diag(R*S);
    # MATLAB: end
    R_S = R @ S
    # Ensure R_S is a column vector for diag
    if R_S.ndim == 1:
        R_S = R_S.flatten()
    elif R_S.ndim == 2:
        R_S = R_S.flatten()
    
    diag_R_S = np.diag(R_S)
    
    if pZ.GI.size > 0:
        GI_new = np.hstack([M @ pZ.GI, diag_R_S])
    else:
        GI_new = diag_R_S
    
    # MATLAB: pZ.id stays the same
    # pZ.E stays the same
    
    return PolyZonotope(c_new, G_new, GI_new, pZ.E, pZ.id)
