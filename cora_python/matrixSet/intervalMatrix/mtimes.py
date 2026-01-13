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
        return IntervalMatrix(res_int.inf, res_int.sup - res_int.inf)
    
    # factor2 is numeric, factor1 is intervalMatrix
    if isinstance(factor2, np.ndarray):
        # MATLAB: res.int = factor1.int * factor2;
        res_int = interval_mtimes(factor1.int, factor2)
        return IntervalMatrix(res_int.inf, res_int.sup - res_int.inf)
    
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
        return IntervalMatrix(res_int.inf, res_int.sup - res_int.inf)
    
    # intervalMatrix * zonotope
    if isinstance(factor1, IntervalMatrix):
        try:
            from cora_python.contSet.zonotope import Zonotope
            if isinstance(factor2, Zonotope):
                return _aux_mtimes_zonotope(factor1, factor2)
        except ImportError:
            pass
    
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
    if Z_G.size > 0:
        Z_combined = np.hstack([Z_c, Z_G])
    else:
        Z_combined = Z_c
    Zabssum = np.sum(np.abs(Z_combined), axis=1, keepdims=True)
    
    # MATLAB: Z.c = T*Z.c;
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
