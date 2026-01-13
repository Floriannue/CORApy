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
    
    # Both are intervalMatrices
    if isinstance(factor1, IntervalMatrix) and isinstance(factor2, IntervalMatrix):
        # MATLAB: res.int = factor1.int * factor2.int;
        res_int = interval_mtimes(factor1.int, factor2.int)
        return IntervalMatrix(res_int.inf, res_int.sup - res_int.inf)
    
    raise TypeError(f"Unsupported multiplication: {type(factor1)} * {type(factor2)}")
