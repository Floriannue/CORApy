"""
mtimes - overloaded '*' operator for the linear map of a full-dimensional
   space
   case R^0: can only be multiplied by 0 (not representable in MATLAB)

Syntax:
   fs = mtimes(factor1,factor2)

Inputs:
   factor1 - fullspace object, numerical scalar/matrix
   factor2 - fullspace object, numerical scalar/matrix

Outputs:
   res - linearly mapped full-dimensional space

Example: 
   fs = fullspace(2);
   M = [2 1; -1 3];
   M*fs

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger
Written:       22-March-2023
Last update:   ---
Last revision: ---
Automatic python translation: Florian Nüssel BA 2025

----------------------------- BEGIN CODE ------------------------------
"""

import numpy as np
from cora_python.g.functions.matlab.validate.check.equal_dim_check import equal_dim_check
from cora_python.g.functions.matlab.validate.preprocessing.find_class_arg import find_class_arg
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def mtimes(factor1, factor2):
    """
    Overloaded '*' operator for the linear map of a full-dimensional space
    case R^0: can only be multiplied by 0 (not representable in MATLAB)
    
    Args:
        factor1: fullspace object, numerical scalar/matrix
        factor2: fullspace object, numerical scalar/matrix
        
    Returns:
        res: linearly mapped full-dimensional space
    """
    # find the fullspace object
    fs, M = find_class_arg(factor1, factor2, 'Fullspace')
    
    # Check dimensions for matrix multiplication specifically
    if isinstance(M, np.ndarray) and M.ndim > 1:
        # matrix @ fullspace: matrix.shape[1] should equal fullspace.dim()
        if M.shape[1] != fs.dimension:
            raise CORAerror('CORA:dimensionMismatch', factor1, factor2)
    else:
        # For non-matrix operations, use general dimension check
        equal_dim_check(factor1, factor2)
    
    if fs.dimension == 0:
        raise CORAerror('CORA:notSupported', 'Linear map of R^0 not supported')
    
    if np.isscalar(M):
        # multiplication with scalar
        if withinTol(M, 0):
            # Import here to avoid circular imports
            from cora_python.contSet.interval import Interval
            res = Interval(np.zeros(fs.dimension), np.zeros(fs.dimension))
        else:
            # all other scalar values: keep fs as is
            res = fs
    
    elif isinstance(M, np.ndarray) or isinstance(M, list):
        M = np.array(M)
        zerorows = ~np.any(M, axis=1)
        if np.any(zerorows):
            # Import here to avoid circular imports
            from cora_python.contSet.interval import Interval
            # Create bounds with zeros for zero rows, inf for others
            lb = np.where(zerorows, 0, -np.inf)
            ub = np.where(zerorows, 0, np.inf)
            res = Interval(lb, ub)
        elif M.shape[0] != fs.dimension:
            # all other matrices: still fullspace
            # Import here to avoid circular imports
            from cora_python.contSet.fullspace import Fullspace
            res = Fullspace(M.shape[0])
        else:
            # square matrix
            res = fs
    
    elif hasattr(M, '__class__') and M.__class__.__name__ in ['IntervalMatrix', 'MatZonotope', 'MatPolytope']:
        # throw error for now...
        raise CORAerror('CORA:noops', M, fs)
    
    return res

# ------------------------------ END OF CODE ------------------------------ 