"""
mtimes - overloaded '*' operator for the linear map of an empty set

Syntax:
    O = mtimes(factor1, factor2)

Inputs:
    factor1 - emptySet object, numerical scalar/matrix
    factor2 - emptySet object, numerical scalar/matrix

Outputs:
    O - emptySet object

Example: 
    O = EmptySet(2)
    M = np.array([[2, 1], [-1, 3]])
    result = M * O

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       22-March-2023 (MATLAB)
Last update:   05-April-2023 (MW, bug fix, MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .emptySet import EmptySet

from cora_python.g.functions.matlab.validate.preprocessing.find_class_arg import find_class_arg as findClassArg


def mtimes(factor1: Union['EmptySet', np.ndarray], factor2: Union['EmptySet', np.ndarray]) -> 'EmptySet':
    """
    Overloaded '*' operator for the linear map of an empty set
    
    Args:
        factor1: emptySet object or numerical scalar/matrix
        factor2: emptySet object or numerical scalar/matrix
        
    Returns:
        O: emptySet object
    """
    # Find the emptySet object
    O, M = findClassArg(factor1, factor2, 'EmptySet')
    
    if np.isscalar(M):
        # Keep O as is...
        pass
        
    elif isinstance(M, np.ndarray):
        # Projection to subspace or higher-dimensional space
        O.dimension = M.shape[0]
        
    # Note: intervalMatrix, matZonotope, matPolytope not yet implemented
    # elif isinstance(M, (IntervalMatrix, MatZonotope, MatPolytope)):
    #     O.dimension = M.dim(1)
    
    return O 