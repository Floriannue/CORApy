"""
dim - dimension of a polytope

Syntax:
    n = dim(P)

Inputs:
    P - polytope object

Outputs:
    n - dimension of the polytope

Example:
    A = [1 0; 0 1; -1 0; 0 -1];
    b = [1; 1; 1; 1];
    P = polytope(A, b);
    n = dim(P)

Authors: Matthias Althoff (MATLAB)
         Python translation by AI Assistant
Written: 30-September-2006 (MATLAB)
Last update: 22-March-2007 (MATLAB)
Python translation: 2025
"""

import numpy as np
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAError
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .polytope import Polytope

def dim(P: 'Polytope') -> int:
    """
    Returns the dimension of the polytope
    
    Args:
        P: Polytope object
        
    Returns:
        int: Dimension of the polytope (spatial dimensions)
    """
    if P._V is not None and P._V.size > 0:
        return P._V.shape[0]  # spatial dimensions (rows)
    elif P._A is not None and P._A.size > 0:
        return P._A.shape[1]  # number of variables
    elif P._Ae is not None and P._Ae.size > 0:
        return P._Ae.shape[1]  # number of variables
    else:
        return 0 