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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .polytope import Polytope

def dim(P: 'Polytope') -> int:
    """
    Returns the dimension of the ambient space of a polytope
    
    Args:
        P: Polytope object
        
    Returns:
        int: Dimension of the polytope
    """
    if P._dim_val is not None:
        return P._dim_val

    # If _dim_val is not set (should not happen for empty/fullspace due to recent fix),
    # infer from existing properties. This is a fallback.
    n = 0
    if P.A.size > 0:
        n = P.A.shape[1]
    elif P.Ae.size > 0:
        n = P.Ae.shape[1]
    elif P.V.size > 0:
        n = P.V.shape[0]
    
    # If still 0, and the polytope is supposed to have a dimension (e.g., initialized as Polytope()),
    # or it's a copy of a 0-dim set, then the dimension is 0.
    # No additional logic needed here, as the constructor should have handled non-zero dimensions.
    
    # Cache the computed dimension (if it was 0 and now it's inferred, it's correct)
    P._dim_val = n
    return n 