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
    if P._isHRep:
        # either constraints A*x <= b  or  Ae*x == be  given
        if P._A is not None and P._A.size > 0:
            n = P._A.shape[1]
        elif P._Ae is not None and P._Ae.size > 0:
            n = P._Ae.shape[1]
        else:
            # constraints, such as zeros(0,n) given
            A_cols = P._A.shape[1] if P._A is not None else 0
            Ae_cols = P._Ae.shape[1] if P._Ae is not None else 0
            n = max(A_cols, Ae_cols)
    elif P._isVRep:
        n = P._V.shape[0] if P._V is not None and P._V.size > 0 else 0
    else:
        # Fallback based on which attributes are populated
        dims = []
        if hasattr(P, '_A') and P._A is not None and P._A.size > 0:
            dims.append(P._A.shape[1])
        if hasattr(P, '_Ae') and P._Ae is not None and P._Ae.size > 0:
            dims.append(P._Ae.shape[1])
        if hasattr(P, '_V') and P._V is not None and P._V.size > 0:
            dims.append(P._V.shape[0])
        n = max(dims) if dims else 0
    
    return n 