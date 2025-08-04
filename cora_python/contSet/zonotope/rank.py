"""
rank - computes the dimension of the affine hull of a zonotope

Syntax:
    r = rank(Z)

Inputs:
    Z - zonotope object
    tol - numeric, tolerance

Outputs:
    r - dimension of the affine hull

Example:
    from cora_python.contSet.zonotope import Zonotope, rank
    import numpy as np
    Z = Zonotope(np.array([[1], [0]]), np.array([[1, 1, 0], [0, 0, 1]]))
    r = rank(Z)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Matthias Althoff (MATLAB)
               Python translation by AI Assistant
Written:       06-May-2009 (MATLAB)
Last update:   15-January-2024 (TL, added tol) (MATLAB)
               2025 (Tiange Yang, Florian Nüssel, Python translation by AI Assistant)
"""
import numpy as np
from typing import Optional
from .zonotope import Zonotope

def rank(Z: Zonotope, tol: Optional[float] = None) -> int:
    """
    Computes the dimension of the affine hull of a zonotope.
    """
    # Handle empty generators case
    if Z.G.size == 0:
        return 0
    
    # Compute rank using numpy's matrix_rank
    if tol is None:
        r = np.linalg.matrix_rank(Z.G)
    else:
        r = np.linalg.matrix_rank(Z.G, tol=tol)
    
    return r 