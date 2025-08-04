"""
This module contains the function for computing the rank of an ellipsoid.
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ellipsoid import Ellipsoid

def rank(E: 'Ellipsoid') -> int:
    """
    rank - computes the dimension of the affine hull of an ellipsoid

    Syntax:
        r = rank(E)

    Inputs:
        E - ellipsoid object

    Outputs:
        r - dimension of the affine hull

    Example: 
        E = Ellipsoid(np.array([[1,0],[0,1]]));
        r = rank(E) 

    Authors:       Victor Gassmann
    Written:       16-March-2021
    Last update:   04-July-2022 (VG, allow class array input)
    Last revision: ---
                   Automatic python translation: Florian Nüssel BA 2025
    """
    # Empty case
    if E.representsa_('emptySet', E.TOL):
        return 0

    d = np.linalg.svd(E.Q, compute_uv=False)

    if d.size == 0: # Handle case of empty Q matrix directly after SVD
        return 0

    mev_th = d[0] * E.TOL if d.size > 0 else 0 # this line is technically redundant after above check, but good for clarity
    r = np.sum((d > 0) & (d >= mev_th))
    
    return int(r) 