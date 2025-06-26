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
    """
    # Empty case
    if E.representsa_('emptySet', E.TOL):
        return 0

    d = np.linalg.svd(E.Q, compute_uv=False)
    mev_th = d[0] * E.TOL if d.size > 0 else 0
    r = np.sum((d > 0) & (d >= mev_th))
    
    return int(r) 