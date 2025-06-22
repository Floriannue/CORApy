"""
This module contains the function for computing the rank of an ellipsoid.
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid


def rank(E: 'Ellipsoid') -> int:
    """
    Computes the dimension of the affine hull of an ellipsoid.
    
    Args:
        E: ellipsoid object
        
    Returns:
        r: dimension of the affine hull
    """
    # Empty case
    if E.representsa_('emptySet', 1e-15):
        return 0
    
    # Find minimum svd threshold using reciprocal condition number
    d = np.linalg.svd(E.Q, compute_uv=False)
    
    # Use tolerance from ellipsoid (default to machine epsilon if not available)
    tol = getattr(E, 'TOL', 1e-15)
    mev_th = d[0] * tol
    
    r = np.sum((d > 0) & (d >= mev_th))
    
    return int(r) 