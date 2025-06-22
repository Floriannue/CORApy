"""
This module contains the function for getting the generator matrix of an ellipsoid.
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid


def generators(E: 'Ellipsoid') -> np.ndarray:
    """
    Returns the generator matrix of an ellipsoid in generator representation.
    
    Args:
        E: ellipsoid object
        
    Returns:
        G: generator matrix
    """
    
    Q = E.Q
    if Q is None or Q.size == 0:
        return np.zeros((E.dim(), 0))
    
    U, D, _ = np.linalg.svd(Q)
    
    # Only keep non-zero singular values (for degenerate ellipsoids)
    tol = 1e-10
    nonzero_idx = D > tol
    
    if not np.any(nonzero_idx):
        # All singular values are zero (completely degenerate)
        return np.zeros((E.dim(), 0))
    
    # Keep only columns corresponding to non-zero singular values
    U_reduced = U[:, nonzero_idx]
    D_reduced = D[nonzero_idx]
    
    G = U_reduced @ np.diag(np.sqrt(D_reduced))
    
    return G 