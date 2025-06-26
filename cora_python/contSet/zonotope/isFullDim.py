"""
isFullDim method for zonotope class
"""

import numpy as np
from typing import Tuple, Optional
from .zonotope import Zonotope


def isFullDim(Z: Zonotope, tol: float = 1e-6) -> Tuple[bool, Optional[np.ndarray]]:
    """
    Checks if the dimension of the affine hull of a zonotope is
    equal to the dimension of its ambient space
    
    Args:
        Z: zonotope object
        tol: tolerance for rank computation
        
    Returns:
        res: True if full-dimensional, False otherwise
        subspace: (optional) Returns a set of orthogonal unit vectors
                 x_1,...,x_k such that Z is strictly contained in
                 center(Z)+span(x_1,...,x_k)
    """
    if not Z.representsa_('emptySet', np.finfo(float).eps):
        Zdim = Z.dim()
        
        # Compute SVD of generator matrix
        U, Sigma, _ = np.linalg.svd(Z.G, full_matrices=True)
        
        # Count significant singular values
        s = Sigma
        r = np.sum(s > tol)
        
        res = Zdim == r
        
        if not res:
            # Return subspace spanned by significant singular vectors
            subspace = U[:, :r]
        else:
            # Full-dimensional case
            subspace = np.eye(Zdim)
    else:
        # Empty set case
        res = False
        subspace = None
    
    return res, subspace 