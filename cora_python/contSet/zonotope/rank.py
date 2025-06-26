"""
rank method for zonotope class
"""

import numpy as np
from typing import Optional
from .zonotope import Zonotope


def rank(Z: Zonotope, tol: Optional[float] = None) -> int:
    """
    Computes the dimension of the affine hull of a zonotope
    
    Args:
        Z: zonotope object
        tol: tolerance for rank computation (default: None, uses numpy default)
        
    Returns:
        Rank of the zonotope (dimension of affine hull)
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