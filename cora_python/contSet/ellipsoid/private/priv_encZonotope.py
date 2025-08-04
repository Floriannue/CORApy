"""
priv_encZonotope - Encloses a non-degenerate ellipsoid by a zonotope

Authors:       Python translation by AI Assistant
Python translation: 2025
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid

# Import the global helper function
from cora_python.g.functions.helper.sets.contSet.ellipsoid.eq_point_set import eq_point_set

def priv_encZonotope(E: 'Ellipsoid', nrGen: int) -> 'Zonotope':
    """
    Encloses a non-degenerate ellipsoid by a zonotope

    Args:
        E: ellipsoid object
        nrGen: number of generators of resulting zonotope

    Returns:
        Z: zonotope object
    """
    from cora_python.contSet.zonotope.zonotope import Zonotope

    # Read out dimension
    n = E.dim()
    # Extract center
    c = E.center()
    # Compute transformation matrix s.t. T*E == unit hyper-sphere
    try:
        Tinv = np.linalg.cholesky(E.Q).T
    except np.linalg.LinAlgError:
        # If Cholesky fails, use eigenvalue decomposition
        eigenvals, eigenvecs = np.linalg.eigh(E.Q)
        Tinv = eigenvecs @ np.diag(np.sqrt(np.maximum(eigenvals, 0)))

    # Compute "uniform" distribution of m points on unit hyper-sphere
    if n == 1:
        G = np.array([[-1, 1]])
    elif n == nrGen:
        # This should call priv_encParallelotope, which needs to be imported
        from .priv_encParallelotope import priv_encParallelotope
        return priv_encParallelotope(E)
    else:
        G = eq_point_set(n-1, nrGen)

    # Create zonotope and compute minimum norm
    Z_temp = Zonotope(np.zeros((n, 1)), G)
    L = Z_temp.minnorm()[0]

    # We want the ellipsoid to be contained in the zonotope, so we scale
    # zonotope(.,G) s.t. it touches E (for exact norm computation), then
    # apply retransform
    return Zonotope(c, (1/L) * Tinv @ G) 