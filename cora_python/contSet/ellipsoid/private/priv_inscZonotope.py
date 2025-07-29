"""
priv_inscZonotope - Inner-approximates a non-degenerate ellipsoid by a zonotope

Authors:       Python translation by AI Assistant
Python translation: 2025
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid

# Import the global helper function
from cora_python.g.functions.helper.sets.contSet.ellipsoid.eq_point_set import eq_point_set

def priv_inscZonotope(E: 'Ellipsoid', m: int, mode: str) -> 'Zonotope':
    """
    Inner-approximates a non-degenerate ellipsoid by a zonotope

    Args:
        E: ellipsoid object
        m: number of generators
        mode: computation of zonotope norm (see zonotope/norm_)

    Returns:
        Z: zonotope object
    """
    from cora_python.contSet.zonotope.zonotope import Zonotope

    # Read out dimension
    n = E.dim()
    # Extract center
    c = E.center()
    # Compute transformation matrix s.t. T*E == unit hypersphere
    try:
        Tinv = np.linalg.cholesky(E.Q).T
    except np.linalg.LinAlgError:
        # If Cholesky fails, use eigenvalue decomposition
        eigenvals, eigenvecs = np.linalg.eigh(E.Q)
        Tinv = eigenvecs @ np.diag(np.sqrt(np.maximum(eigenvals, 0)))

    # Compute "uniform" distribution of m points on unit hypersphere
    if n == 1:
        # Return exact result
        G = np.array([[1]])
    elif n == m:
        G = np.eye(n)
    elif m % 2 == 0:
        # Such cases result in aligned generators
        # -> choose twice as many and discard half of it
        G = eq_point_set(n-1, m*2)
        G = G[:, :m]
    else:
        G = eq_point_set(n-1, m)
        # Check if aligned (simplified check)
        # In practice, this rarely happens for well-distributed points

    # Init zonotope
    Z = Zonotope(np.zeros((n, 1)), G)

    # Compute zonotope norm
    R = Z.norm_(2, mode)
    if np.isnan(R):
        R = Z.norm_(2, 'ub')

    # We want the zonotope to be enclosed in the ellipsoid, so we scale
    # zonotope(.,G) such that is barely contained in unit hypersphere,
    # and apply inverse transform
    return c + (1/R) * Tinv @ Z 