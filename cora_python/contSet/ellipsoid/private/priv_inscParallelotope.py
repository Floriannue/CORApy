"""
priv_inscParallelotope - Inner-approximates a non-degenerate ellipsoid by a parallelotope

Authors:       Python translation by AI Assistant
Python translation: 2025
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid

def priv_inscParallelotope(E: 'Ellipsoid') -> 'Zonotope':
    """
    Inner-approximates a non-degenerate ellipsoid by a parallelotope

    Args:
        E: ellipsoid object

    Returns:
        Z: zonotope object
    """
    from cora_python.contSet.zonotope.zonotope import Zonotope

    n = E.dim()
    try:
        sqrt_Q = np.linalg.cholesky(E.Q).T
        T = np.linalg.inv(sqrt_Q)
    except np.linalg.LinAlgError:
        # If Cholesky fails, use eigenvalue decomposition
        eigenvals, eigenvecs = np.linalg.eigh(E.Q)
        sqrt_Q = eigenvecs @ np.diag(np.sqrt(np.maximum(eigenvals, 0)))
        T = np.linalg.pinv(sqrt_Q)

    # Transform ellipsoid into sphere -> square into sphere -> back transform
    return Zonotope(E.q, np.linalg.inv(T) * (1/np.sqrt(n)) * np.eye(n)) 