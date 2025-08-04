"""
priv_encParallelotope - Encloses a non-degenerate ellipsoid by a parallelotope

Authors:       Python translation by AI Assistant
Python translation: 2025
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid


def priv_encParallelotope(E: 'Ellipsoid') -> 'Zonotope':
    """
    Encloses a non-degenerate ellipsoid by a parallelotope

    Args:
        E: ellipsoid object

    Returns:
        Z: zonotope object
    """
    from cora_python.contSet.zonotope.zonotope import Zonotope

    # Transform ellipsoid into sphere -> square around sphere -> back transform
    try:
        sqrt_Q = np.linalg.cholesky(E.Q).T
    except np.linalg.LinAlgError:
        # If Cholesky fails, use eigenvalue decomposition
        eigenvals, eigenvecs = np.linalg.eigh(E.Q)
        sqrt_Q = eigenvecs @ np.diag(np.sqrt(np.maximum(eigenvals, 0)))

    return Zonotope(E.q, sqrt_Q) 