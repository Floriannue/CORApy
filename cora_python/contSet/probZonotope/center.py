"""
center - Returns the center of a probabilistic zonotope

Syntax:
    c = center(probZ)

Inputs:
    probZ - ProbZonotope object

Outputs:
    c - center of the probabilistic zonotope as column vector

MATLAB reference: cora_matlab/contSet/@probZonotope/center.m
    c = probZ.Z(:,1);

Authors:       Matthias Althoff (MATLAB)
               Python translation by AI Assistant
Written:       30-September-2006 (MATLAB)
Last update:   22-March-2007 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .probZonotope import ProbZonotope


def center(probZ: "ProbZonotope") -> np.ndarray:
    """
    Returns the center of the probabilistic zonotope.

    Follows MATLAB: c = probZ.Z(:,1);

    In Python we support the numeric-matrix representation used in MATLAB:
        - probZ.Z is an (n × m) ndarray with first column being the center.

    For robustness, if probZ.Z is not an ndarray but has attribute ``c``,
    we fall back to that (e.g. if Z was kept as a Zonotope).
    """
    Z = probZ.Z

    # Preferred representation: numeric matrix [c, G]
    if isinstance(Z, np.ndarray) and Z.size > 0:
        # Ensure 2D
        Z = np.atleast_2d(Z)
        # First column as column vector
        return Z[:, :1]

    # Fallback: Zonotope-like object with attribute c
    if hasattr(Z, "c"):
        c = np.asarray(getattr(Z, "c"))
        return c.reshape(-1, 1)

    # Empty / degenerate case
    return np.zeros((0, 1))

