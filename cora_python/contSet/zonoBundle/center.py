"""
center - returns an estimate for the center of a zonotope bundle

Syntax:
    c = center(zB)

Inputs:
    zB - zonoBundle object

Outputs:
    c - center

Example:
    Z1 = zonotope(zeros(2,1),[1 0.5; -0.2 1]);
    Z2 = zonotope(ones(2,1),[1 -0.5; 0.2 1]);
    zB = zonoBundle({Z1,Z2});
    c = center(zB)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Matthias Althoff, Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       03-February-2011 (MATLAB)
Last update:   24-April-2023 (MW, return empty if empty, otherwise cheby, MATLAB)
Last revision: ---
Python translation: 2025
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cora_python.contSet.zonoBundle.zonoBundle import ZonoBundle


def center(zB: 'ZonoBundle') -> np.ndarray:
    """
    Returns an estimate for the center of a zonotope bundle
    
    Args:
        zB: zonoBundle object
        
    Returns:
        numpy.ndarray: center of the zonotope bundle
    """
    if zB.parallelSets == 0:
        # fully-empty zonotope bundle
        return np.array([]).reshape(zB.dim(), 0)

    # MATLAB: use constrained zonotope and its center (Chebyshev if non-empty)
    cZ = zB.conZonotope()
    if cZ.representsa_('emptySet', np.finfo(float).eps):
        return np.array([]).reshape(cZ.dim(), 0)

    return cZ.center()