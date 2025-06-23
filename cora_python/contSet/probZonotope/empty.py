"""
empty - instantiates an empty probabilistic zonotope

Syntax:
    pZ = empty(n)

Inputs:
    n - dimension

Outputs:
    pZ - empty probZonotope object

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       09-January-2024 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import TYPE_CHECKING

from cora_python.contSet.probZonotope.probZonotope import ProbZonotope
from cora_python.contSet.zonotope.zonotope import Zonotope

if TYPE_CHECKING:
    pass


def empty(n: int = 0) -> ProbZonotope:
    """
    Instantiates an empty probabilistic zonotope
    
    Args:
        n: dimension (default: 0)
        
    Returns:
        pZ: empty probZonotope object
    """
    
    # Create empty zonotope
    empty_zono = Zonotope.empty(n)
    
    # Create empty probabilistic zonotope
    g = np.zeros((n, 0)) if n > 0 else np.zeros((0, 0))
    gamma = 2  # Default gamma value
    
    return ProbZonotope(empty_zono, g, gamma) 