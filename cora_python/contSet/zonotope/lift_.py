"""
lift_ - lifts a zonotope onto a higher-dimensional space

Syntax:
    Z = lift_(Z, N, dims)

Inputs:
    Z - zonotope object
    N - dimension of the higher-dimensional space
    proj - states of the high-dimensional space that correspond to the
          states of the low-dimensional space

Outputs:
    Z - projected zonotope

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: contSet/lift, contSet/projectHighDim

Authors:       Tobias Ladner (MATLAB)
               Python translation by AI Assistant
Written:       19-September-2023 (MATLAB)
                2025 (Tiange Yang, Florian Nüssel, Python translation by AI Assistant)
"""

import numpy as np
from typing import Optional, List
from .zonotope import Zonotope
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def lift_(Z: Zonotope, N: int, proj: List[int]) -> Zonotope:
    """
    Lifts a zonotope onto a higher-dimensional space
    
    Args:
        Z: zonotope object
        N: dimension of the higher-dimensional space
        proj: states of the high-dimensional space that correspond to the
              states of the low-dimensional space
              
    Returns:
        Projected zonotope
        
    Example:
        Z = Zonotope(np.array([[0], [0]]), np.array([[1, 0], [0, 1]]))
        Z_lifted = lift_(Z, 3, [0, 1])
    """
    # Check for None values
    if Z.c is None or Z.G is None:
        raise CORAerror('CORA:wrongInputInConstructor', 
                       'Zonotope center or generators are None')
    
    if len(proj) == N:
        # Use project
        from .project import project
        Z = project(Z, proj)
    else:
        # Projection to higher dimension is not defined as function expects new
        # dimensions to be unbounded
        raise CORAerror('CORA:notDefined', 
                       'New dimensions cannot be unbounded as the set representation is always bounded.', 
                       'contSet/projectHighDim')
    
    return Z 