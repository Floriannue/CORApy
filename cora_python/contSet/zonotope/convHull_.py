"""
convHull_ - computes an enclosure for the convex hull of a zonotope and
    another set or a point

Syntax:
    Z = convHull_(Z, S)

Inputs:
    Z - zonotope object
    S - contSet object

Outputs:
    Z - zonotope enclosing the convex hull

Example: 
    Z1 = Zonotope(np.array([[2, 1, 0], [2, 0, 1]]))
    Z2 = Zonotope(np.array([[-2, 1, 0], [-2, 0, 1]]))

    Z = convHull_(Z1, Z2)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: contSet/convHull, conZonotope/convHull_

Authors: Niklas Kochdumper (MATLAB)
         Python translation by AI Assistant
Written: 26-November-2019 (MATLAB)
Last update: 29-September-2024 (MW, integrate precedence) (MATLAB)
         2025 (Tiange Yang, Florian Nüssel, Python translation by AI Assistant)
"""

import numpy as np
from typing import Optional, Union, TYPE_CHECKING
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from .zonotope import Zonotope
from cora_python.g.functions.helper.sets.contSet.contSet.reorder_numeric import reorder_numeric

if TYPE_CHECKING:
    from cora_python.contSet.contSet import ContSet
            



def convHull_(Z: 'Zonotope', S: Optional[Union['ContSet', np.ndarray]] = None, method: str = 'exact') -> 'Zonotope':
    """
    Computes an enclosure for the convex hull of a zonotope and another set or a point
    
    Args:
        Z: Zonotope object
        S: contSet object or numeric (optional)
        method: Method for computation (default: 'exact')
        
    Returns:
        Zonotope: Zonotope enclosing the convex hull
        
    Example:
        >>> Z1 = Zonotope([2, 2], [[1, 0], [0, 1]])
        >>> Z2 = Zonotope([-2, -2], [[1, 0], [0, 1]])
        >>> Z = convHull_(Z1, Z2)
    """
    
    # Zonotope is already convex
    if S is None:
        return Z

    # 新增：自身与自身的convex hull直接返回自身
    if S is Z:
        return Z
    
    # Ensure that numeric is second input argument (reorder if necessary)
    Z_out, S = reorder_numeric(Z, S)
    
    # Check dimensions
    if hasattr(S, 'dim') and hasattr(Z_out, 'dim'):
        if S.dim() != Z_out.dim():
            raise CORAerror('CORA:dimensionMismatch',
                          f'Dimension mismatch: {Z_out.dim()} vs {S.dim()}')
    
    # Call function with lower precedence if applicable
    if hasattr(S, 'precedence') and hasattr(Z_out, 'precedence') and S.precedence < Z_out.precedence:
        return S.convHull(Z_out, method)
    
    # Convex hull with empty set
    # Only check representsa_ for objects that have the necessary methods
    if hasattr(S, '__class__') and hasattr(S, 'isemptyobject') and S.representsa_('emptySet', 1e-15):
        return Z_out
    elif Z_out.representsa_('emptySet', 1e-15):
        return S if isinstance(S, Zonotope) else Zonotope(S, np.array([]).reshape(len(S), 0))
    
    # Use enclose method
    if isinstance(S, Zonotope):
        S_zono = S
    else:
        # Convert S to zonotope
        S_zono = Zonotope(S)
    
    return Z_out.enclose(S_zono)