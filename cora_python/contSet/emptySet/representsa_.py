"""
representsa_ - checks if an empty set can also be represented by a
    different set, e.g., a special case

Syntax:
    res = representsa_(O,type,tol)
    [res,S] = representsa_(O,type,tol)

Inputs:
    O - emptySet object
    type - other set representation or 'origin', 'point', 'hyperplane'
    tol - tolerance

Outputs:
    res - true/false
    S - converted set

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: contSet/representsa

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       25-July-2023 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import TYPE_CHECKING, Union, Tuple, Optional, Any

from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

if TYPE_CHECKING:
    from .emptySet import EmptySet


def representsa_(O: 'EmptySet', type_: str, tol: float = 1e-9, *varargin, **kwargs) -> bool:
    """
    Checks if an empty set can also be represented by a different set
    
    Args:
        O: emptySet object
        type_: other set representation or 'origin', 'point', 'hyperplane'
        tol: tolerance (unused for emptySet)
        *varargin: additional arguments (unused)
        
    Returns:
        res: True/False
    """
    # Import here to avoid circular imports
    
    # check empty object case
    from cora_python.contSet.contSet.representsa_emptyObject import representsa_emptyObject
    empty, res, _ = representsa_emptyObject(O, type_, return_conv=False)
    if empty:
        return res

    # dimension
    n = O.dim()

    if type_ == 'origin':
        # empty set can never be the origin
        res = False

    elif type_ == 'point':
        # empty set can never be a point
        res = False

    # all contSet can be empty
    elif type_ == 'capsule':
        res = True

    elif type_ == 'conHyperplane':
        res = True

    elif type_ == 'conPolyZono':
        res = True

    elif type_ == 'conZonotope':
        res = True

    elif type_ == 'ellipsoid':
        res = True

    elif type_ == 'halfspace':
        # empty set is always empty
        res = False

    elif type_ == 'interval':
        res = True

    elif type_ == 'levelSet':
        # not supported
        raise CORAerror('CORA:notSupported',
                       f'Comparison of emptySet to {type_} not supported.')

    elif type_ == 'polytope':
        res = True

    elif type_ == 'polyZonotope':
        res = True

    elif type_ == 'probZonotope':
        # cannot be true
        res = False

    elif type_ == 'zonoBundle':
        res = True

    elif type_ == 'zonotope':
        res = True

    elif type_ == 'hyperplane':
        # hyperplanes cannot be empty
        res = False

    elif type_ == 'parallelotope':
        res = False

    elif type_ == 'convexSet':
        res = True

    elif type_ == 'emptySet':
        # obviously true
        res = True

    elif type_ == 'emptyset':  # Handle case variations
        res = True

    elif type_ == 'fullspace':
        res = False

    else:
        # Unknown type
        res = False

    return res 