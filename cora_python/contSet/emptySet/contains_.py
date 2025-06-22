"""
contains_ - determines if an empty set contains a set or a point

Syntax:
    [res,cert,scaling] = contains_(O,S,method,tol,maxEval,certToggle,scalingToggle)

Inputs:
    O - emptySet object
    S - contSet object or numerical vector
    method - method used for the containment check.
       Currently, the only available options are 'exact' and 'approx'.
    tol - tolerance for the containment check; not used here.
    maxEval - Currently has no effect
    certToggle - if set to 'true', cert will be computed (see below),
       otherwise cert will be set to NaN.
    scalingToggle - if set to 'true', scaling will be computed (see
       below), otherwise scaling will be set to inf.

Outputs:
    res - true/false
    cert - certificate (not used for emptySet)
    scaling - scaling factor

Example: 
    O = EmptySet(2)
    p = np.array([[1], [1]])
    res, cert, scaling = contains_(O, p)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: contSet/contains

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       22-March-2023 (MATLAB)
Last update:   05-April-2023 (MW, rename contains_) (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import TYPE_CHECKING, Union, Tuple

if TYPE_CHECKING:
    from .emptySet import EmptySet
    from cora_python.contSet.contSet.contSet import ContSet


def contains_(O: 'EmptySet', S: Union['ContSet', np.ndarray], method: str = 'exact', 
              tol: float = 1e-9, maxEval: int = 1000, certToggle: bool = False, 
              scalingToggle: bool = True, *varargin) -> Tuple[bool, Union[bool, float], float]:
    """
    Determines if an empty set contains a set or a point
    
    Args:
        O: emptySet object
        S: contSet object or numerical vector
        method: method used for the containment check ('exact' or 'approx')
        tol: tolerance for the containment check (not used here)
        maxEval: Currently has no effect
        certToggle: if True, cert will be computed, otherwise cert will be NaN
        scalingToggle: if True, scaling will be computed, otherwise scaling will be inf
        *varargin: additional arguments (unused)
        
    Returns:
        res: True/False
        cert: certificate (True for empty set)
        scaling: scaling factor
    """
    # dimensions are already checked...
    res = False
    cert = True
    scaling = np.inf
    
    # Check if S is an emptySet
    if hasattr(S, '__class__') and S.__class__.__name__ == 'EmptySet':
        # empty set contains the empty set
        res = True
        scaling = 0.0

    # Check if S is a contSet that represents an empty set
    elif hasattr(S, 'representsa'):
        try:
            if S.representsa('emptySet', tol):
                # empty set contains contSet objects if they also represent the empty set
                res = True
                scaling = 0.0
        except:
            pass

    # Check if S is an empty numerical array
    elif isinstance(S, np.ndarray) and S.size == 0:
        # empty set contains empty vectors
        res = True
        scaling = 0.0

    # Handle cert and scaling toggles
    if not certToggle:
        cert = np.nan
    if not scalingToggle:
        scaling = np.inf

    return res, cert, scaling 