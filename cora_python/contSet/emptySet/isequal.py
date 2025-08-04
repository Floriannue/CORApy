"""
isequal - checks if an empty set is equal to another set or point

Syntax:
    res = isequal(O,S)
    res = isequal(O,S,tol)

Inputs:
    O - emptySet object
    S - contSet object or numerical vector
    tol - (optional) tolerance

Outputs:
    res - true/false

Example: 
    O1 = EmptySet(2)
    O2 = EmptySet(3)
    res1 = isequal(O1, O1)
    res2 = isequal(O1, O2)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       22-March-2023 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from .emptySet import EmptySet
    from cora_python.contSet.contSet.contSet import ContSet


def isequal(O: 'EmptySet', S: Union['ContSet', np.ndarray], tol: float = 1e-15) -> bool:
    """
    Checks if an empty set is equal to another set or point
    
    Args:
        O: emptySet object
        S: contSet object or numerical vector
        tol: tolerance (optional)
        
    Returns:
        res: True/False
    """
    from cora_python.g.functions.matlab.validate.check import inputArgsCheck
    from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
    
    # check input arguments (simplified version)
    # inputArgsCheck would be more complex in full implementation
    
    # Check if S is an emptySet
    if hasattr(S, '__class__') and S.__class__.__name__ == 'EmptySet':
        # note: tolerance has no effect, only for overloading purposes
        return O.dimension == S.dimension

    # other contSet classes
    if hasattr(S, 'dim') and hasattr(S, 'representsa_'):
        try:
            return S.dim() == O.dimension and S.representsa_('emptySet', tol)
        except:
            return False

    # Check if S has representsa method (for base contSet compatibility)
    if hasattr(S, 'representsa'):
        try:
            return S.dim() == O.dimension and S.representsa('emptySet', tol)
        except:
            return False

    # vector/numpy array
    if isinstance(S, np.ndarray):
        # Empty array with correct number of elements
        return S.size == 0 and len(S.shape) > 0 and (
            S.shape[0] == O.dimension or 
            (len(S.shape) == 2 and S.shape[0] * S.shape[1] == O.dimension and S.size == 0)
        )

    # If we get here, types are incompatible
    return False 