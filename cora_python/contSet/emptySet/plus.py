"""
plus - overloaded '+' operator for the Minkowski addition of a
    full-dimensional space and another set or vector

Syntax:
    S_out = O + S
    S_out = plus(O, S)

Inputs:
    O - emptySet object, numeric
    S - contSet object, numeric

Outputs:
    S_out - Minkowski sum

Example: 
    O = EmptySet(2)
    Z = Zonotope([1, 1], [[2, 1], [-1, 0]])
    result = O + Z

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
from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .emptySet import EmptySet
    from cora_python.contSet.contSet.contSet import ContSet

from cora_python.g.functions.helper.sets.contSet.reorder_numeric import reorder_numeric
from cora_python.g.functions.matlab.validate.check.equal_dim_check import equal_dim_check


def plus(O: Union['EmptySet', np.ndarray], S: Union['ContSet', np.ndarray]) -> 'EmptySet':
    """
    Overloaded '+' operator for the Minkowski addition of an empty set and another set or vector
    
    Args:
        O: emptySet object or numeric
        S: contSet object or numeric
        
    Returns:
        S_out: Minkowski sum (always empty set)
    """
    # Ensure that numeric is second input argument
    S_out, S = reorder_numeric(O, S)
    
    # Check dimensions of ambient space
    equal_dim_check(S_out, S)
    
    # Result is always empty set
    return S_out 