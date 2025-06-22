"""
center - returns the center of an empty set

Syntax:
    c = center(O)

Inputs:
    O - emptySet object

Outputs:
    c - center

Example: 
    O = EmptySet(2)
    c = center(O)

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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .emptySet import EmptySet

from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def center(O: 'EmptySet') -> np.ndarray:
    """
    Returns the center of an empty set
    
    Args:
        O: emptySet object
        
    Returns:
        c: center (empty array with shape (dimension, 0))
    """
    # Return empty array with shape (dimension, 0) to match MATLAB behavior
    return np.empty((O.dimension, 0)) 