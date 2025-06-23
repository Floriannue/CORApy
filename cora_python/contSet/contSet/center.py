"""
center - returns the center of a set

Syntax:
    c = center(S)

Inputs:
    S - contSet object

Outputs:
    c - numeric

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: -

Authors:       Tobias Ladner (MATLAB)
               Python translation by AI Assistant
Written:       12-September-2023
Last update:   ---
Last revision: ---
"""

import numpy as np
from typing import TYPE_CHECKING
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

if TYPE_CHECKING:
    from cora_python.contSet.contSet.contSet import ContSet

def center(S: 'ContSet') -> np.ndarray:
    """
    Returns the center of a set.
    
    This function delegates to the object's center method if available,
    otherwise raises an error.
    
    Args:
        S: contSet object
        
    Returns:
        numpy.ndarray: center of the set
        
    Raises:
        CORAerror: If center is not implemented for the specific set type
    """
    
    # Fallback error
    raise CORAerror("CORA:noops", f"Function center not implemented for class {type(S).__name__}") 