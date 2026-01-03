"""
isemptyobject - checks if linearReset object is empty

Syntax:
    res = isemptyobject(linReset)

Inputs:
    linReset - linearReset object

Outputs:
    res - true/false

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger
Written:       15-May-2023
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

from typing import Any, Union, List


def isemptyobject(linReset: Union[Any, List[Any]]) -> Union[bool, List[bool]]:
    """
    Checks if linearReset object is empty
    
    Args:
        linReset: linearReset object or list of linearReset objects
        
    Returns:
        res: True if empty, False otherwise (or list of bools for array input)
    """
    # Handle array input
    if isinstance(linReset, list):
        return [isemptyobject(r) for r in linReset]
    
    import numpy as np
    
    # An empty linearReset has all its matrices/vectors as empty arrays
    # and its dimensions as 0 or 1 (for inputDim default)
    # Check if arrays are empty (size == 0) and dimensions match MATLAB default
    A_empty = (isinstance(linReset.A, np.ndarray) and linReset.A.size == 0)
    B_empty = (isinstance(linReset.B, np.ndarray) and linReset.B.size == 0)
    c_empty = (isinstance(linReset.c, np.ndarray) and linReset.c.size == 0)
    if A_empty and B_empty and c_empty and \
       linReset.preStateDim == 0 and \
       linReset.inputDim == 1 and \
       linReset.postStateDim == 0:
        return True
    return False

