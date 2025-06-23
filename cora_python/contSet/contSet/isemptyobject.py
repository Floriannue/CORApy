"""
isemptyobject - checks whether a contSet object contains any information
    at all; consequently, the set is equivalent to the empty set 

Syntax:
    res = isemptyobject(S)

Inputs:
    S - contSet object

Outputs:
    res - true/false

Example: 
    E = ellipsoid([1,0;0,1]);
    isemptyobject(E);

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       03-June-2022
Last update:   18-August-2022 (MW, extend to class arrays)
               24-July-2023 (MW, move checks to subclasses, throw error)
Last revision: 07-February-2025 (TL, removed old, unused aux checks)
"""

from typing import TYPE_CHECKING
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

if TYPE_CHECKING:
    from cora_python.contSet.contSet.contSet import ContSet

def isemptyobject(S: 'ContSet') -> bool:
    """
    Checks whether a contSet object contains any information at all.
    
    This function delegates to the object's isemptyobject method if available,
    otherwise raises an error.
    
    Args:
        S: contSet object
        
    Returns:
        bool: True if the object is empty, False otherwise
        
    Raises:
        CORAerror: If isemptyobject is not implemented for the specific set type
    """
    # Fallback error
    raise CORAerror('CORA:noops', f"Function isemptyobject not implemented for class {type(S).__name__}") 