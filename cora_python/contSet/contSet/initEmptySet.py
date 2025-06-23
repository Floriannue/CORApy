"""
initEmptySet - instantiates an empty set of a contSet class

This function is deprecated since CORA 2024.1.0 and has been replaced by 'empty'.
Note that the function 'initEmptySet' will be removed in a future release.

Syntax:
    S = initEmptySet(type)

Inputs:
    type - ContSet class name

Outputs:
    S - instantiated set

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
"""
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cora_python.contSet.contSet.contSet import ContSet

def initEmptySet(set_type: str) -> 'ContSet':
    """
    initEmptySet - instantiates an empty set of a contSet class
    
    This function is deprecated and throws an error directing users to use
    the 'empty' method instead.
    
    Args:
        set_type: ContSet class name
        
    Raises:
        CORAerror: This function is deprecated, use 'empty' instead
    """
    raise CORAerror('CORA:specialError', 
        "The function 'contSet.initEmptySet' is deprecated (since CORA 2024.1.0) and has been replaced by 'contSet.empty'.\n" +
        "Note that the function 'initEmptySet' will be removed in a future release.") 