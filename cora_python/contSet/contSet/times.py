"""
times - element-wise multiplication for contSet objects

This function implements the .* operator for contSet objects.
The base implementation throws an error - to be overridden in subclasses.

Syntax:
    res = times(factor1, factor2)

Inputs:
    factor1 - contSet object or numeric
    factor2 - contSet object or numeric

Outputs:
    res - contSet object

Authors:       Tobias Ladner (MATLAB)
               Python translation by AI Assistant
Written:       06-April-2023
Last update:   ---
Last revision: ---
"""

from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cora_python.contSet.contSet.contSet import ContSet

def times(factor1: 'ContSet', factor2: 'ContSet'):
    """
    Element-wise multiplication for contSet objects.
    
    This function delegates to the object's times method if available,
    otherwise raises an error.
    
    Args:
        factor1: contSet object or numeric
        factor2: contSet object or numeric
        
    Returns:
        contSet: result of element-wise multiplication
        
    Raises:
        CORAerror: If times is not implemented for the specific set type
    """
    
    # Neither is a contSet object with times method
    raise CORAerror("CORA:noops", "Function times requires at least one contSet object") 