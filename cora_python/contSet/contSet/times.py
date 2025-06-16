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

from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAError

def times(factor1, factor2):
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
        CORAError: If times is not implemented for the specific set type
    """
    # Check if the first object has a times method and use it
    if hasattr(factor1, 'times') and callable(getattr(factor1, 'times')):
        return factor1.times(factor2)
    
    # Check if the second object has a times method and use it (commutative)
    if hasattr(factor2, 'times') and callable(getattr(factor2, 'times')):
        return factor2.times(factor1)
    
    # Neither is a contSet object with times method
    raise CORAError("CORA:noops", "Function times requires at least one contSet object") 