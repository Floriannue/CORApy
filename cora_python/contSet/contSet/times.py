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
    
    This base implementation throws an error - to be overridden in subclasses.
    
    Args:
        factor1: contSet object or numeric
        factor2: contSet object or numeric
        
    Returns:
        contSet: result of element-wise multiplication
        
    Raises:
        CORAError: This method should be overridden in subclasses
    """
    # Determine which object is the contSet
    if hasattr(factor1, '__class__') and hasattr(factor1, 'times'):
        # is overridden in subclass if implemented; throw error
        raise CORAError("CORA:noops", f"Function times not implemented for class {type(factor1).__name__}")
    elif hasattr(factor2, '__class__') and hasattr(factor2, 'times'):
        # is overridden in subclass if implemented; throw error
        raise CORAError("CORA:noops", f"Function times not implemented for class {type(factor2).__name__}")
    else:
        # Neither is a contSet object with times method
        raise CORAError("CORA:noops", "Function times requires at least one contSet object") 