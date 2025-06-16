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

from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAError

def center(S):
    """
    Returns the center of a set.
    
    This function delegates to the object's center method if available,
    otherwise raises an error.
    
    Args:
        S: contSet object
        
    Returns:
        numpy.ndarray: center of the set
        
    Raises:
        CORAError: If center is not implemented for the specific set type
    """
    # Check if the object has a center method and use it
    if hasattr(S, 'center') and callable(getattr(S, 'center')):
        return S.center()
    
    # Fallback error
    raise CORAError("CORA:noops", f"Function center not implemented for class {type(S).__name__}") 