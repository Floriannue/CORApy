"""
project - projects a set onto the specified dimensions

Description:
    computes the set { [s_{(dims_{(1)})}, ..., s_{(dims_{(m)})}]^T | s \in \mathcal{S} } \subset \R^m

Syntax:
    S = project(S, dims)

Inputs:
    S - contSet object
    dims - dimensions of projection

Outputs:
    S - contSet object

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: contSet/projectHighDim, contSet/lift

Authors:       Tobias Ladner (MATLAB)
               Python translation by AI Assistant
Written:       12-September-2023
Last update:   ---
Last revision: ---
"""
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAError

def project(S, dims):
    """
    Projects a set onto the specified dimensions.
    
    This function delegates to the object's project method if available,
    otherwise raises an error.
    
    Args:
        S: contSet object
        dims: list or array of dimensions for projection
        
    Returns:
        contSet: projected set
        
    Raises:
        CORAError: If project is not implemented for the specific set type
    """
    # Check if the object has a project method and use it
    if hasattr(S, 'project') and callable(getattr(S, 'project')):
        return S.project(dims)
    
    # Fallback error
    raise CORAError("CORA:noops", f"Function project not implemented for class {type(S).__name__}") 