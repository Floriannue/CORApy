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

try:
    from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAError
except ImportError:
    # Fallback for when running from within the cora_python directory
    from g.functions.matlab.validate.postprocessing.CORAerror import CORAError

def project(S, dims):
    """
    Projects a set onto the specified dimensions.
    
    This base implementation throws an error - to be overridden in subclasses.
    
    Args:
        S: contSet object
        dims: list or array of dimensions for projection
        
    Returns:
        contSet: projected set
        
    Raises:
        CORAError: This method should be overridden in subclasses
    """
    # is overridden in subclass if implemented; throw error
    raise CORAError("CORA:noops", f"Function project not implemented for class {type(S).__name__}") 