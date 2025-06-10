"""
dim - returns the dimension of the ambient space of a continuous set;
    currently only used for the empty set constructor contSet()

Syntax:
    n = dim(S)

Inputs:
    S - contSet object

Outputs:
    n - dimension of the ambient space

Example: 
    S = contSet();
    n = dim(S)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       17-June-2022
Last update:   22-March-2023 (MW, adapt to new constructor syntax)
Last revision: ---
"""

def dim(S):
    """
    Returns the dimension of the ambient space of a continuous set.
    
    This function delegates to the object's dim() method if available,
    otherwise returns 0 for the base contSet class.
    
    Args:
        S: contSet object
        
    Returns:
        int: dimension of the ambient space
    """
    # Check if the object has a dim method and use it
    if hasattr(S, 'dim') and callable(getattr(S, 'dim')):
        return S.dim()
    
    # Fallback for base contSet objects
    return 0 