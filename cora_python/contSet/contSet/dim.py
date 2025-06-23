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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cora_python.contSet.contSet.contSet import ContSet

def dim(S: 'ContSet') -> int:
    """
    Returns the dimension of the ambient space of a continuous set.
    
    This function delegates to the object's dim() method if available,
    otherwise returns 0 for the base contSet class.
    
    Args:
        S: contSet object
        
    Returns:
        int: dimension of the ambient space
    """
    # Check if subclass has overridden dim method
    base_class = type(S).__bases__[0] if type(S).__bases__ else None
    if (hasattr(type(S), 'dim') and 
        base_class and hasattr(base_class, 'dim') and
        type(S).dim is not base_class.dim):
        return type(S).dim(S)
    else:
        return 0