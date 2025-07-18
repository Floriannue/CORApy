"""
project - projects an empty set onto the specified dimensions

Syntax:
   O = project(O,dims)

Inputs:
   O - emptySet object
   dims - dimensions for projection

Outputs:
   O - projected emptySet object

Example: 
   O = emptySet(4);
   project(O,[1,3])

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger
Written:       22-March-2023
Last update:   ---
Last revision: ---
Automatic python translation: Florian Nüssel BA 2025
"""
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.contSet.emptySet import EmptySet

def project(self, dims):
    """
    Projects an empty set onto the specified dimensions
    
    Args:
        dims: dimensions for projection (1-based indexing like MATLAB)
        
    Returns:
        O: projected emptySet object
        
    Raises:
        CORAerror: if dimensions are out of valid range
    """

    # Validate dimensions (MATLAB uses 1-based indexing)
    if any(d < 1 or d > self.dimension for d in dims):
        raise CORAerror('CORA:outOfDomain', f'1:{self.dimension}')
    else:
        # Return new emptySet with projected dimension
        return EmptySet(len(dims)) 