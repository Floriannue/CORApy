"""
isFullDim - checks if the dimension of the affine hull of an empty set is
   equal to the dimension of its ambient space

Syntax:
   res = isFullDim(O)

Inputs:
   O - emptySet object

Outputs:
   res - true/false

Example: 
   O = emptySet(2);
   res = isFullDim(O);

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

def isFullDim(self):
    """
    Checks if the dimension of the affine hull of an empty set is
    equal to the dimension of its ambient space
    
    Returns:
        bool: Always False for empty sets
    """
    # always false
    return False 