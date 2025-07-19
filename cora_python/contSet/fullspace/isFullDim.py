"""
isFullDim - checks if the dimension of the affine hull of a full-dim.
   space is equal to the dimension of its ambient space
   case R^0: true

Syntax:
   res = isFullDim(fs)

Inputs:
   fs - fullspace object

Outputs:
   res - true/false

Example: 
   fs = fullspace(2);
   res = isFullDim(fs);

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger
Written:       22-March-2023
Last update:   ---
Last revision: ---
Automatic python translation: Florian Nüssel BA 2025

----------------------------- BEGIN CODE ------------------------------
"""

def isFullDim(fs):
    """
    Checks if the dimension of the affine hull of a full-dim.
    space is equal to the dimension of its ambient space
    case R^0: true
    
    Args:
        fs: fullspace object
        
    Returns:
        res: true/false
    """
    res = True
    
    return res

# ------------------------------ END OF CODE ------------------------------ 