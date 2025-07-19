"""
box - returns the box outer-approximation of a full-dimensional space
   case R^0: R^0

Syntax:
   fs = box(fs)

Inputs:
   fs - fullspace object

Outputs:
   fs - fullspace object

Example: 
   fs = fullspace(2);
   fs = box(fs);

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

def box(fs):
    """
    Returns the box outer-approximation of a full-dimensional space
    case R^0: R^0
    
    Args:
        fs: fullspace object
        
    Returns:
        fs: fullspace object
    """
    # the fullspace is already its box outer-approximation
    return fs

# ------------------------------ END OF CODE ------------------------------ 