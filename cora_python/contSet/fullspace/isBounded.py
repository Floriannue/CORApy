"""
isBounded - determines if a set is bounded

Syntax:
   res = isBounded(fs)

Inputs:
   fs - fullSpace object

Outputs:
   res - true/false

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Tobias Ladner
Written:       14-October-2024
Last update:   ---
Last revision: ---
Automatic python translation: Florian Nüssel BA 2025

----------------------------- BEGIN CODE ------------------------------
"""

def isBounded(fs, *args):
    """
    Determines if a set is bounded
    
    Args:
        fs: fullSpace object
        *args: additional arguments (unused)
        
    Returns:
        res: true/false
    """
    res = fs.dim() == 0
    
    return res

# ------------------------------ END OF CODE ------------------------------ 