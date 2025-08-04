"""
Inf - instantiates a fullspace fullspace object

Syntax:
   fs = fullspace.Inf(n)

Inputs:
   n - dimension

Outputs:
   fs - fullspace fullspace object

Example: 
   fs = fullspace.Inf(2);

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger
Written:       09-January-2024
Last update:   15-January-2024 (TL, parse input)
Last revision: ---
Automatic python translation: Florian Nüssel BA 2025

----------------------------- BEGIN CODE ------------------------------
"""

from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck

def Inf(n=None):
    """
    Instantiates a fullspace fullspace object
    
    Args:
        n: dimension (optional, defaults to 0)
        
    Returns:
        fs: fullspace fullspace object
    """
    # parse input
    if n is None:
        n = 0
    
    inputArgsCheck([[n, 'att', 'numeric', ['scalar', 'nonnegative']]])
    
    # call constructor (and check n there)
    # Import here to avoid circular imports
    from cora_python.contSet.fullspace import Fullspace
    fs = Fullspace(n)
    
    return fs

# ------------------------------ END OF CODE ------------------------------ 