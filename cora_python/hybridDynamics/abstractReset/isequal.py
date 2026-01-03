"""
isequal - checks if two reset functions have equal pre-/post-state and
   input dimensions

Syntax:
    res = isequal(reset1,reset2)

Inputs:
    reset1 - abstractReset object
    reset2 - abstractReset object

Outputs:
    res - true/false

Example: 
    reset1 = abstractReset(2,1,2);
    reset2 = abstractReset(2,1,3);
    isequal(reset1,reset1);
    isequal(reset1,reset2);

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger
Written:       09-October-2024
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

from typing import Any, Optional


def isequal(reset1: Any, reset2: Any, *args, **kwargs) -> bool:
    """
    Checks if two reset functions have equal pre-/post-state and input dimensions
    
    Args:
        reset1: abstractReset object
        reset2: abstractReset object
        *args: Optional tolerance arguments (for compatibility)
        **kwargs: Optional keyword arguments (for compatibility)
        
    Returns:
        res: True if dimensions match, False otherwise
    """
    # compare number of states and inputs
    res = (reset1.preStateDim == reset2.preStateDim 
           and reset1.inputDim == reset2.inputDim 
           and reset1.postStateDim == reset2.postStateDim)
    
    return res

