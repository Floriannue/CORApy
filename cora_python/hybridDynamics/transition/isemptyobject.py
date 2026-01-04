"""
isemptyobject - checks if a transition object is empty

Syntax:
    res = isemptyobject(trans)

Inputs:
    trans - transition object

Outputs:
    res - true/false

Example: 
    % guard set
    guard = polytope([-1,0],0,[0,1],0);

    % reset function
    reset1 = linearReset([1,0;0,-0.75]);

    % transition
    trans = transition(guard,reset1,1);

    % comparison
    res = isemptyobject(trans)
    res = isemptyobject(transition())

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger
Written:       15-May-2023
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

from typing import Any
import numpy as np


def isemptyobject(trans: Any) -> bool:
    """
    Check if a transition object is empty
    
    Args:
        trans: Transition object
    
    Returns:
        bool: True if transition is empty, False otherwise
    """
    # MATLAB: [r,c] = size(trans);
    # For now, handle single objects (not arrays)
    
    # MATLAB: check target (has to be given)
    # targ = trans(i,j).target;
    # if isnumeric(targ) && isempty(targ)
    if hasattr(trans, 'target'):
        target = trans.target
        # MATLAB: isnumeric(targ) && isempty(targ)
        if isinstance(target, np.ndarray) and target.size == 0:
            return True
    
    return False

