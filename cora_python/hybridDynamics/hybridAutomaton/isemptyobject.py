"""
isemptyobject - checks if a hybrid automaton object is empty

Syntax:
    res = isemptyobject(HA)

Inputs:
    HA - hybridAutomaton object

Outputs:
    res - true/false

Example: 
    ---

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger
Written:       16-May-2023
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

from typing import Any
import numpy as np


def isemptyobject(HA: Any) -> bool:
    """
    Check if a hybrid automaton object is empty
    
    Args:
        HA: HybridAutomaton object
    
    Returns:
        bool: True if hybrid automaton is empty, False otherwise
    """
    # MATLAB: [r,c] = size(HA);
    # For now, handle single objects (not arrays)
    
    # MATLAB: res(i,j) = all(isemptyobject(HA(i,j).location));
    if hasattr(HA, 'location') and HA.location is not None:
        if isinstance(HA.location, (list, tuple, np.ndarray)):
            if len(HA.location) == 0:
                return True
            # Check if all locations are empty
            all_empty = True
            for loc in HA.location:
                if hasattr(loc, 'isemptyobject'):
                    if not loc.isemptyobject():
                        all_empty = False
                        break
                else:
                    all_empty = False
                    break
            return all_empty
        elif isinstance(HA.location, np.ndarray) and HA.location.size == 0:
            return True
    
    return False

