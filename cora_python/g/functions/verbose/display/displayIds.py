"""
displayIds - Displays the center and generators of a zonotope

Syntax:
    displayIds(id, varName)

Inputs:
    id - id vector
    varName - name of id variable

Outputs:
    (to console)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: contSet/display

Authors:       Tobias Ladner
Written:       31-July-2023
Last update:   ---
Last revision: ---
                Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np


def displayIds(id: np.ndarray, varName: str) -> list:
    """
    Displays the id vector (returns list of strings)
    
    Args:
        id: id vector
        varName: name of id variable
        
    Returns:
        list: List of strings representing the formatted output
    """
    lines = []
    
    if id.size == 0:
        lines.append(f"{varName}: (0 ids)")
        return lines
    
    idMin = int(np.min(id))
    idMax = int(np.max(id))
    lines.append(f"{varName}: ({len(id)} ids)")
    
    # Check if id is a consecutive range
    if id.size > 0:
        expected_range = np.arange(idMin, idMax + 1)
        if id.size == len(expected_range) and np.array_equal(np.sort(id.flatten()), expected_range):
            # short version
            lines.append(f"    ({idMin}:{idMax})'")
        else:
            # disp all ids
            lines.append(str(id))
    
    return lines

