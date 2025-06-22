"""
display - Displays the properties of a capsule object (center, generator,
   radius) on the command window

Syntax:
   display(C)

Inputs:
   C - capsule

Outputs:
   ---

Example: 
   C = capsule([1;1],[1;0],0.5);
   display(C);

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors: Matthias Althoff (MATLAB)
         Python translation by AI Assistant
Written: 04-March-2019 (MATLAB)
Last update: 02-May-2020 (MW, add empty case)
Python translation: 2025
"""

import numpy as np


def display(C) -> str:
    """
    Displays the properties of a capsule object
    
    Args:
        C: capsule object
        
    Returns:
        str: formatted display string
    """
    lines = []
    lines.append("")
    
    # Check for empty capsule
    if C.is_empty():
        lines.append(f"Empty capsule in R^{C.dim()}")
        lines.append("")
        return "\n".join(lines)
    
    lines.append("capsule")
    lines.append("")
    
    # Display dimension
    lines.append(f"dimension: {C.dim()}")
    lines.append("")
    
    # Display center
    lines.append("center:")
    if C.c is not None:
        lines.append(str(C.c.flatten()))
    else:
        lines.append("[]")
    lines.append("")
    
    # Display generator
    lines.append("generator:")
    if C.g is not None:
        lines.append(str(C.g.flatten()))
    else:
        lines.append("[]")
    lines.append("")
    
    # Display radius
    lines.append("radius:")
    lines.append(str(C.r))
    lines.append("")
    
    return "\n".join(lines) 