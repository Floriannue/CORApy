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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .capsule import Capsule

def display_(C: 'Capsule', var_name: str = None) -> str:
    """
    Displays the properties of a capsule object (internal function that returns string)
    
    Args:
        C: capsule object
        var_name: Optional variable name
        
    Returns:
        str: formatted display string
    """
    from cora_python.contSet.contSet.representsa import representsa
    from cora_python.g.functions.verbose.display.dispEmptySet import dispEmptySet
    from cora_python.g.functions.verbose.display.dispRn import dispRn
    from cora_python.contSet.contSet.display import display_ as contSet_display
    
    # Special cases
    if representsa(C, 'emptySet'):
        return dispEmptySet(C, var_name)
    elif representsa(C, 'fullspace'):
        return dispRn(C, var_name)
    
    lines = []
    lines.append("")
    
    # Display input variable
    if var_name is None:
        var_name = 'C'
    lines.append(f"{var_name} =")
    lines.append("")
    
    # Display dimension (using contSet display)
    contSet_str = contSet_display(C)
    lines.append(contSet_str)
    lines.append("")
    
    # Display center
    lines.append("center: ")
    if C.c is not None:
        lines.append(str(C.c))
    else:
        lines.append("[]")
    lines.append("")
    
    # Display generator
    lines.append("generator: ")
    if C.g is not None:
        lines.append(str(C.g))
    else:
        lines.append("[]")
    lines.append("")
    
    # Display radius
    lines.append("radius: ")
    lines.append(str(C.r))
    lines.append("")
    
    return "\n".join(lines)


def display(C: 'Capsule', var_name: str = None) -> None:
    """
    Displays the properties of a capsule object (prints to stdout)
    
    Args:
        C: capsule object
        var_name: Optional variable name
    """
    print(display_(C, var_name), end='') 