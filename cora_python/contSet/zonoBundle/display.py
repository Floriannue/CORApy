"""
display - Displays the properties of a zonoBundle object (center and 
    generator matrix of each zonotope) on the command window

Syntax:
    display(zB)

Inputs:
    zB - zonoBundle object

Outputs:
    str - display string

Example: 
    Z1 = zonotope(zeros(2,1),[1 0.5; -0.2 1]);
    Z2 = zonotope(ones(2,1),[1 -0.5; 0.2 1]);
    zB = zonoBundle({Z1,Z2});
    display(zB)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Matthias Althoff (MATLAB)
               Python translation by AI Assistant
Written:       09-November-2010 (MATLAB)
Last update:   02-May-2020 (MW, add empty case, MATLAB)
               09-June-2020 (MW, remove dependency from zonotope/display, MATLAB)
Last revision: ---
Python translation: 2025
"""

import numpy as np
from typing import TYPE_CHECKING
from cora_python.contSet.contSet.representsa import representsa

if TYPE_CHECKING:
    from cora_python.contSet.zonoBundle.zonoBundle import ZonoBundle


def display_(zB: 'ZonoBundle') -> str:
    """
    Display the properties of a zonoBundle object (internal function that returns string)
    
    Args:
        zB: zonoBundle object
        
    Returns:
        str: Display string
    """
    # Special cases
    if representsa(zB, 'emptySet'):
        from cora_python.g.functions.verbose.display.dispEmptySet import dispEmptySet
        return dispEmptySet(zB)
    elif representsa(zB, 'fullspace'):
        from cora_python.g.functions.verbose.display.dispRn import dispRn
        return dispRn(zB)
    
    # Build display string
    lines = []
    
    # Display dimension
    lines.append(f"zonoBundle in R^{zB.dim()}")
    lines.append("")
    
    # Cap number of generators
    maxGens = 10
    
    # Display each zonotope
    for i in range(zB.parallelSets):
        lines.append(f"zonotope {i+1}:")
        lines.append("")
        
        # Display center
        lines.append("c: ")
        center_val = zB.Z[i].center()
        lines.append(str(center_val.flatten()))
        
        # Display generators
        G = zB.Z[i].generators()
        lines.append("G: ")
        if G.shape[1] <= maxGens:
            lines.append(str(G))
        else:
            # Show first few generators and indicate truncation
            lines.append(str(G[:, :maxGens]))
            lines.append(f"... ({G.shape[1]} generators total)")
        
        lines.append("")
    
    return "\n".join(lines)


def display(zB: 'ZonoBundle') -> None:
    """
    Display the properties of a zonoBundle object (prints to stdout)
    
    Args:
        zB: zonoBundle object
    """
    print(display_(zB), end='') 