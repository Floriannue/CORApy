"""
display - displays the properties of a polyZonotope object (center,
    dependent generator matrix, exponent matrix, independent generator
    matrix, identifiers) on the command window

Syntax:
    display(pZ)

Inputs:
    pZ - polyZonotope object

Outputs:
    ---

Example: 
    pZ = polyZonotope([2;1],[1 0; -2 1],[1; 0],[0 2; 1 0])

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger
Written:       02-May-2020
Last update:   09-June-2020 (show values)
Last revision: ---
                Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import TYPE_CHECKING
from cora_python.contSet.contSet.representsa import representsa
from cora_python.g.functions.verbose.display.dispEmptySet import dispEmptySet
from cora_python.g.functions.verbose.display.dispRn import dispRn
from cora_python.g.functions.verbose.display.displayGenerators import displayGenerators
from cora_python.g.functions.verbose.display.displayIds import displayIds
from cora_python.g.macros import DISPLAYDIM_MAX
from cora_python.contSet.contSet.display import display_ as contSet_display
from cora_python.contSet.contSet.center import center

if TYPE_CHECKING:
    from .polyZonotope import PolyZonotope


def display_(pZ: 'PolyZonotope', var_name: str = None) -> str:
    """
    Displays the properties of a polyZonotope object (internal function that returns string)
    
    Args:
        pZ: polyZonotope object
        var_name: Optional variable name
        
    Returns:
        str: String representation
    """
    lines = []
    
    # Special cases
    if representsa(pZ, 'emptySet'):
        if var_name:
            return dispEmptySet(pZ, var_name)
        return dispEmptySet(pZ)
    elif representsa(pZ, 'fullspace'):
        if var_name:
            return dispRn(pZ, var_name)
        return dispRn(pZ)
    
    # Display input variable
    lines.append("")
    if var_name is None:
        var_name = 'pZ'
    lines.append(f"{var_name} =")
    lines.append("")
    
    # Display dimension
    contSet_str = contSet_display(pZ)
    lines.append(contSet_str)
    lines.append("")
    
    # Display center
    lines.append("c: ")
    center_val = center(pZ)
    lines.append(str(center_val))
    lines.append("")
    
    # Display dependent generators
    if hasattr(pZ, 'G') and pZ.G is not None and pZ.G.size > 0:
        gen_lines = displayGenerators(pZ.G, DISPLAYDIM_MAX(), 'G')
        lines.extend(gen_lines)
        lines.append("")
    
    # Display independent generators
    if hasattr(pZ, 'GI') and pZ.GI is not None and pZ.GI.size > 0:
        gen_lines = displayGenerators(pZ.GI, DISPLAYDIM_MAX(), 'GI')
        lines.extend(gen_lines)
        lines.append("")
    
    # Display exponential matrix
    if hasattr(pZ, 'E') and pZ.E is not None and pZ.E.size > 0:
        gen_lines = displayGenerators(pZ.E, DISPLAYDIM_MAX(), 'E')
        lines.extend(gen_lines)
        lines.append("")
    
    # Display id
    if hasattr(pZ, 'id') and pZ.id is not None and pZ.id.size > 0:
        id_lines = displayIds(pZ.id, 'id')
        lines.extend(id_lines)
        lines.append("")
    
    return "\n".join(lines)


def display(pZ: 'PolyZonotope', var_name: str = None) -> None:
    """
    Displays the properties of a polyZonotope object (prints to stdout)
    
    Args:
        pZ: polyZonotope object
        var_name: Optional variable name
    """
    print(display_(pZ, var_name), end='')

