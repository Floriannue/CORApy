"""
display - displays the properties of a conPolyZono object (center,
    generator matrices, exponent matrix, constraint system) on the 
    command window

Syntax:
    display(cPZ)

Inputs:
    cPZ - conPolyZono object

Outputs:
    ---

Example: 
    c = [0;0];
    G = [1 0 1 -1; 0 1 1 1];
    E = [1 0 1 2; 0 1 1 0; 0 0 1 1];
    A = [1 -0.5 0.5];
    b = 0.5;
    EC = [0 1 2; 1 0 0; 0 1 0];
 
    cPZ = conPolyZono(c,G,E,A,b,EC)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: polyZonotope/display

Authors:       Niklas Kochdumper
Written:       19-January-2021
Last update:   ---
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

if TYPE_CHECKING:
    from .conPolyZono import ConPolyZono


def display_(cPZ: 'ConPolyZono', var_name: str = None) -> str:
    """
    Displays the properties of a conPolyZono object (internal function that returns string)
    
    Args:
        cPZ: conPolyZono object
        var_name: Optional variable name
        
    Returns:
        str: String representation
    """
    lines = []
    
    # Special cases
    if representsa(cPZ, 'emptySet'):
        if var_name:
            return dispEmptySet(cPZ, var_name)
        return dispEmptySet(cPZ)
    elif representsa(cPZ, 'fullspace'):
        if var_name:
            return dispRn(cPZ, var_name)
        return dispRn(cPZ)
    
    # Display input variable
    lines.append("")
    if var_name is None:
        var_name = 'cPZ'
    lines.append(f"{var_name} =")
    lines.append("")
    
    # Display dimension
    contSet_str = contSet_display(cPZ)
    lines.append(contSet_str)
    lines.append("")
    
    # Display center
    lines.append("c: ")
    lines.append(str(cPZ.c))
    lines.append("")
    
    # Display generators
    if hasattr(cPZ, 'G') and cPZ.G is not None and cPZ.G.size > 0:
        gen_lines = displayGenerators(cPZ.G, DISPLAYDIM_MAX(), 'G')
        lines.extend(gen_lines)
        lines.append("")
    
    # Display exponential matrix
    if hasattr(cPZ, 'E') and cPZ.E is not None and cPZ.E.size > 0:
        gen_lines = displayGenerators(cPZ.E, DISPLAYDIM_MAX(), 'E')
        lines.extend(gen_lines)
        lines.append("")
    
    # Display constraint offset
    lines.append("b:")
    if hasattr(cPZ, 'b') and cPZ.b is not None:
        lines.append(str(cPZ.b))
    else:
        lines.append("[]")
    lines.append("")
    
    # Display constraint generators
    if hasattr(cPZ, 'A') and cPZ.A is not None and cPZ.A.size > 0:
        gen_lines = displayGenerators(cPZ.A, DISPLAYDIM_MAX(), 'A')
        lines.extend(gen_lines)
        lines.append("")
    
    # Display constraint exponential matrix
    if hasattr(cPZ, 'EC') and cPZ.EC is not None and cPZ.EC.size > 0:
        gen_lines = displayGenerators(cPZ.EC, DISPLAYDIM_MAX(), 'EC')
        lines.extend(gen_lines)
        lines.append("")
    
    # Display independent generators
    if hasattr(cPZ, 'GI') and cPZ.GI is not None and cPZ.GI.size > 0:
        gen_lines = displayGenerators(cPZ.GI, DISPLAYDIM_MAX(), 'GI')
        lines.extend(gen_lines)
        lines.append("")
    
    # Display id
    if hasattr(cPZ, 'id') and cPZ.id is not None and cPZ.id.size > 0:
        id_lines = displayIds(cPZ.id, 'id')
        lines.extend(id_lines)
        lines.append("")
    
    return "\n".join(lines)


def display(cPZ: 'ConPolyZono', var_name: str = None) -> None:
    """
    Displays the properties of a conPolyZono object (prints to stdout)
    
    Args:
        cPZ: conPolyZono object
        var_name: Optional variable name
    """
    print(display_(cPZ, var_name), end='')

