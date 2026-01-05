"""
display - Displays the properties of a conZonotope object (center,
    generators, and constraints for the factors) on the command window

Syntax:
    display(cZ)

Inputs:
    cZ - conZonotope object

Outputs:
    ---

Example: 
    Z = [0 1 0 1;0 1 2 -1];
    A = [-2 1 -1]; b = 2;
    cZ = conZonotope(Z,A,b)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Dmitry Grebenyuk, Mark Wetzlinger
Written:       20-December-2017
Last update:   01-May-2020 (MW, added empty case)
               09-June-2020 (MW, restrict number of shown generators)
Last revision: ---
                Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import TYPE_CHECKING
from cora_python.contSet.contSet.representsa import representsa
from cora_python.g.functions.verbose.display.dispEmptySet import dispEmptySet
from cora_python.g.functions.verbose.display.dispRn import dispRn
from cora_python.g.functions.verbose.display.displayGenerators import displayGenerators
from cora_python.g.macros import DISPLAYDIM_MAX
from cora_python.contSet.contSet.display import display_ as contSet_display

if TYPE_CHECKING:
    from .conZonotope import ConZonotope


def display_(cZ: 'ConZonotope', var_name: str = None) -> str:
    """
    Displays the properties of a conZonotope object (internal function that returns string)
    
    Args:
        cZ: conZonotope object
        var_name: Optional variable name
        
    Returns:
        str: String representation
    """
    lines = []
    
    # Special cases
    if representsa(cZ, 'emptySet'):
        if var_name:
            return dispEmptySet(cZ, var_name)
        return dispEmptySet(cZ)
    elif representsa(cZ, 'fullspace'):
        if var_name:
            return dispRn(cZ, var_name)
        return dispRn(cZ)
    
    # Display input variable
    lines.append("")
    if var_name is None:
        var_name = 'cZ'
    lines.append(f"{var_name} =")
    lines.append("")
    
    # Display dimension
    contSet_str = contSet_display(cZ)
    lines.append(contSet_str)
    lines.append("")
    
    # Display center and generators
    lines.append("c: ")
    lines.append(str(cZ.c))
    lines.append("")
    
    G = cZ.G
    gen_lines = displayGenerators(G, DISPLAYDIM_MAX(), 'G')
    lines.extend(gen_lines)
    lines.append("")
    
    # Display constraint system
    if (cZ.A.size == 0 if isinstance(cZ.A, np.ndarray) else len(cZ.A) == 0) and \
       (cZ.b.size == 0 if isinstance(cZ.b, np.ndarray) else len(cZ.b) == 0):
        lines.append("Constraint system (Ax = b): no constraints.")
        lines.append("")
    else:
        lines.append("Constraint system (Ax = b):")
        lines.append("")
        
        A_lines = displayGenerators(cZ.A, DISPLAYDIM_MAX(), 'A')
        lines.extend(A_lines)
        lines.append("")
        
        lines.append("b: ")
        lines.append(str(cZ.b))
        lines.append("")
    
    return "\n".join(lines)


def display(cZ: 'ConZonotope', var_name: str = None) -> None:
    """
    Displays the properties of a conZonotope object (prints to stdout)
    
    Args:
        cZ: conZonotope object
        var_name: Optional variable name
    """
    print(display_(cZ, var_name), end='')

