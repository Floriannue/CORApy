"""
display - displays the polygon in the command window

Syntax:
    display(pgon)

Inputs:
    pgon - polygon

Outputs:
    -

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: -

Authors:       Tobias Ladner
Written:       09-October-2024
Last update:   ---
Last revision: ---
                Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import TYPE_CHECKING
from cora_python.contSet.contSet.representsa import representsa
from cora_python.g.functions.verbose.display.dispEmptySet import dispEmptySet
from cora_python.g.functions.verbose.display.dispRn import dispRn
from cora_python.contSet.contSet.display import display_ as contSet_display

if TYPE_CHECKING:
    from .polygon import Polygon


def display_(pgon: 'Polygon', var_name: str = None) -> str:
    """
    Displays the polygon (internal function that returns string)
    
    Args:
        pgon: polygon object
        var_name: Optional variable name
        
    Returns:
        str: String representation
    """
    lines = []
    
    # Special cases
    if representsa(pgon, 'emptySet'):
        if var_name:
            return dispEmptySet(pgon, var_name)
        return dispEmptySet(pgon)
    elif representsa(pgon, 'fullspace'):
        if var_name:
            return dispRn(pgon, var_name)
        return dispRn(pgon)
    
    # Display input variable
    lines.append("")
    if var_name is None:
        var_name = 'pgon'
    lines.append(f"{var_name} =")
    lines.append("")
    
    # Display dimension
    contSet_str = contSet_display(pgon)
    lines.append(contSet_str)
    
    # Display vertices
    from cora_python.contSet.polygon.polygon import Polygon
    if hasattr(pgon, 'vertices_'):
        vertices = pgon.vertices_()
    else:
        # Fallback if vertices_ not available
        vertices = np.array([])
    lines.append(f"- vertices: {vertices.shape[1] if vertices.size > 0 else 0}")
    lines.append(f"- number of regions: {pgon.nrOfRegions if hasattr(pgon, 'nrOfRegions') else 0}")
    lines.append(f"- number of holes: {pgon.nrOfHoles if hasattr(pgon, 'nrOfHoles') else 0}")
    lines.append("")
    
    return "\n".join(lines)


def display(pgon: 'Polygon', var_name: str = None) -> None:
    """
    Displays the polygon (prints to stdout)
    
    Args:
        pgon: polygon object
        var_name: Optional variable name
    """
    print(display_(pgon, var_name), end='')

