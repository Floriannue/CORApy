"""
display - Displays the center and generators of a matrix zonotope

Syntax:
    display(matZ)

Inputs:
    matZ - matZonotope object

Outputs:
    ---

Example: 
    matZ = matZonotope()
    C = np.array([[0, 0], [0, 0]])
    G = np.zeros((2, 2, 2))
    G[:,:,0] = np.array([[1, 3], [-1, 2]])
    G[:,:,1] = np.array([[2, 0], [1, -1]])
    matZ = matZonotope(C, G)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Matthias Althoff (MATLAB)
               Python translation by AI Assistant
Written:       18-June-2010 (MATLAB)
Last update:   25-April-2024 (TL, harmonized display with contSet classes) (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .matZonotope import matZonotope

from cora_python.g.macros.DISPLAYDIM_MAX import DISPLAYDIM_MAX


def display_(matZ: 'matZonotope', var_name: str = None) -> str:
    """
    Displays a matZonotope object (internal function that returns string)
    
    Args:
        matZ: matZonotope object
        var_name: Optional variable name
        
    Returns:
        str: Formatted string representation
    """
    lines = []
    
    # Check for empty object
    if matZ.isempty():
        if var_name:
            lines.append(f"\n{var_name} =")
        lines.append("  (empty matZonotope)")
        return "\n".join(lines)
    
    # display input variable
    if var_name:
        lines.append(f"\n{var_name} =")
    lines.append("")
    
    # display dimension
    lines.append("matZonotope:")
    lines.append("dimension: ")
    dims = matZ.dim()
    lines.append(str(dims))
    
    # compute number of dims to display
    if isinstance(dims, tuple):
        dispdims = tuple(min(d, DISPLAYDIM_MAX()) for d in dims)
    else:
        dispdims = (min(dims, DISPLAYDIM_MAX()), min(dims, DISPLAYDIM_MAX()))
    num_gens = matZ.numgens()
    dispgens = min(num_gens, DISPLAYDIM_MAX())
    
    # display center
    lines.append("center: ")
    center_display = matZ.C[:dispdims[0], :dispdims[1]]
    lines.append(str(center_display))
    
    # display generators
    lines.append(f"generators: ({num_gens} generators)")
    if num_gens > 0:
        gens_display = matZ.G[:dispdims[0], :dispdims[1], :dispgens]
        lines.append(str(gens_display))
    
    addEmptyLine = False
    if any(dispdims[i] < dims[i] for i in range(len(dims))):
        lines.append(f"    Remainder of dimensions (>{DISPLAYDIM_MAX()}) not shown. Check workspace.")
        addEmptyLine = True
    if dispgens < num_gens:
        lines.append(f"    Remainder of generators (>{DISPLAYDIM_MAX()}) not shown. Check workspace.")
        addEmptyLine = True
    if addEmptyLine:
        lines.append("")
    
    if matZ.G.size == 0:
        lines.append("")
    
    return "\n".join(lines)


def display(matZ: 'matZonotope', var_name: str = None) -> None:
    """
    Displays a matZonotope object (prints to stdout)
    
    Args:
        matZ: matZonotope object
        var_name: Optional variable name
    """
    print(display_(matZ, var_name), end='')

