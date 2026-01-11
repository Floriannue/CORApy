"""
display - Displays the vertices of a matrix polytope

Syntax:
    display(matP)

Inputs:
    matP - matPolytope object

Outputs:
    -

Example:
    matP = matPolytope()
    V = np.zeros((2, 2, 3))
    V[:,:,0] = np.array([[2, 3], [2, 1]])
    V[:,:,1] = np.array([[3, 4], [3, 2]])
    V[:,:,2] = np.array([[1, 1], [1, 0]])
    matP = matPolytope(V)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Matthias Althoff (MATLAB)
               Python translation by AI Assistant
Written:       21-June-2010 (MATLAB)
Last update:   03-April-2023 (MW, add empty case) (MATLAB)
               25-April-2024 (TL, harmonized display with contSet classes) (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .matPolytope import MatPolytope


def display_(matP: 'MatPolytope', var_name: str = None) -> str:
    """
    Displays a matPolytope object (internal function that returns string)
    
    Args:
        matP: matPolytope object
        var_name: Optional variable name
        
    Returns:
        str: Formatted string representation
    """
    lines = []
    
    # Check for empty object
    if matP.V.size == 0:
        if var_name:
            lines.append(f"\n{var_name} =")
        lines.append("  (empty matPolytope)")
        return "\n".join(lines)
    
    # display input variable
    if var_name:
        lines.append(f"\n{var_name} =")
    lines.append("")
    
    # display dimension, number of vertices
    lines.append("dimension: ")
    dims = matP.dim()
    if isinstance(dims, tuple):
        lines.append(str(dims))
    else:
        lines.append(str(dims))
    lines.append("nr of vertices: ")
    num_verts = matP.numverts()
    lines.append(str(num_verts))
    
    # display vertices
    lines.append("vertices: ")
    lines.append(str(matP.V))
    
    return "\n".join(lines)


def display(matP: 'MatPolytope', var_name: str = None) -> None:
    """
    Displays a matPolytope object (prints to stdout)
    
    Args:
        matP: matPolytope object
        var_name: Optional variable name
    """
    print(display_(matP, var_name), end='')

