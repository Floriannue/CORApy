"""
display - Displays the properties of a zonotope object (center and
    generator matrix) on the command window

Syntax:
    display(Z)

Inputs:
    Z - zonotope object

Outputs:
    ---

Example:
    Z = zonotope([1, 0], [[1, 2, -1], [0, -1, 1]])
    display(Z)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Matthias Althoff (MATLAB)
               Python translation by AI Assistant
Written:       14-September-2006 (MATLAB)
Last update:   09-June-2020 (MATLAB)
Python translation: 2025
"""

import numpy as np
from cora_python.contSet.contSet.representsa import representsa


def display(Z):
    """
    Displays the properties of a zonotope object
    
    Args:
        Z: zonotope object
        
    Returns:
        str: String representation of the zonotope
    """
    lines = []
    
    # Check for special cases using global representsa
    if representsa(Z, 'emptySet'):
        result = f"Empty zonotope in R^{Z.dim()}"
        return result
    if representsa(Z, 'fullspace'):
        result = f"Fullspace zonotope in R^{Z.dim()}"
        return result
    
    # Display basic information
    lines.append(f"Zonotope in R^{Z.dim()}")
    lines.append("")
    
    # Display center
    lines.append("c:")
    lines.append(str(Z.c))
    lines.append("")
    
    # Display generators (limit display for large matrices)
    if Z.G.size > 0:
        lines.append("G:")
        # Limit display to reasonable size (similar to MATLAB's DISPLAYDIM_MAX)
        max_display = 10
        if Z.G.shape[1] <= max_display:
            lines.append(str(Z.G))
        else:
            lines.append(f"{Z.G[:, :max_display]} ... ({Z.G.shape[1]} generators total)")
    else:
        lines.append("G: (no generators)")
    lines.append("")
    
    result = "\n".join(lines)
    return result 