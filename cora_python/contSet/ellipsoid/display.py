"""
display - displays the properties of an ellipsoid object on the command window

Syntax:
    display(E)

Inputs:
    E - ellipsoid object

Outputs:
    (to command window)

Example: 
    E = Ellipsoid([[1, 0], [0, 1]])
    display(E)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Matthias Althoff (MATLAB)
               Python translation by AI Assistant
Written:       19-June-2015 (MATLAB)
Last update:   02-May-2020 (MW, add property validation, MATLAB)
               04-July-2022 (VG, class array, MATLAB)
               05-October-2024 (MW, remove class array, MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ellipsoid import Ellipsoid

from cora_python.g.functions.verbose.display.dispEmptySet import dispEmptySet


def display(E: 'Ellipsoid') -> str:
    """
    Displays the properties of an ellipsoid object
    
    Args:
        E: ellipsoid object
        
    Returns:
        str: formatted display string
    """
    # Check if ellipsoid represents empty set
    if E.representsa_('emptySet'):
        return dispEmptySet(E)
    
    # Check if ellipsoid represents fullspace
    if E.representsa_('fullspace'):
        # This shouldn't happen for ellipsoids, but handle it
        return f"ellipsoid:\n- dimension: {E.dim()}\n- represents fullspace"
    
    # Get dimensions
    n = E.dim()
    
    # Format output like MATLAB
    result = f"\nellipsoid =\n\n"
    
    # Display dimension (using contSet display)
    result += f"dimension: {n}\n\n"
    
    # Display center
    result += "q: \n"
    if E.q is not None and E.q.size > 0:
        # Format similar to MATLAB display
        q_flat = E.q.flatten()
        if n == 1:
            result += f"    {q_flat[0]:.4f}\n\n"
        else:
            for i in range(n):
                result += f"    {q_flat[i]:.4f}\n"
            result += "\n"
    else:
        result += "    []\n\n"
    
    # Display shape matrix
    result += "Q: \n"
    if E.Q is not None and E.Q.size > 0:
        # Format matrix display
        for i in range(E.Q.shape[0]):
            result += "    "
            for j in range(E.Q.shape[1]):
                result += f"{E.Q[i,j]:8.4f}  "
            result += "\n"
        result += "\n"
    else:
        result += "    []\n\n"
    
    # Display degeneracy
    result += "degenerate: \n"
    try:
        is_degenerate = not E.isFullDim()
        result += f"    {str(is_degenerate).lower()}\n"
    except:
        result += "    unknown\n"
    
    return result.rstrip()  # Remove trailing newline 