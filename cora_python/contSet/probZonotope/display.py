"""
display - Displays the properties of a probZonotope object (center,
    interval generators, probabilistic generators, covariance matrix) on
    the command window

Syntax:
    display(probZ)

Inputs:
    probZ - probabilistic zonotope object

Outputs:
    ---

Example:

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Matthias Althoff
Written:       03-August-2007 
Last update:   26-February-2008
               09-June-2020 (MW, update formatting of output)
Last revision: ---
                Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import TYPE_CHECKING
from cora_python.contSet.contSet.display import display_ as contSet_display
from cora_python.g.functions.verbose.display.displayGenerators import displayGenerators
from cora_python.contSet.contSet.center import center

if TYPE_CHECKING:
    from .probZonotope import ProbZonotope


def display_(probZ: 'ProbZonotope', var_name: str = None) -> str:
    """
    Displays the properties of a probZonotope object (internal function that returns string)
    
    Args:
        probZ: probZonotope object
        var_name: Optional variable name
        
    Returns:
        str: String representation
    """
    lines = []
    
    # Display input variable
    lines.append("")
    if var_name is None:
        var_name = 'probZ'
    lines.append(f"{var_name} =")
    lines.append("")
    
    # Display dimension
    contSet_str = contSet_display(probZ)
    lines.append(contSet_str)
    lines.append("")
    
    # Display center
    lines.append("center: ")
    center_val = center(probZ)
    lines.append(str(center_val))
    lines.append("")
    
    maxGens = 10
    
    # Display interval generators
    if hasattr(probZ, 'Z') and probZ.Z is not None and probZ.Z.size > 0:
        if probZ.Z.shape[1] > 1:
            interval_gens = probZ.Z[:, 1:]  # Skip first column (center)
        else:
            interval_gens = np.array([]).reshape(probZ.Z.shape[0], 0)
        gen_lines = displayGenerators(interval_gens, maxGens, 'interval generators')
        lines.extend(gen_lines)
        lines.append("")
    
    # Display probabilistic generators
    if hasattr(probZ, 'g') and probZ.g is not None and probZ.g.size > 0:
        gen_lines = displayGenerators(probZ.g, maxGens, 'probabilistic generators')
        lines.extend(gen_lines)
        lines.append("")
    
    # Display covariance matrix
    lines.append("covariance matrix: ")
    if hasattr(probZ, 'cov') and probZ.cov is not None:
        lines.append(str(probZ.cov))
    else:
        lines.append("[]")
    lines.append("")
    
    return "\n".join(lines)


def display(probZ: 'ProbZonotope', var_name: str = None) -> None:
    """
    Displays the properties of a probZonotope object (prints to stdout)
    
    Args:
        probZ: probZonotope object
        var_name: Optional variable name
    """
    print(display_(probZ, var_name), end='')

