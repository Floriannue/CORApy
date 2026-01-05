"""
displayGenerators - Displays the center and generators of a zonotope

Syntax:
    displayGenerators(G, maxGens, varName)

Inputs:
    G - generator matrix
    maxGens - max number of displayed generators
    varName - name of generator matrix variable

Outputs:
    (to console)

Example: 
    Z = zonotope(rand(2,50));
    G = generators(Z);
    displayGenerators(G,10,'G');

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: contSet/display

Authors:       Mark Wetzlinger
Written:       09-June-2020
Last update:   ---
Last revision: ---
                Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from cora_python.g.macros import DISPLAYDIM_MAX


def displayGenerators(G: np.ndarray, maxGens: int, varName: str) -> list:
    """
    Displays the center and generators of a zonotope (returns list of strings)
    
    Args:
        G: generator matrix
        maxGens: max number of displayed generators
        varName: name of generator matrix variable
        
    Returns:
        list: List of strings representing the formatted output
    """
    lines = []
    
    # Display generators
    nrOfGens = G.shape[1] if G.size > 0 else 0
    if nrOfGens == 1:
        genStr = "generator"
    else:
        genStr = "generators"
    lines.append(f"{varName}: ({nrOfGens} {genStr})")
    
    if nrOfGens <= maxGens:
        lines.append(str(G))
    else:
        lines.append(str(G[:, :maxGens]))
        lines.append(f"    Remainder of generators (>{DISPLAYDIM_MAX()}) of {varName} not shown. Check workspace.")
        lines.append("")
    
    return lines

