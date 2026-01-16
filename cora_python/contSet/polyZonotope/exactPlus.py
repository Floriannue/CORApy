"""
exactPlus - computes the addition of two sets while preserving the
    dependencies between the two sets

Syntax:
    pZ = exactPlus(pZ1,pZ2)

Inputs:
    pZ1 - polyZonotope object
    pZ2 - polyZonotope object

Outputs:
    pZ - polyZonotope object

Example: 
    pZ1 = polyZonotope([0;0],[2 1 2;0 2 2],[],[1 0 3;0 1 1]);
    pZ2 = [1 2;-1 1]*pZ1;
   
    pZ = pZ1 + pZ2;
    pZ_ = exactPlus(pZ1,pZ2);

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: mtimes, zonotope/plus

Authors:       Niklas Kochdumper
Written:       26-March-2018 
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cora_python.contSet.polyZonotope import PolyZonotope


def exactPlus(pZ1: 'PolyZonotope', pZ2: 'PolyZonotope') -> 'PolyZonotope':
    """
    Computes the addition of two sets while preserving the dependencies
    
    Args:
        pZ1: polyZonotope object
        pZ2: polyZonotope object
        
    Returns:
        pZ: polyZonotope object
    """
    from cora_python.contSet.polyZonotope import PolyZonotope
    from cora_python.g.functions.helper.sets.contSet.polyZonotope.mergeExpMatrix import mergeExpMatrix
    from cora_python.g.functions.helper.sets.contSet.polyZonotope.removeRedundantExponents import removeRedundantExponents
    
    # bring the exponent matrices to a common representation
    # MATLAB: [id,E1,E2] = mergeExpMatrix(pZ1.id,pZ2.id,pZ1.E,pZ2.E);
    id, E1, E2 = mergeExpMatrix(pZ1.id, pZ2.id, pZ1.E, pZ2.E)
    
    # add up all generators that belong to identical exponents
    # MATLAB: [Enew,Gnew] = removeRedundantExponents([E1,E2],[pZ1.G,pZ2.G]);
    E_combined = np.hstack([E1, E2]) if E1.size > 0 and E2.size > 0 else (E1 if E1.size > 0 else E2)
    G_combined = np.hstack([pZ1.G, pZ2.G]) if pZ1.G.size > 0 and pZ2.G.size > 0 else (pZ1.G if pZ1.G.size > 0 else pZ2.G)
    
    # Handle empty cases
    if E_combined.size == 0:
        E_combined = np.zeros((0, G_combined.shape[1] if G_combined.size > 0 else 0), dtype=int)
    if G_combined.size == 0:
        G_combined = np.zeros((pZ1.dim(), 0))
        E_combined = np.zeros((0, 0), dtype=int)
    
    Enew, Gnew = removeRedundantExponents(E_combined, G_combined)
    
    # assemble the properties of the resulting polynomial zonotope
    # MATLAB: pZ = pZ1;
    pZ = pZ1.copy()
    # MATLAB: pZ.G = Gnew;
    pZ.G = Gnew
    # MATLAB: pZ.E = Enew;
    pZ.E = Enew
    
    # MATLAB: pZ.c = pZ1.c + pZ2.c;
    pZ.c = pZ1.c + pZ2.c
    # MATLAB: pZ.GI = [pZ1.GI,pZ2.GI];
    if pZ1.GI.size > 0 and pZ2.GI.size > 0:
        pZ.GI = np.hstack([pZ1.GI, pZ2.GI])
    elif pZ1.GI.size > 0:
        pZ.GI = pZ1.GI
    elif pZ2.GI.size > 0:
        pZ.GI = pZ2.GI
    else:
        pZ.GI = np.zeros((pZ1.dim(), 0))
    # MATLAB: pZ.id = id;
    pZ.id = id
    
    return pZ
