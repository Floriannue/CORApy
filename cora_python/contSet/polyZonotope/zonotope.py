"""
zonotope - computes an enclosing zonotope of the polynomial zonotope

Syntax:
    Z = zonotope(pZ)

Inputs:
    pZ - polyZonotope object

Outputs:
    Z - zonotope object

Example: 
    pZ = polyZonotope([0;0],[2 0 1 1;0 2 1 2],[0;0],[1 0 3 1;0 1 0 2]);
    Z = zonotope(pZ);

    figure; hold on;
    plot(pZ,[1,2],'Filled','r');
    plot(Z,[1,2],'b');

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: interval, polytope

Authors:       Niklas Kochdumper (MATLAB)
               Automatic python translation: Florian Nüssel BA 2025
Written:       24-March-2018 (MATLAB)
Last update:   ---
Last revision: ---
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cora_python.contSet.polyZonotope.polyZonotope import PolyZonotope
    from cora_python.contSet.zonotope.zonotope import Zonotope


def zonotope(pZ: 'PolyZonotope') -> 'Zonotope':
    """
    Computes an enclosing zonotope of the polynomial zonotope
    
    Args:
        pZ: polyZonotope object
        
    Returns:
        Z: zonotope object
    """
    from cora_python.contSet.zonotope.zonotope import Zonotope
    
    if pZ.G.size > 0:
        # Determine dependent generators with exponents that are all even
        # MATLAB: isEvenColumn = all(mod(pZ.E, 2) == 0, 1)
        # Check if all exponents in each column are even
        isEvenColumn = np.all((pZ.E % 2) == 0, axis=0)
        Gquad = pZ.G[:, isEvenColumn]
        
        # Compute zonotope parameter
        # MATLAB: c = pZ.c + 0.5 * sum(Gquad,2)
        # Sum along axis 1 (columns) to get row sums
        c = pZ.c + 0.5 * np.sum(Gquad, axis=1, keepdims=True)
        
        # MATLAB: G = [pZ.G(:, ~isEvenColumn), 0.5*Gquad, pZ.GI]
        # Combine generators: non-even dependent, half of even dependent, and independent
        G_non_even = pZ.G[:, ~isEvenColumn]
        G_quad_half = 0.5 * Gquad
        
        # Combine all generators horizontally (MATLAB uses [A, B, C] which is horizontal concatenation)
        # Use hstack to combine, handling empty arrays properly
        generators_list = []
        if G_non_even.size > 0:
            generators_list.append(G_non_even)
        if G_quad_half.size > 0:
            generators_list.append(G_quad_half)
        if pZ.GI.size > 0:
            generators_list.append(pZ.GI)
        
        if generators_list:
            G = np.hstack(generators_list)
        else:
            # No generators - create empty generator matrix
            n = pZ.dim()
            G = np.array([]).reshape(n, 0)
        
        # Generate zonotope
        Z = Zonotope(c, G)
    else:
        # Only independent generators
        Z = Zonotope(pZ.c, pZ.GI)
    
    return Z

