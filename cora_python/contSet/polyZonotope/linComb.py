"""
linComb - computes the linear combination of two polynomial zonotopes

Syntax:
    pZ = linComb(pZ1,S)

Inputs:
    pZ1 - polyZonotope object
    S - contSet object

Outputs:
    pZ - polyZonotope enclosing pZ1 and pZ2

Example: 
    pZ1 = polyZonotope([-2;-2],[2 0 1;0 2 1],[],[1 0 3;0 1 1]);
    pZ2 = polyZonotope([3;3],[1 -2 1; 2 3 1],[],[1 0 2;0 1 1]);
   
    pZ = linComb(pZ1,pZ2);

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: zonotope/enclose

Authors:       Niklas Kochdumper
Written:       25-June-2018
Last update:   05-May-2020 (MW, standardized error message)
               06-March-2023 (TL, optimizations using logical indexing)
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from cora_python.contSet.polyZonotope import PolyZonotope
    from cora_python.contSet.contSet.contSet import ContSet

from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.helper.sets.contSet.polyZonotope.removeRedundantExponents import removeRedundantExponents


def linComb(pZ1: Union['PolyZonotope', 'ContSet'], S: 'ContSet') -> 'PolyZonotope':
    """
    Computes the linear combination of two polynomial zonotopes
    
    Args:
        pZ1: polyZonotope object or contSet object
        S: contSet object
        
    Returns:
        pZ: polyZonotope enclosing the linear combination
    """
    from cora_python.contSet.polyZonotope import PolyZonotope
    from cora_python.contSet.zonotope.enclose import enclose
    
    # determine polyZonotope object
    # MATLAB: if ~isa(pZ1,'polyZonotope')
    if not isinstance(pZ1, PolyZonotope):
        temp = pZ1
        pZ1 = S
        S = temp
    
    # convert other set representations to polynomial zonotopes
    # MATLAB: if ~isa(S,'polyZonotope')
    if not isinstance(S, PolyZonotope):
        from cora_python.contSet.zonotope import Zonotope
        from cora_python.contSet.interval import Interval
        
        if isinstance(S, (Zonotope, Interval)) or \
           (hasattr(S, '__class__') and S.__class__.__name__ in ['Polytope', 'ZonoBundle', 'ConZonotope']):
            # MATLAB: S = polyZonotope(S);
            S = PolyZonotope(S)
        elif hasattr(S, '__class__') and S.__class__.__name__ == 'ConPolyZono':
            # MATLAB: pZ = linComb(S,pZ1);
            return S.linComb(pZ1)
        elif isinstance(S, (int, float, np.ndarray)):
            # MATLAB: S = polyZonotope(S,[],[],[]);
            S_array = np.asarray(S)
            if S_array.ndim == 1:
                S_array = S_array.reshape(-1, 1)
            S = PolyZonotope(S_array, np.zeros((S_array.shape[0], 0)), 
                           np.zeros((S_array.shape[0], 0)),
                           np.zeros((0, 0), dtype=int))
        else:
            # throw error for given arguments
            raise CORAerror('CORA:noops', pZ1, S)
    
    # extend generator and exponent matrix by center vector
    # MATLAB: G1 = [pZ1.c, pZ1.G];
    G1 = np.hstack([pZ1.c, pZ1.G]) if pZ1.G.size > 0 else pZ1.c
    # MATLAB: G2 = [S.c, S.G];
    G2 = np.hstack([S.c, S.G]) if S.G.size > 0 else S.c
    
    # MATLAB: E1 = [zeros(size(pZ1.E,1),1),pZ1.E];
    if pZ1.E.size > 0:
        E1 = np.hstack([np.zeros((pZ1.E.shape[0], 1), dtype=int), pZ1.E])
    else:
        E1 = np.zeros((0, 1), dtype=int)
    # MATLAB: E2 = [zeros(size(S.E,1),1),S.E];
    if S.E.size > 0:
        E2 = np.hstack([np.zeros((S.E.shape[0], 1), dtype=int), S.E])
    else:
        E2 = np.zeros((0, 1), dtype=int)
    
    # compute linear comb. of the dependent generators according to the
    # equation comb = (0.5 + 0.5 a)*pZ1 + (0.5 - 0.5 a)*pZ2, a \in [-1,1]
    # MATLAB: G = 0.5 * [G1, G1, G2, -G2];
    G = 0.5 * np.hstack([G1, G1, G2, -G2])
    
    h1 = E1.shape[1]
    h2 = E2.shape[1]
    
    # MATLAB: zero1 = zeros(size(E1,1),size(E2,2));
    zero1 = np.zeros((E1.shape[0], E2.shape[1]), dtype=int)
    # MATLAB: zero2 = zeros(size(E2,1),size(E1,2));
    zero2 = np.zeros((E2.shape[0], E1.shape[1]), dtype=int)
    
    # MATLAB: E = [E1, E1, zero1, zero1; ...
    #              zero2, zero2, E2, E2; ...
    #              zeros(1,h1), ones(1,h1), zeros(1,h2), ones(1,h2)];
    E = np.block([
        [E1, E1, zero1, zero1],
        [zero2, zero2, E2, E2],
        [np.zeros((1, h1), dtype=int), np.ones((1, h1), dtype=int), 
         np.zeros((1, h2), dtype=int), np.ones((1, h2), dtype=int)]
    ])
    
    # compute convex hull of the independent generators by using the
    # enclose function for linear zonotopes
    # MATLAB: temp = zeros(length(pZ1.c),1);
    temp = np.zeros((len(pZ1.c), 1))
    # MATLAB: Z1 = zonotope([temp, pZ1.GI]);
    from cora_python.contSet.zonotope import Zonotope
    Z1 = Zonotope(np.hstack([temp, pZ1.GI]) if pZ1.GI.size > 0 else temp)
    # MATLAB: Z2 = zonotope([temp, S.GI]);
    Z2 = Zonotope(np.hstack([temp, S.GI]) if S.GI.size > 0 else temp)
    
    # MATLAB: Z = enclose(Z1,Z2);
    Z = enclose(Z1, Z2)
    # MATLAB: GI = Z.G;
    GI = Z.generators()
    
    # add up all generators that belong to identical exponents
    # MATLAB: [Enew,Gnew] = removeRedundantExponents(E,G);
    Enew, Gnew = removeRedundantExponents(E, G)
    
    # extract the center vector
    # MATLAB: ind = sum(Enew,1) == 0;
    ind = np.sum(Enew, axis=0) == 0
    
    # MATLAB: c = sum(Gnew(:,ind),2);
    c = np.sum(Gnew[:, ind], axis=1, keepdims=True) if np.any(ind) else np.zeros((Gnew.shape[0], 1))
    # MATLAB: Gnew(:,ind) = [];
    Gnew = Gnew[:, ~ind] if np.any(ind) else Gnew
    # MATLAB: Enew(:,ind) = [];
    Enew = Enew[:, ~ind] if np.any(ind) else Enew
    
    # construct resulting polynomial zonotope object
    # MATLAB: pZ = polyZonotope(c,Gnew,GI,Enew);
    pZ = PolyZonotope(c, Gnew, GI, Enew)
    # MATLAB: pZ.id = (1:size(Enew,1))';
    if Enew.size > 0:
        pZ.id = np.arange(1, Enew.shape[0] + 1).reshape(-1, 1)
    else:
        pZ.id = np.zeros((0, 1), dtype=int)
    
    return pZ
