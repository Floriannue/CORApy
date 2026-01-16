"""
cartProd_ - Returns the Cartesian product a polynomial zonotope and
    another set representation

Syntax:
    pZ = cartProd_(pZ,S)

Inputs:
    pZ - polyZonotope object
    S - contSet object

Outputs:
    pZ - polyZonotope object

Example: 
    pZ = polyZonotope(2,[1 3 1],[],[1,2,3]);
    Z = zonotope([1,3]);

    pZcart = cartProd(pZ,Z);

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: contSet/cartProd, zonotope/cartProd_

Authors:       Niklas Kochdumper
Written:       25-June-2018
Last update:   05-May-2020 (MW, standardized error message)
Last revision: 27-March-2023 (MW, rename cartProd_)
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from cora_python.contSet.polyZonotope import PolyZonotope
    from cora_python.contSet.contSet import ContSet

from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def cartProd_(pZ: Union['PolyZonotope', 'ContSet'], S: 'ContSet', *varargin) -> 'PolyZonotope':
    """
    Returns the Cartesian product of a polynomial zonotope and another set
    
    Args:
        pZ: polyZonotope object (or other set that can be converted)
        S: contSet object
        *varargin: additional arguments
        
    Returns:
        pZ: polyZonotope object representing the Cartesian product
    """
    from cora_python.contSet.polyZonotope import PolyZonotope
    from cora_python.contSet.zonotope import Zonotope
    from cora_python.contSet.interval import Interval
    
    # convert other set representations to polyZonotopes (first set)
    if not isinstance(pZ, PolyZonotope):
        if isinstance(pZ, (Zonotope, Interval)):
            # MATLAB: pZ = zonotope(pZ);
            if isinstance(pZ, Interval):
                Z = pZ.zonotope()
            else:
                Z = pZ
            # MATLAB: pZ = polyZonotope(center(pZ),[],generators(pZ),[]);
            pZ = PolyZonotope(Z.center(), np.array([]).reshape(Z.dim(), 0), 
                             Z.generators(), np.array([], dtype=int).reshape(0, 0))
            
        elif hasattr(pZ, '__class__') and pZ.__class__.__name__ == 'ConPolyZono':
            # MATLAB: S = conPolyZono(S);
            # MATLAB: pZ = cartProd_(pZ,S,'exact');
            from cora_python.contSet.conPolyZono import ConPolyZono
            S = ConPolyZono(S)
            pZ = ConPolyZono(pZ)
            return pZ.cartProd_(S, 'exact', *varargin)
            
        elif hasattr(pZ, '__class__') and pZ.__class__.__name__ in ['Polytope', 'ZonoBundle', 'ConZonotope']:
            # MATLAB: pZ = polyZonotope(pZ);
            pZ = PolyZonotope(pZ)
            
        elif isinstance(pZ, (np.ndarray, list, tuple)) or (hasattr(pZ, '__array__') and not hasattr(pZ, '__class__')):
            # MATLAB: isnumeric(pZ)
            # MATLAB: pZ = polyZonotope(pZ,[],[],[]);
            pZ_array = np.asarray(pZ)
            if pZ_array.ndim == 1:
                pZ_array = pZ_array.reshape(-1, 1)
            pZ = PolyZonotope(pZ_array, np.array([]).reshape(pZ_array.shape[0], 0),
                             np.array([]).reshape(pZ_array.shape[0], 0),
                             np.array([], dtype=int).reshape(0, 0))
        else:
            # throw error for given arguments
            raise CORAerror('CORA:noops', pZ, S)
    
    # convert other set representations to polyZonotopes (second set)
    if not isinstance(S, PolyZonotope):
        if isinstance(S, (Zonotope, Interval)):
            # MATLAB: S = zonotope(S);
            if isinstance(S, Interval):
                Z = S.zonotope()
            else:
                Z = S
            # MATLAB: S = polyZonotope(center(S),[],generators(S));
            S = PolyZonotope(Z.center(), np.array([]).reshape(Z.dim(), 0),
                            Z.generators(), np.array([], dtype=int).reshape(0, 0))
            
        elif hasattr(S, '__class__') and S.__class__.__name__ == 'ConPolyZono':
            # MATLAB: pZ = conPolyZono(pZ);
            # MATLAB: pZ = cartProd_(pZ,S,'exact');
            from cora_python.contSet.conPolyZono import ConPolyZono
            pZ = ConPolyZono(pZ)
            S = ConPolyZono(S)
            return pZ.cartProd_(S, 'exact', *varargin)
            
        elif hasattr(S, '__class__') and S.__class__.__name__ in ['Polytope', 'ZonoBundle', 'ConZonotope']:
            # MATLAB: S = polyZonotope(S);
            S = PolyZonotope(S)
            
        elif isinstance(S, (np.ndarray, list, tuple)) or (hasattr(S, '__array__') and not hasattr(S, '__class__')):
            # MATLAB: isnumeric(S)
            # MATLAB: S = polyZonotope(S);
            S_array = np.asarray(S)
            if S_array.ndim == 1:
                S_array = S_array.reshape(-1, 1)
            S = PolyZonotope(S_array, np.array([]).reshape(S_array.shape[0], 0),
                            np.array([]).reshape(S_array.shape[0], 0),
                            np.array([], dtype=int).reshape(0, 0))
        else:
            # throw error for given arguments
            raise CORAerror('CORA:noops', pZ, S)
    
    # concatenate center vector
    # MATLAB: c = [pZ.c;S.c];
    c = np.vstack([pZ.c, S.c])
    
    # generator matrix, exponent matrix and identifier vector
    # MATLAB: G = blkdiag(pZ.G,S.G);
    # MATLAB: E = blkdiag(pZ.E,S.E);
    # MATLAB: id = [pZ.id; max([pZ.id;0])+S.id];
    
    # Handle empty matrices for blkdiag
    pZ_G = pZ.G if pZ.G.size > 0 else np.zeros((pZ.dim(), 0))
    S_G = S.G if S.G.size > 0 else np.zeros((S.dim(), 0))
    
    pZ_E = pZ.E if pZ.E.size > 0 else np.zeros((0, 0), dtype=int)
    S_E = S.E if S.E.size > 0 else np.zeros((0, 0), dtype=int)
    
    # blkdiag equivalent: block diagonal matrix
    # G = [pZ.G, zeros; zeros, S.G]
    n_pZ = pZ_G.shape[0]
    n_S = S_G.shape[0]
    p_pZ = pZ_G.shape[1]
    p_S = S_G.shape[1]
    
    G = np.block([
        [pZ_G, np.zeros((n_pZ, p_S))],
        [np.zeros((n_S, p_pZ)), S_G]
    ])
    
    # E = [pZ.E, zeros; zeros, S.E]
    e_pZ = pZ_E.shape[0]
    e_S = S_E.shape[0]
    c_pZ = pZ_E.shape[1]
    c_S = S_E.shape[1]
    
    E = np.block([
        [pZ_E, np.zeros((e_pZ, c_S), dtype=int)],
        [np.zeros((e_S, c_pZ), dtype=int), S_E]
    ])
    
    # id = [pZ.id; max([pZ.id;0])+S.id];
    if pZ.id.size > 0:
        max_pZ_id = np.max(pZ.id)
    else:
        max_pZ_id = 0
    
    if S.id.size > 0:
        S_id_shifted = S.id + max_pZ_id
    else:
        S_id_shifted = np.array([]).reshape(0, 1)
    
    if pZ.id.size > 0 and S_id_shifted.size > 0:
        id = np.vstack([pZ.id, S_id_shifted])
    elif pZ.id.size > 0:
        id = pZ.id
    elif S_id_shifted.size > 0:
        id = S_id_shifted
    else:
        id = np.array([]).reshape(0, 1)
    
    # matrix of independent generators
    # MATLAB: GI = blkdiag(pZ.GI,S.GI);
    pZ_GI = pZ.GI if pZ.GI.size > 0 else np.zeros((pZ.dim(), 0))
    S_GI = S.GI if S.GI.size > 0 else np.zeros((S.dim(), 0))
    
    gi_pZ = pZ_GI.shape[1]
    gi_S = S_GI.shape[1]
    
    GI = np.block([
        [pZ_GI, np.zeros((n_pZ, gi_S))],
        [np.zeros((n_S, gi_pZ)), S_GI]
    ])
    
    # generate new polyZonotope
    # MATLAB: pZ = polyZonotope(c,G,GI,E,id);
    pZ = PolyZonotope(c, G, GI, E, id)
    
    return pZ
