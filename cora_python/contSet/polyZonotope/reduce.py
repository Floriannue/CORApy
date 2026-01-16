"""
reduce - reduces the order of a polynomial zonotope

Syntax:
    pZ = reduce(pZ, option, order)

Inputs:
    pZ - polyZonotope object
    option - reduction algorithm (see zonotope/reduce)
    order - order of reduced polynomial zonotope

Outputs:
    pZ - reduced polynomial zonotope

Example: 
    pZ = polyZonotope([0;0],[2 0 1;0 1 1],[0.1,-0.4;0.2,0.3],[1 0 3;0 1 1]);
    pZred = reduce(pZ,'girard',2);

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: zonotope/reduce

Authors:       Niklas Kochdumper
Written:       23-March-2018 
Last update:   06-July-2021 (MW, add adaptive)
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import TYPE_CHECKING, Union, Optional

if TYPE_CHECKING:
    from cora_python.contSet.polyZonotope import PolyZonotope

from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def reduce(pZ: 'PolyZonotope', option: str, order: Union[int, float], *varargin) -> 'PolyZonotope':
    """
    Reduces the order of a polynomial zonotope
    
    Args:
        pZ: polyZonotope object
        option: reduction algorithm (see zonotope/reduce)
        order: order of reduced polynomial zonotope
        *varargin: additional arguments passed to zonotope reduce
        
    Returns:
        pZ: reduced polynomial zonotope
    """
    from cora_python.contSet.polyZonotope import PolyZonotope
    from cora_python.contSet.polyZonotope.private.priv_reduceAdaptive import priv_reduceAdaptive
    
    # adaptive order reduction
    if option == 'adaptive':
        # note: var 'order' is not an order here, it's diagpercent
        pZ = priv_reduceAdaptive(pZ, order)
        return pZ
    
    # Check for approxdep_ prefix options
    if 'approxdep_' in option:
        # remove independent generators
        pZ.GI = np.zeros((pZ.dim(), 0))
    
    # extract dimensions
    N = len(pZ.c)
    P = pZ.G.shape[1] if pZ.G.size > 0 else 0
    Q = pZ.GI.shape[1] if pZ.GI.size > 0 else 0
    
    # number of generators that stay unreduced (N generators are added again
    # after reduction)
    K = max(0, int(np.floor(N * order - N)))
    
    # check if it is necessary to reduce the order
    if P + Q > N * order and K >= 0:
        
        # concatenate all generators
        if P > 0 and Q > 0:
            G = np.hstack([pZ.G.copy(), pZ.GI.copy()])
        elif P > 0:
            G = pZ.G.copy()
        elif Q > 0:
            G = pZ.GI.copy()
        else:
            G = np.zeros((N, 0))
        
        # half the generator length for exponents that are all even
        # MATLAB: ind = ~any(mod(pZ.E,2),1);
        # MATLAB: G(:,ind) = 0.5 * G(:,ind);
        # Note: This only applies to dependent generators (first P columns of G)
        if pZ.E.size > 0 and P > 0:
            ind = ~np.any(pZ.E % 2, axis=0)
            G[:, :P][:, ind] = 0.5 * G[:, :P][:, ind]
        
        # calculate the length of the generator vectors with a special metric
        # MATLAB: len = sum(G.^2,1);
        len_vals = np.sum(G**2, axis=0)
        
        # determine the smallest generators (= generators that are removed)
        # MATLAB: [~,ind] = sort(len,'descend');
        # MATLAB: ind = ind(K+1:end);
        sorted_indices = np.argsort(len_vals)[::-1]  # descending order
        ind = sorted_indices[K:]  # K+1:end in 0-based indexing
        
        # split the indices into the ones for dependent and independent
        # generators
        # MATLAB: indDep = ind(ind <= P);
        indDep = ind[ind < P]  # 0-based: ind <= P becomes ind < P
        # MATLAB: indInd = ind(ind > P);
        # MATLAB: indInd = indInd - P * ones(size(indInd));
        indInd = ind[ind >= P] - P  # 0-based: ind > P becomes ind >= P
        
        # construct a zonotope from the generators that are removed
        if len(indDep) > 0:
            Grem = pZ.G[:, indDep]
            Erem = pZ.E[:, indDep] if pZ.E.size > 0 else np.zeros((0, len(indDep)), dtype=int)
        else:
            Grem = np.zeros((N, 0))
            Erem = np.zeros((0, 0), dtype=int)
        
        if len(indInd) > 0:
            GIrem = pZ.GI[:, indInd]
        else:
            GIrem = np.zeros((N, 0))
        
        pZtemp = PolyZonotope(np.zeros((N, 1)), Grem, GIrem, Erem)
        
        # zonotope over-approximation
        zono = pZtemp.zonotope()
        
        # reduce the constructed zonotope with the reduction techniques for
        # linear zonotopes
        # MATLAB: zonoRed = reduce(zono,option,1,varargin{:});
        from cora_python.contSet.zonotope.reduce import reduce as zonotope_reduce
        zonoRed_result = zonotope_reduce(zono, option, 1, *varargin)
        # Handle tuple return (some methods return tuple, others return single value)
        if isinstance(zonoRed_result, tuple):
            zonoRed = zonoRed_result[0]  # First element is the reduced zonotope
        else:
            zonoRed = zonoRed_result
        
        # remove the generators that got reduced from the generator matrices
        if len(indDep) > 0:
            # MATLAB: pZ.G(:,indDep) = [];
            keep_dep = np.setdiff1d(np.arange(P), indDep)
            if len(keep_dep) > 0:
                pZ.G = pZ.G[:, keep_dep]
                if pZ.E.size > 0:
                    pZ.E = pZ.E[:, keep_dep]
            else:
                pZ.G = np.zeros((N, 0))
                pZ.E = np.zeros((0, 0), dtype=int)
        
        if len(indInd) > 0:
            # MATLAB: pZ.GI(:,indInd) = [];
            keep_ind = np.setdiff1d(np.arange(Q), indInd)
            if len(keep_ind) > 0:
                pZ.GI = pZ.GI[:, keep_ind]
            else:
                pZ.GI = np.zeros((N, 0))
        
        # add the reduced generators as new independent generators 
        # MATLAB: pZ.c = pZ.c + center(zonoRed);
        pZ.c = pZ.c + zonoRed.center()
        # MATLAB: pZ.GI = [pZ.GI, generators(zonoRed)];
        zonoRed_G = zonoRed.generators()
        if zonoRed_G.size > 0:
            if pZ.GI.size > 0:
                pZ.GI = np.hstack([pZ.GI, zonoRed_G])
            else:
                pZ.GI = zonoRed_G
    
    if 'approxdep_' in option:
        # again remove rest generators
        pZ.GI = np.zeros((pZ.dim(), 0))
    
    # remove all exponent vector dimensions that have no entries
    # MATLAB: ind = sum(pZ.E,2)>0;
    if pZ.E.size > 0:
        ind = np.sum(pZ.E, axis=1) > 0
        pZ.E = pZ.E[ind, :]
        if pZ.id.size > 0:
            pZ.id = pZ.id[ind]
    else:
        pZ.E = np.zeros((0, 0), dtype=int)
        pZ.id = np.zeros((0, 1), dtype=int)
    
    return pZ
