"""
quadMap - computes the quadratic map of a polyZonotope

Syntax:
    pZ = quadMap(pZ,Q)
    pZ = quadMap(pZ,pZ2,Q)
    pZ = quadMap(pZ,pZ2,Q,dep)

Inputs:
    pZ, pZ2 - polyZonotope objects
    Q - quadratic coefficients as a list of matrices
    dep - keep dependencies (dep = True) or not (dep = False) 

Outputs:
    pZ - polyZonotope object

Example: 
    % quadratic multiplication
    pZ = polyZonotope([1;2],[1 -2 1; 2 3 1],[0;0],[1 0 2;0 1 1]);
    
    Q = [[1 2; -1 2], [-3 0; 1 1]];

    pZquad = quadMap(pZ,Q);

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: zonotope/quadMap, cubMap

Authors:       Niklas Kochdumper
Written:       23-March-2018
Last update:   21-April-2020 (remove zero-length independent generators)
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import TYPE_CHECKING, Union, List, Optional

if TYPE_CHECKING:
    from cora_python.contSet.polyZonotope import PolyZonotope

from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def quadMap(pZ: 'PolyZonotope', *varargin) -> 'PolyZonotope':
    """
    Computes the quadratic map of a polyZonotope
    
    Args:
        pZ: polyZonotope object
        *varargin: Either (Q,) for single case, (pZ2, Q) for mixed case, or (pZ2, Q, dep) for mixed with dependency flag
        
    Returns:
        pZ: polyZonotope object
    """
    from cora_python.contSet.polyZonotope import PolyZonotope
    
    nargin = len(varargin)
    
    if nargin < 1 or nargin > 3:
        raise CORAerror('CORA:wrongInput', 'Invalid number of input arguments for quadMap')
    
    if nargin == 1:
        Q = varargin[0]
        pZ = _aux_quadMapSingle(pZ, Q)
    elif nargin == 2:
        pZ2 = varargin[0]
        Q = varargin[1]
        pZ = _aux_quadMapMixed(pZ, pZ2, Q, False)
    elif nargin == 3:
        pZ2 = varargin[0]
        Q = varargin[1]
        dep = varargin[2]
        if not isinstance(dep, bool):
            raise CORAerror('CORA:wrongValue', 'fourth', 'has to be boolean.')
        pZ = _aux_quadMapMixed(pZ, pZ2, Q, dep)
    
    return pZ


def _aux_quadMapSingle(pZ: 'PolyZonotope', Q: List[np.ndarray]) -> 'PolyZonotope':
    """
    compute an over-approximation of the quadratic map 
    
    {x_i = x^T Q[i] x | x \in pZ} 
    """
    from cora_python.contSet.polyZonotope import PolyZonotope
    from cora_python.contSet.zonotope.quadMap import quadMap as zonotope_quadMap
    from cora_python.g.functions.helper.sets.contSet.polyZonotope.removeRedundantExponents import removeRedundantExponents
    
    # split into a zonotope Z that overapproximates the dependent generators,
    # and a zonotope Zrem that contains the independent generators
    # (Z + Zrem)' Q (Z + Zrem) = ...
    #       Z' Q Z  +  Zrem' Q Z  +  Z' Q Zrem  +  Zrem' Q Zrem 
    
    pZtemp = pZ.copy()
    pZtemp.GI = np.zeros((pZ.dim(), 0))
    
    Z = pZtemp.zonotope()
    Zrem = PolyZonotope(np.zeros((pZ.dim(), 1)), np.zeros((pZ.dim(), 0)), pZ.GI, 
                       np.zeros((0, 0), dtype=int), np.zeros((0, 1), dtype=int))
    Zrem = Zrem.zonotope()
    
    # return directly if pZ is a zonotope (rest matrix only)
    if pZ.G.size == 0:
        Zrem_ = PolyZonotope(pZ.c, np.zeros((pZ.dim(), 0)), pZ.GI,
                            np.zeros((0, 0), dtype=int), np.zeros((0, 1), dtype=int))
        Zrem_ = Zrem_.zonotope()
        Zr = zonotope_quadMap(Zrem_, Q)
        pZ.c = Zr.center()
        pZ.GI = Zr.generators()
        pZ.G = np.zeros((pZ.dim(), 0))
        pZ.E = np.zeros((0, 0), dtype=int)
        pZ.id = np.zeros((0, 1), dtype=int)
        return pZ
    
    # extend generator and exponent matrix by center
    Gext = np.hstack([pZ.c, pZ.G])
    Eext = np.hstack([np.zeros((pZ.E.shape[0], 1), dtype=int), pZ.E]) if pZ.E.size > 0 else np.zeros((0, 1), dtype=int)
    
    # initialize the resulting generator and exponent matrix 
    N = Gext.shape[1]
    dim_Q = len(Q)
    M = N * (N + 1) // 2
    
    Equad = np.zeros((pZ.E.shape[0] if pZ.E.size > 0 else 0, M), dtype=int)
    Gquad = np.zeros((dim_Q, M))
    
    # create the exponent matrix that corresponds to the quadratic map
    counter = 0
    
    for j in range(N):
        N_minus_j_plus_1 = N - j
        Equad[:, counter:counter+N_minus_j_plus_1] = Eext[:, j:N] + Eext[:, j:j+1] @ np.ones((1, N_minus_j_plus_1))
        counter = counter + N_minus_j_plus_1
    
    # loop over all dimensions
    for i in range(dim_Q):
        if Q[i] is not None and Q[i].size > 0:
            # quadratic evaluation
            quadMat = Gext.T @ Q[i] @ Gext
            
            # copy the result into the generator matrix
            counter = 0
            
            for j in range(N):
                N_minus_j_plus_1 = N - j
                # MATLAB: [quadMat(j,j), quadMat(j,j+1:N) + quadMat(j+1:N,j)'];
                diag_elem = quadMat[j, j]
                off_diag = quadMat[j, j+1:N] + quadMat[j+1:N, j].T
                Gquad[i, counter:counter+N_minus_j_plus_1] = np.concatenate([[diag_elem], off_diag.flatten()])
                counter = counter + N_minus_j_plus_1
    
    # add up all generators that belong to identical exponents
    Enew, Gnew = removeRedundantExponents(Equad, Gquad)
    
    # quadratic and mixed multiplication of remaining generators
    if pZ.GI.size > 0:
        # quadratic multiplication
        Ztemp1 = zonotope_quadMap(Zrem, Q)
        
        # mixed multiplications
        Ztemp2 = zonotope_quadMap(Z, Zrem, Q)
        Ztemp3 = zonotope_quadMap(Zrem, Z, Q)
        
        pZ.c = Ztemp1.center() + Ztemp2.center() + Ztemp3.center()
        GI = np.hstack([Ztemp1.generators(), Ztemp2.generators(), Ztemp3.generators()])
        
        # delete generators of length zero
        GI = GI[:, np.any(GI, axis=0)]
    else:
        pZ.c = np.zeros((len(Q), 1))
        GI = np.zeros((len(Q), 0))
    
    # assemble the properties of the resulting polynomial zonotope
    if Enew.size > 0 and np.sum(Enew[:, 0]) == 0:
        pZ.c = pZ.c + Gnew[:, 0:1]
        pZ.G = Gnew[:, 1:]
        pZ.E = Enew[:, 1:]
        pZ.GI = GI
    else:
        pZ.G = Gnew
        pZ.E = Enew
        pZ.GI = GI
    
    # Update id if needed
    if pZ.E.size > 0:
        if pZ.id.size == 0 or pZ.id.shape[0] != pZ.E.shape[0]:
            pZ.id = np.arange(1, pZ.E.shape[0] + 1).reshape(-1, 1)
    else:
        pZ.id = np.zeros((0, 1), dtype=int)
    
    return pZ


def _aux_quadMapMixed(pZ1: 'PolyZonotope', pZ2: 'PolyZonotope', Q: List[np.ndarray], dep: bool) -> 'PolyZonotope':
    """
    compute an over-approximation of the quadratic map 
    
    {x_i = x1^T Q[i] x2 | x1 \in pZ1, x2 \in pZ2} 
    
    of two polyZonotope objects.
    """
    from cora_python.contSet.polyZonotope import PolyZonotope
    from cora_python.contSet.zonotope.quadMap import quadMap as zonotope_quadMap
    from cora_python.g.functions.helper.sets.contSet.polyZonotope.removeRedundantExponents import removeRedundantExponents
    from cora_python.g.functions.helper.sets.contSet.polyZonotope.mergeExpMatrix import mergeExpMatrix
    
    # bring the exponent matrices to a common representation
    if not dep:
        if pZ1.id.size > 0:
            max_id1 = np.max(pZ1.id)
        else:
            max_id1 = 0
        if pZ2.id.size > 0:
            pZ2.id = pZ2.id + max_id1
    
    id, E1, E2 = mergeExpMatrix(pZ1.id, pZ2.id, pZ1.E, pZ2.E)
    id = np.arange(1, len(id) + 1).reshape(-1, 1)
    
    # split into a zonotope Z that overapproximates the dependent generators,
    # and a zonotope Zrem that contains the independent generators
    pZtemp = pZ1.copy()
    pZtemp.GI = np.zeros((pZ1.dim(), 0))
    Z1 = pZtemp.zonotope()
    Zrem1 = PolyZonotope(np.zeros((pZ1.dim(), 1)), np.zeros((pZ1.dim(), 0)), pZ1.GI,
                        np.zeros((0, 0), dtype=int), np.zeros((0, 1), dtype=int))
    Zrem1 = Zrem1.zonotope()
    
    pZtemp = pZ2.copy()
    pZtemp.GI = np.zeros((pZ2.dim(), 0))
    Z2 = pZtemp.zonotope()
    Zrem2 = PolyZonotope(np.zeros((pZ2.dim(), 1)), np.zeros((pZ2.dim(), 0)), pZ2.GI,
                        np.zeros((0, 0), dtype=int), np.zeros((0, 1), dtype=int))
    Zrem2 = Zrem2.zonotope()
    
    # construct extended generator and exponent matrix (extended by center)
    Gext1 = np.hstack([pZ1.c, pZ1.G])
    Eext1 = np.hstack([np.zeros((E1.shape[0], 1), dtype=int), E1]) if E1.size > 0 else np.zeros((0, 1), dtype=int)
    
    Gext2 = np.hstack([pZ2.c, pZ2.G])
    Eext2 = np.hstack([np.zeros((E2.shape[0], 1), dtype=int), E2]) if E2.size > 0 else np.zeros((0, 1), dtype=int)
    
    # initialize the resulting generator and exponent matrix 
    N1 = Gext1.shape[1]
    N2 = Gext2.shape[1]
    
    dim_Q = len(Q)
    M = N1 * N2
    
    Equad = np.zeros((E1.shape[0] if E1.size > 0 else 0, M), dtype=int)
    Gquad = np.zeros((dim_Q, M))
    
    # create the exponent matrix that corresponds to the quadratic map
    counter = 0
    
    for j in range(N2):
        Equad[:, counter:counter+N1] = Eext1 + Eext2[:, j:j+1] @ np.ones((1, N1))
        counter = counter + N1
    
    # loop over all dimensions
    for i in range(len(Q)):
        if Q[i] is not None and Q[i].size > 0:
            # quadratic evaluation
            quadMat = Gext1.T @ Q[i] @ Gext2
            Gquad[i, :] = quadMat.flatten(order='F')  # column-major order like MATLAB
    
    # add up all generators that belong to identical exponents
    Enew, Gnew = removeRedundantExponents(Equad, Gquad)
    
    # mixed multiplication of remaining generators
    GI = np.zeros((Zrem1.dim(), 0))
    Zquad_rest = zonotope_quadMap(Zrem1, Zrem2, Q)
    G_add = Zquad_rest.generators()
    c = Zquad_rest.center().copy()
    if G_add.size > 0:
        GI = np.hstack([GI, G_add])
    
    Zquad_mixed1 = zonotope_quadMap(Z1, Zrem2, Q)
    G_add = Zquad_mixed1.generators()
    c = c + Zquad_rest.center()  # MATLAB line 251: c = c + Zquad_rest.c (matches MATLAB exactly)
    if G_add.size > 0:
        GI = np.hstack([GI, G_add])
    
    Zquad_mixed2 = zonotope_quadMap(Zrem1, Z2, Q)
    G_add = Zquad_mixed2.generators()
    c = c + Zquad_rest.center()  # MATLAB line 256: c = c + Zquad_rest.c (matches MATLAB exactly)
    if G_add.size > 0:
        GI = np.hstack([GI, G_add])
    
    # remove zero-length independent generators
    GI = GI[:, np.any(GI, axis=0)]
    
    # assemble the properties of the resulting polynomial zonotope
    if Enew.size > 0 and np.sum(Enew[:, 0]) == 0:
        pZ = PolyZonotope(c + Gnew[:, 0:1], Gnew[:, 1:], GI, Enew[:, 1:], id)
    else:
        pZ = PolyZonotope(c, Gnew, GI, Enew, id)
    
    return pZ
