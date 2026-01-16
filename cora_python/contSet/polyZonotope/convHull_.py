"""
convHull_ - Computes the convex hull of a polynomial zonotope and another
    set representation

Syntax:
    pZ = convHull_(pZ)
    pZ = convHull_(pZ,S)

Inputs:
    pZ - polyZonotope object
    S - contSet object

Outputs:
    S_out - convex hull

Example: 
    pZ = polyZonotope([0;0],[1 0;0 1],[],[1 3]);
    S_out = convHull(pZ);

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: contSet/convHull, zonotope/enclose

Authors:       Niklas Kochdumper
Written:       25-June-2018
Last update:   05-May-2020 (MW, standardized error message)
Last revision: 29-September-2024 (MW, integrate precedence)
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from cora_python.contSet.polyZonotope import PolyZonotope
    from cora_python.contSet.contSet.contSet import ContSet

from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.matlab.validate.check.equal_dim_check import equal_dim_check as equalDimCheck


def convHull_(pZ: 'PolyZonotope', S: 'ContSet' = None, *varargin) -> 'PolyZonotope':
    """
    Computes the convex hull of a polynomial zonotope and another set
    
    Args:
        pZ: polyZonotope object
        S: contSet object (optional)
        *varargin: additional arguments
        
    Returns:
        S_out: convex hull as polyZonotope
    """
    from cora_python.contSet.polyZonotope import PolyZonotope
    
    # parse input arguments
    if S is None:
        # MATLAB: S_out = linComb(pZ,pZ);
        S_out = pZ.linComb(pZ)
        return S_out
    
    # ensure that numeric is second input argument
    # MATLAB: [pZ,S] = reorderNumeric(pZ,S);
    # In Python, we handle reordering manually
    if isinstance(pZ, (int, float, np.ndarray)) and isinstance(S, PolyZonotope):
        temp = pZ
        pZ = S
        S = temp
    
    # check dimensions
    equalDimCheck(pZ, S)
    
    # call function with lower precedence
    # MATLAB: if isa(S,'contSet') && S.precedence < pZ.precedence
    if hasattr(S, 'precedence') and hasattr(pZ, 'precedence') and S.precedence < pZ.precedence:
        # MATLAB: S_out = convHull(S,pZ,varargin{:});
        return S.convHull_(pZ, *varargin)
    
    # convex hull with empty set
    # MATLAB: if representsa_(S,'emptySet',eps)
    if hasattr(S, 'representsa_') and S.representsa_('emptySet', np.finfo(float).eps):
        return pZ
    
    # convert to polynomial zonotope
    try:
        S = PolyZonotope(S)
    except Exception as ME:
        raise CORAerror('CORA:noops', pZ, S)
    
    # polyZonotope-polyZonotope case
    S_out = _aux_convHullMult(pZ, S)
    
    return S_out


def _aux_convHullMult(pZ1: 'PolyZonotope', pZ2: 'PolyZonotope') -> 'PolyZonotope':
    """
    compute the convex hull of two polynomial zonotopes
    """
    from cora_python.contSet.polyZonotope import PolyZonotope
    from cora_python.contSet.zonotope.enclose import enclose
    
    # remove independent generators
    # MATLAB: pZ1_ = polyZonotope(pZ1.c,pZ1.G,[],pZ1.E,pZ1.id);
    pZ1_ = PolyZonotope(pZ1.c, pZ1.G, np.zeros((pZ1.dim(), 0)), pZ1.E, pZ1.id)
    # MATLAB: pZ2_ = polyZonotope(S.c,S.G,[],S.E,S.id);
    pZ2_ = PolyZonotope(pZ2.c, pZ2.G, np.zeros((pZ2.dim(), 0)), pZ2.E, pZ2.id)
    
    # compute convex hull of dependent part using the linear combination
    # MATLAB: pZ = linComb(linComb(pZ1_,pZ1_),linComb(pZ2_,pZ2_));
    pZ1_lin = pZ1_.linComb(pZ1_)
    pZ2_lin = pZ2_.linComb(pZ2_)
    pZ = pZ1_lin.linComb(pZ2_lin)
    
    # compute convex hull of the independent part using the convex hull for
    # zonotopes
    # MATLAB: c0 = zeros(length(pZ1.c),1);
    c0 = np.zeros((len(pZ1.c), 1))
    # MATLAB: Z1 = zonotope(c0, pZ1.GI);
    from cora_python.contSet.zonotope import Zonotope
    Z1 = Zonotope(c0, pZ1.GI)
    # MATLAB: Z2 = zonotope(c0, S.GI);
    Z2 = Zonotope(c0, pZ2.GI)
    
    # MATLAB: Z = enclose(Z1,Z2);
    Z = enclose(Z1, Z2)
    # MATLAB: GI = Z.G;
    GI = Z.generators()
    
    # construct the resulting set
    pZ.GI = GI
    
    return pZ
