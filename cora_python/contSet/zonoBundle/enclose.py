"""
enclose - encloses a zonotope bundle and its affine transformation (see
   Proposition 5 in [1])

Description:
    Computes the set
    { a x1 + (1 - a) * x2 | x1 \\in zB, x2 \\in S, a \\in [0,1] }
    where S = M*zB + Splus

Syntax:
    zB = enclose(zB, S)
    zB = enclose(zB, M, Splus)

Inputs:
    zB - zonoBundle object
    S - contSet object
    M - matrix for the linear transformation
    Splus - zonoBundle object added to the linear transformation

Outputs:
    zB - zonoBundle object that encloses given zonotope bundle and set

References:
    [1] M. Althoff. "Zonotope bundles for the efficient computation of
        reachable sets", 2011

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: zonotope/enclose

Authors:       Matthias Althoff (MATLAB)
               Automatic python translation: Florian Nüssel BA 2025
Written:       10-November-2010 (MATLAB)
Last update:   25-January-2016 (MATLAB)
Last revision: ---
"""

from typing import Any
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def enclose(zB, *args):
    from cora_python.contSet.zonoBundle import ZonoBundle
    from cora_python.contSet.zonotope import Zonotope

    # Parse input arguments
    if len(args) == 1:
        S = args[0]
    elif len(args) == 2:
        M, Splus = args
        S = (M @ zB) + Splus
    else:
        raise CORAerror('CORA:wrongInputInConstructor', 'enclose expects 2 or 3 input arguments.')

    # Work on a copy to match MATLAB value semantics
    zB_out = ZonoBundle(zB)

    if isinstance(S, ZonoBundle):
        if S.parallelSets != zB_out.parallelSets:
            raise CORAerror('CORA:wrongInput', 'zonoBundle sizes do not match for enclose.')
        for i in range(zB_out.parallelSets):
            zB_out.Z[i] = zB_out.Z[i].enclose(S.Z[i])
        return zB_out

    if isinstance(S, Zonotope):
        for i in range(zB_out.parallelSets):
            zB_out.Z[i] = zB_out.Z[i].enclose(S)
        return zB_out

    raise CORAerror('CORA:noops', zB_out, S)
