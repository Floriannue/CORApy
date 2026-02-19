"""
plus - Overloaded '+' operator for the Minkowski addition of a zonotope
       bundle with a zonotope or with a vector (see Prop. 2 in [1])

Syntax:
    S_out = zB + S
    S_out = plus(zB, S)

Inputs:
    zB - zonoBundle object, numeric
    S - contSet object, numeric

Outputs:
    S_out - zonoBundle object after Minkowski addition

References:
    [1] M. Althoff. "Zonotope bundles for the efficient computation of
        reachable sets", 2011

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: mtimes

Authors:       Matthias Althoff (MATLAB)
               Automatic python translation: Florian Nüssel BA 2025
Written:       09-November-2010 (MATLAB)
Last update:   05-May-2020 (MW, standardized error message, MATLAB)
Last revision: ---
"""

import numpy as np
from typing import Any

from cora_python.g.functions.matlab.validate.check.equal_dim_check import equal_dim_check as equalDimCheck
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.contSet.interval.zonotope import zonotope as interval_to_zonotope


def plus(zB, S: Any):
    """
    Minkowski addition of a zonotope bundle with another set or a vector.
    """
    from cora_python.contSet.zonoBundle import ZonoBundle
    from cora_python.contSet.zonotope import Zonotope
    from cora_python.contSet.interval import Interval

    # ensure that numeric is second input argument
    if isinstance(zB, (int, float, np.ndarray)) and isinstance(S, ZonoBundle):
        zB, S = S, zB

    if not isinstance(zB, ZonoBundle):
        raise CORAerror('CORA:wrongInputInFunction', 'First input must be a ZonoBundle.')

    # work on a copy to match MATLAB value semantics
    S_out = ZonoBundle(zB)

    # call function with lower precedence
    if hasattr(S, 'precedence') and S.precedence < S_out.precedence:
        return S + S_out

    try:
        # over-approximate the set if the summand is a zonotope bundle or interval
        if isinstance(S, ZonoBundle) or isinstance(S, Interval):
            S_zono = interval_to_zonotope(S.interval() if isinstance(S, ZonoBundle) else S)
            for i in range(S_out.parallelSets):
                S_out.Z[i] = S_out.Z[i] + S_zono
            return S_out

        if isinstance(S, Zonotope):
            for i in range(S_out.parallelSets):
                S_out.Z[i] = S_out.Z[i] + S
            return S_out

        if isinstance(S, np.ndarray):
            if S.ndim == 1:
                S = S.reshape(-1, 1)
            if S.ndim == 2 and S.shape[1] == 1:
                for i in range(S_out.parallelSets):
                    S_out.Z[i] = S_out.Z[i] + S
                return S_out

    except Exception as exc:
        # check whether different dimension of ambient space
        equalDimCheck(S_out, S)

        # check for empty sets
        if S_out.representsa_('emptySet', np.finfo(float).eps) or (
            hasattr(S, 'representsa_') and S.representsa_('emptySet', np.finfo(float).eps)
        ):
            return ZonoBundle.empty(S_out.dim())

        raise exc

    raise CORAerror('CORA:noops', S_out, S)
