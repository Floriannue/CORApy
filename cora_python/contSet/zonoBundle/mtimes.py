"""
mtimes - Overloaded '*' operator for the multiplication of a matrix or an
   interval matrix with a zonotope bundle (see Prop. 1 in [1])

Syntax:
    zB = factor1 * factor2
    zB = mtimes(factor1, factor2)

Inputs:
    factor1 - zonoBundle object, numeric matrix or scalar
    factor2 - zonoBundle object, numeric scalar

Outputs:
    zB - zonoBundle object

References:
    [1] M. Althoff. "Zonotope bundles for the efficient computation of
        reachable sets", 2011

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: zonotope/mtimes

Authors:       Matthias Althoff, Mark Wetzlinger (MATLAB)
               Automatic python translation: Florian Nüssel BA 2025
Written:       09-November-2010 (MATLAB)
Last update:   ---
Last revision: 04-October-2024 (MW, remove InferiorClasses from zonotope, MATLAB)
"""

import numpy as np
from typing import Any

from cora_python.g.functions.matlab.validate.check.equal_dim_check import equal_dim_check as equalDimCheck
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def mtimes(factor1: Any, factor2: Any):
    """
    Matrix/scalar multiplication with a zonoBundle.
    """
    from cora_python.contSet.zonoBundle import ZonoBundle

    try:
        # matrix/scalar * zonoBundle
        if isinstance(factor1, (int, float, np.ndarray)) and isinstance(factor2, ZonoBundle):
            zB_in = factor2
            list_out = []
            for i in range(zB_in.parallelSets):
                list_out.append(factor1 @ zB_in.Z[i] if isinstance(factor1, np.ndarray) else zB_in.Z[i] * factor1)
            return ZonoBundle(list_out)

        # zonoBundle * scalar (note: zonoBundle * matrix not supported)
        if isinstance(factor2, (int, float)) and isinstance(factor1, ZonoBundle):
            zB_in = factor1
            list_out = []
            for i in range(zB_in.parallelSets):
                list_out.append(zB_in.Z[i] * factor2)
            return ZonoBundle(list_out)

    except Exception as exc:
        equalDimCheck(factor1, factor2)
        raise exc

    raise CORAerror('CORA:noops', factor1, factor2)
