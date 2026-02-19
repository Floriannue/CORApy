"""
cartProd_ - Cartesian product of a zonotope bundle and a set

Syntax:
    zB = cartProd_(zB, S)

Inputs:
    zB - zonoBundle object
    S - contSet object

Outputs:
    zB - zonoBundle object

Example:
    Z1 = zonotope([1;-1],[2 -1 3; 0 1 -1]);
    Z2 = Z1 + [1;0];
    zB1 = zonoBundle({Z1,Z2});
    Z1 = zonotope([0;-2],[1 -3 2 0 1; -2 3 -1 -1 0]);
    Z2 = Z1 + [-1;0];
    zB2 = zonoBundle({Z1,Z2});
    zB = cartProd(zB1,zB2);

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: contSet/cartProd, zonotope/cartProd_

Authors:       Niklas Kochdumper (MATLAB)
               Automatic python translation: Florian Nüssel BA 2025
Written:       13-June-2018 (MATLAB)
Last update:   24-September-2019 (MATLAB)
               05-May-2020 (MW, standardized error message, MATLAB)
Last revision: 27-March-2023 (MW, rename cartProd_, MATLAB)
"""

import numpy as np
from typing import Any

from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def cartProd_(zB, S: Any, *args):
    from cora_python.contSet.zonoBundle import ZonoBundle
    from cora_python.contSet.zonotope import Zonotope
    from cora_python.contSet.interval import Interval

    # first or second set is zonoBundle
    if isinstance(zB, ZonoBundle):
        if isinstance(S, ZonoBundle):
            list_out = []
            for i in range(zB.parallelSets):
                for j in range(S.parallelSets):
                    list_out.append(zB.Z[i].cartProd_(S.Z[j]))
            return ZonoBundle(list_out)

        if isinstance(S, (Zonotope, Interval)) or isinstance(S, np.ndarray):
            zB_out = ZonoBundle(zB)
            for i in range(zB_out.parallelSets):
                zB_out.Z[i] = zB_out.Z[i].cartProd_(S)
            return zB_out

        raise CORAerror('CORA:noops', zB, S)

    # zB is numeric
    if isinstance(zB, np.ndarray) and isinstance(S, ZonoBundle):
        zB_out = ZonoBundle(S)
        for i in range(zB_out.parallelSets):
            temp = Zonotope(zB)
            zB_out.Z[i] = temp.cartProd_(zB_out.Z[i])
        return zB_out

    raise CORAerror('CORA:noops', zB, S)
