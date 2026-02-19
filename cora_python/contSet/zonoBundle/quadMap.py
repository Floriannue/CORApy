"""
quadMap - computes {Q_{ijk}*x_j*x_k|x \\in Z}

Syntax:
    zB = quadMap(zB, Q)

Inputs:
    zB - zonoBundle object
    Q - quadratic coefficients as a list of matrices

Outputs:
    zB - zonoBundle object

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: zonotope/quadMap

Authors:       Niklas Kochdumper (MATLAB)
               Automatic python translation: Florian Nüssel BA 2025
Written:       13-June-2018 (MATLAB)
Last update:   ---
Last revision: ---
"""

from typing import Any, List
from cora_python.contSet.zonoBundle import ZonoBundle


def quadMap(zB: ZonoBundle, Q: List[Any]) -> ZonoBundle:
    zB_out = ZonoBundle(zB)
    for i in range(zB_out.parallelSets):
        zB_out.Z[i] = zB_out.Z[i].quadMap(Q)
    return zB_out
