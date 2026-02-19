"""
and_ - returns the intersection of a zonotope bundle and another set

Syntax:
    zB = and_(zB, S)

Inputs:
    zB - zonoBundle object
    S - contSet object

Outputs:
    zB - zonotope bundle after intersection

Example:
    Z1 = zonotope([0 1 2 0;0 1 0 2]);
    Z2 = zonotope([3 -0.5 3 0;-1 0.5 0 3]);
    zB = zonoBundle({Z1,Z2});
    P = polytope([1 1],2);

    res = zB & P;

    figure; hold on; xlim([-1,4]); ylim([-4,4]);
    plot(P,[1,2],'r','FaceAlpha',0.5);
    plot(res,[1,2],'FaceColor','g');
    plot(zB,[1,2],'b','LineWidth',3);

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: contSet/and, zonotope/and_

Authors:       Matthias Althoff (MATLAB)
               Automatic python translation: Florian Nüssel BA 2025
Written:       16-November-2010 (MATLAB)
Last update:   05-May-2020 (MW, standardized error message, MATLAB)
Last revision: 27-March-2023 (MW, rename and_, MATLAB)
               28-September-2024 (MW, integrate precedence, MATLAB)
"""

from typing import Any
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def and_(zB, S: Any, *args):
    """
    Intersection of a zonotope bundle with another set.
    """
    from cora_python.contSet.zonoBundle import ZonoBundle

    # Work on a copy to match MATLAB value semantics
    zB_out = ZonoBundle(zB)

    # call function with lower precedence
    if hasattr(S, 'precedence') and hasattr(zB_out, 'precedence') and S.precedence < zB_out.precedence:
        return S.and_(zB_out, *args)

    # in all cases: append to list of parallel sets
    cls_name = S.__class__.__name__ if hasattr(S, '__class__') else ''
    if cls_name == 'ZonoBundle':
        for i in range(S.parallelSets):
            zB_out.Z.append(S.Z[i])
        zB_out.parallelSets = zB_out.parallelSets + S.parallelSets
        return zB_out

    if cls_name == 'Zonotope':
        zB_out.Z.append(S)
        zB_out.parallelSets = zB_out.parallelSets + 1
        return zB_out

    if cls_name == 'Interval':
        from cora_python.contSet.interval.zonotope import zonotope
        zB_out.Z.append(zonotope(S))
        zB_out.parallelSets = zB_out.parallelSets + 1
        return zB_out

    # throw error for given arguments
    raise CORAerror('CORA:noops', zB_out, S)
