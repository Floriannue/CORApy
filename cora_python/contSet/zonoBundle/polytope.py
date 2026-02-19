"""
polytope - Converts a zonotope bundle to a polytope

Syntax:
    P = polytope(zB)

Inputs:
    zB - zonoBundle object
    method - (optional) approximation method ('exact', 'outer')

Outputs:
    P - polytope object

Example: 
    Z1 = zonotope(zeros(2,1),[1 0.5; -0.2 1]);
    Z2 = zonotope(ones(2,1),[1 -0.5; 0.2 1]);
    zB = zonoBundle({Z1,Z2});
    P = polytope(zB);

    figure; hold on;
    plot(zB,[1,2],'b');
    plot(P,[1,2],'r--');

Other m-files required: vertices, polytope
Subfunctions: none
MAT-files required: none

See also: interval, vertices

Authors:       Niklas Kochdumper
Written:       06-August-2018
Last update:   10-November-2022 (MW, unify various polytope functions)
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

from typing import Any


def polytope(zB: 'ZonoBundle', *varargin: Any) -> 'Polytope':
    """
    Converts a zonotope bundle to a polytope by intersecting polytopes of
    each parallel zonotope.
    """
    from cora_python.contSet.polytope import Polytope

    if zB.parallelSets == 0:
        return Polytope.empty(zB.dim())

    # compute over-approximative polytope for each zonotope
    Ptmp = []
    for i in range(zB.parallelSets):
        Ptmp.append(zB.Z[i].polytope(*varargin))

    # intersect all polytopes
    P = Ptmp[0]
    for i in range(1, zB.parallelSets):
        P = P.and_(Ptmp[i])

    # set properties
    if hasattr(P, '_bounded_val'):
        P._bounded_val = True

    return P

