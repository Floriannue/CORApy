"""
conZonotope - convert a zonotope bundle to a constrained zonotope
 
Syntax:
    cZ = conZonotope(zB)
 
Inputs:
    zB - zonoBundle object
 
Outputs:
    cZ - conZonotope object
 
Other m-files required: none
Subfunctions: none
MAT-files required: none
 
See also: none

Authors:       Niklas Kochdumper
Written:       23-May-2018
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cora_python.contSet.zonoBundle.zonoBundle import ZonoBundle


def conZonotope(zB: 'ZonoBundle'):
    """
    Convert a zonotope bundle to a constrained zonotope by intersecting
    the parallel zonotopes.
    """
    from cora_python.contSet.conZonotope.conZonotope import ConZonotope

    if zB.parallelSets == 0:
        return ConZonotope.empty(zB.dim())

    # initialization
    cZ = ConZonotope(zB.Z[0])

    # calculate the intersection of the parallel sets
    for i in range(1, zB.parallelSets):
        temp = ConZonotope(zB.Z[i])
        cZ = cZ.and_(temp, 'exact')

    return cZ

