"""
interval - converts a zonotope bundle to an interval according to
    Proposition 6 in [1]

Syntax:
    I = interval(zB)

Inputs:
    zB - zonoBundle object

Outputs:
    I - interval object

References:
    [1] M. Althoff. "Zonotope bundles for the efficient computation of 
        reachable sets", 2011

Other m-files required: interval(constructor)
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Matthias Althoff (MATLAB)
               Python translation by AI Assistant
Written:       10-November-2010 (MATLAB)
Last update:   25-July-2016 (intervalhull replaced by interval, MATLAB)
Last revision: ---
Python translation: 2025
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cora_python.contSet.zonoBundle.zonoBundle import ZonoBundle
    from cora_python.contSet.interval.interval import Interval


def interval(zB: 'ZonoBundle') -> 'Interval':
    """
    Convert a zonotope bundle to an interval
    
    Args:
        zB: zonoBundle object
        
    Returns:
        I: interval object
    """
    from cora_python.contSet.interval.interval import Interval
    
    if zB.parallelSets == 0:
        return Interval.empty(zB.dim())
    
    # enclose all zonotopes by an interval
    IHtmp = []
    for i in range(zB.parallelSets):
        IHtmp.append(zB.Z[i].interval())
    
    # intersect interval hulls
    I = IHtmp[0]
    for i in range(1, zB.parallelSets):
        I = I.and_(IHtmp[i], 'exact')
    
    return I 