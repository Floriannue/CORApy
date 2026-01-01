"""
potOut - determines the reachable sets after intersection with the
   invariant and obtains the fraction of the reachable set that must have
   transitioned; the resulting reachable sets are all converted to
   polytopes

Syntax:
    R = potOut(loc,R,minInd,maxInd)

Inputs:
    loc - location object
    R - reachSet object storing the reachable set
    minInd - vector containing the indices of the set which first
             intersected the guard set for each guard set 
    maxInd - vector containing the indices of the set which last
             intersected the guard set for each guard set

Outputs:
    R - reachSet object

Authors:       Matthias Althoff, Niklas Kochdumper
Written:       11-May-2007 
Last update:   18-September-2007
               21-October-2010
               30-July-2016
               17-May-2018 (NK, only change sets that intersect guards)
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import Any
from cora_python.contSet.polytope import Polytope
from cora_python.g.classes.reachSet.reachSet import ReachSet


def potOut(loc: Any, R: Any, minInd: np.ndarray, maxInd: np.ndarray) -> ReachSet:
    """
    Determines the reachable sets after intersection with the invariant
    
    Args:
        loc: location object
        R: reachSet object storing the reachable set
        minInd: vector containing the indices of the set which first
                intersected the guard set for each guard set 
        maxInd: vector containing the indices of the set which last
                intersected the guard set for each guard set
        
    Returns:
        R: reachSet object
    """
    
    # read out time-point and time-interval reachable sets
    timeInt = R.timeInterval
    timePoint = R.timePoint
    
    # determine all sets that intersected the guard sets -> sets that are
    # partially located outside the invariant
    minInd = np.maximum(minInd, np.ones_like(minInd))
    ind = []
    for i in range(len(minInd)):
        temp = list(range(int(minInd[i]) - 1, int(maxInd[i])))  # Convert to 0-based indexing
        ind.extend(temp)
    # remove redundancies
    ind = list(set(ind))
    ind.sort()
    
    # loop over all sets that intersect the guard sets
    for i in range(len(ind)):
        iSet = ind[i]
        
        # overapproximate reachable set by a halfspace representation
        if iSet < len(timeInt.get('set', [])):
            timeInt['set'][iSet] = Polytope(timeInt['set'][iSet])
        if iSet < len(timePoint.get('set', [])):
            timePoint['set'][iSet] = Polytope(timePoint['set'][iSet])
        
        # intersect with invariant set
        if iSet < len(timeInt.get('set', [])):
            timeInt['set'][iSet] = loc.invariant.and_(timeInt['set'][iSet], 'exact')
        if iSet < len(timePoint.get('set', [])):
            timePoint['set'][iSet] = loc.invariant.and_(timePoint['set'][iSet], 'exact')
    
    # remove last set if it is located outside the invariant
    if (len(timeInt.get('set', [])) > 0 and
        not loc.invariant.isIntersecting_(timeInt['set'][-1], 'exact', 1e-8)):
        timeInt['set'] = timeInt['set'][:-1]
        timeInt['time'] = timeInt['time'][:-1]
        timePoint['set'] = timePoint['set'][:-1]
        timePoint['time'] = timePoint['time'][:-1]
        # field 'error' currently only supported in linearSys analysis
        if 'error' in timeInt:
            timeInt['error'] = timeInt['error'][:-1]
        if 'error' in timePoint:
            timePoint['error'] = timePoint['error'][:-1]
    
    # construct modified reachSet object
    R = ReachSet(timePoint, timeInt, R.parent)
    
    return R

