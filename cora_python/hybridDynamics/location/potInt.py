"""
potInt - determines which reachable sets potentially intersect with which
   guard sets

Syntax:
    [guards,setIndices,setType] = potInt(loc,R,finalLoc)

Inputs:
    loc - location object
    R - reachSet object storing reachable sets of location/reach
    finalLoc - final location of the automaton

Outputs:
    guards - guards that are potentially intersected
    setIndices - indices of the reachable sets that intersect the guards
    setType - which set has been determined to intersect the guard set
              ('time-interval' or 'time-point')

Authors:       Matthias Althoff, Niklas Kochdumper
Written:       08-May-2007 
Last update:   26-October-2007
               20-October-2010
               27-July-2016
               23-November-2017
               03-December-2019 (NK, use approximate intersection test)
               02-June-2023 (MW, immediate reach exit, special case)
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import Any, Tuple, List
from cora_python.contSet.polytope import Polytope
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.zonoBundle import ZonoBundle
from cora_python.contSet.conZonotope import ConZonotope


def potInt(loc: Any, R: Any, finalLoc: Any) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Determines which reachable sets potentially intersect with which guard sets
    
    Args:
        loc: location object
        R: reachSet object storing reachable sets of location/reach
        finalLoc: final location of the automaton
        
    Returns:
        guards: guards that are potentially intersected
        setIndices: indices of the reachable sets that intersect the guards
        setType: which set has been determined to intersect the guard set
                 ('time-interval' or 'time-point')
    """
    
    # check whether time-interval solution given (if the invariant is empty,
    # then the reachable set computation ends without computing any
    # time-interval reachable sets and returns only the start set)
    if hasattr(R, 'timeInterval') and R.timeInterval is not None and len(R.timeInterval.get('set', [])) > 0:
        Rset = R.timeInterval['set']
        setType = 'time-interval'
    else:
        Rset = R.timePoint['set']
        setType = 'time-point'
    
    # number of reachable sets
    nrSets = len(Rset)
    # number of transitions in the location = number of guard sets
    nrTrans = len(loc.transition)
    
    # preallocate variables for output arguments (upper bound of entries)
    guards = np.zeros(nrTrans * nrSets, dtype=int)
    setIndices = np.zeros(nrTrans * nrSets, dtype=int)
    
    # initialize number of intersections
    counter = 0
    
    # loop over all guards
    for i in range(nrTrans):
        
        # read out guard set and target location
        guardSet = loc.transition[i].guard
        target = loc.transition[i].target
        
        # only check if target location is not one of the terminal locations
        if not np.all(target == finalLoc):
            
            # loop over all reachable sets
            for j in range(nrSets):
                
                # check if reachable set intersects the guard set
                if guardSet.isIntersecting_(Rset[j], 'approx', 1e-8):
                    guards[counter] = i + 1  # MATLAB uses 1-based indexing
                    setIndices[counter] = j + 1  # MATLAB uses 1-based indexing
                    counter += 1
    
    # remove zeros from resulting lists
    guards = guards[:counter]
    setIndices = setIndices[:counter]
    
    # special case: when a timer is carried along as a state variable, it can
    # happen that the time step size is chosen such that a time-point solution
    # at t_k exactly intersects the guard set (time-triggered transition), but
    # the time-interval solutions at [t_k-1,t_k], [t_k,t_k+1] are used instead
    
    # we detect this case as follows
    # - only one guard set is intersected
    # - intersected guard set is a conHyperplane object
    # - only one/two subsequent time-interval solutions intersect
    # - time-point solution in the middle is contained in hyperplane
    # - time-interval solution(s) are not contained in hyperplane
    # since containment check is set-in-polytope, which uses support function
    # evaluations, we only do this for sets, where the support function is
    # somewhat quickly evaluated (zonotope, zonoBundle, conZonotope)
    
    # check correct number of intersections and object classes
    if (hasattr(R, 'timeInterval') and R.timeInterval is not None and 
        len(R.timeInterval.get('set', [])) > 0 and
        ((len(guards) == 1 and len(setIndices) == 1) or
         (len(guards) == 2 and guards[0] == guards[1] and
          len(setIndices) == 2 and setIndices[1] - setIndices[0] == 1)) and
        isinstance(loc.transition[guards[0] - 1].guard, Polytope) and
        loc.transition[guards[0] - 1].guard.representsa_('conHyperplane', 1e-12) and
        (isinstance(Rset[setIndices[0] - 1], Zonotope) or
         isinstance(Rset[setIndices[0] - 1], ZonoBundle) or
         isinstance(Rset[setIndices[0] - 1], ConZonotope))):
        
        # check containment of time-point solution
        if loc.transition[guards[0] - 1].guard.contains_(
                R.timePoint['set'][setIndices[0]], 'exact', 1e-10, 0, False, False):
            # ensure that time-interval solutions are not contained
            contained = False
            for idx in setIndices:
                if loc.transition[guards[0] - 1].guard.contains_(
                        Rset[idx - 1], 'exact', 1e-10, 0, False, False):
                    contained = True
                    break
            
            if not contained:
                # choose time-point instead of time-interval solution(s)
                guards = np.array([guards[0]])
                setIndices = np.array([setIndices[0] + 1])
                setType = 'time-point'
    
    return guards, setIndices, setType

