"""
guardIntersect - computes an enclosure of the intersection between the
   reachable set and the guard sets

Syntax:
    [Rguard,actGuards,minInd,maxInd] = 
       guardIntersect(loc,guards,setInd,setType,Rcont,params,options)

Inputs:
    loc - location object
    guards - list containing the guard sets that have been hit
    setInd - list containing the indices of intersecting sets
    setType - which set has been determined to intersect the guard set
              ('time-interval' or 'time-point')
    Rcont - reachSet object storing the reachable set
    params - model parameters
    options - struct with settings for reachability analysis

Outputs:
    Rguard - list containing the intersection with the guards
    actGuards - list with indices of the active guards
    minInd - minimum index of set intersecting guard 
    maxInd - maximum index of set intersecting guard

References: 
  [1] M. Althoff et al. "Computing Reachable Sets of Hybrid Systems Using 
      a Combination of Zonotopes and Polytopes", 2009
  [2] A. Girard et al. "Zonotope/Hyperplane Intersection for Hybrid 
      Systems Reachablity Analysis"
  [3] M. Althoff et al. "Avoiding Geometic Intersection Operations in 
      Reachability Analysis of Hybrid Systems"
  [4] S. Bak et al. "Time-Triggered Conversion of Guards for Reachability
      Analysis of Hybrid Automata"
  [5] N. Kochdumper et al. "Reachability Analysis for Hybrid Systems with 
      Nonlinear Guard Sets", HSCC 2020
  [6] M. Althoff et al. "Zonotope bundles for the efficient computation 
      of reachable sets", 2011

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: location/reach

Authors:       Matthias Althoff, Niklas Kochdumper
Written:       08-May-2007 
Last update:   21-September-2007
               30-July-2016
               23-November-2017
               20-April-2018 (intersect guard sets with invariant)
               23-December-2019 (restructured the code)
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import Any, List, Tuple, Optional
from cora_python.contSet.polytope import Polytope
from cora_python.contSet.levelSet import LevelSet
from cora_python.contSet.polyZonotope import PolyZonotope
from cora_python.contSet.zonotope import Zonotope
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
# Methods are attached to Location class in location/__init__.py
# Use loc.checkFlow(), loc.calcBasis() instead of standalone imports


def guardIntersect(loc: Any, guards: List[int], setInd: List[int], 
                   setType: str, Rcont: Any, params: dict, 
                   options: dict) -> Tuple[List[Any], List[int], List[int], List[int]]:
    """
    Compute an enclosure of the intersection between the reachable set and the guard sets
    
    Args:
        loc: location object
        guards: list containing the guard indices that have been hit
        setInd: list containing the indices of intersecting sets
        setType: which set has been determined to intersect the guard set
                 ('time-interval' or 'time-point')
        Rcont: reachSet object storing the reachable set
        params: model parameters
        options: struct with settings for reachability analysis (must contain 'guardIntersect')
        
    Returns:
        Rguard: list containing the intersection with the guards
        actGuards: list with indices of the active guards
        minInd: minimum index of set intersecting guard 
        maxInd: maximum index of set intersecting guard
    """
    
    # check if there exist guard intersections
    relIndex = np.unique(setInd)
    
    if len(relIndex) == 0:
        return [], [], [], []
    
    # extract the guard sets that got hit
    guardInd = np.unique(guards)
    Pguard = [None] * len(loc.transition)
    
    for i in range(len(guardInd)):
        idx = guardInd[i]
        Pguard[idx] = loc.transition[idx].guard
    
    # group the reachable sets which intersect guards
    Rtp = Rcont.timePoint.set
    if setType == 'time-interval':
        minInd, maxInd, P, actGuards = _aux_groupSets(Rcont.timeInterval.set, guards, setInd)
    elif setType == 'time-point':
        minInd, maxInd, P, actGuards = _aux_groupSets(Rtp, guards, setInd)
    else:
        raise CORAerror('CORA:wrongInput', 'setType', 
                       "setType must be 'time-interval' or 'time-point'")
    
    # loop over all guard intersections
    Rguard = [None] * len(minInd)
    
    for i in range(len(minInd)):
        
        # get current guard set
        guard = Pguard[actGuards[i]]
        
        # remove all intersections where the flow does not point in the
        # direction of the guard set (only check if there is no instant
        # transition, i.e., there is a time-interval reachable set)
        # MATLAB: try [res,P{i}] = checkFlow(loc,guard,P{i},params); ... end
        # Note: MATLAB try-end without catch will propagate errors, but the try
        # block structure suggests checkFlow may not always be applicable.
        # In Python, we match MATLAB behavior: if checkFlow throws, let it propagate.
        if setType == 'time-interval' and (isinstance(guard, LevelSet) or 
                (isinstance(guard, Polytope) and guard.representsa_('conHyperplane', 1e-12))):
            res, P[i] = loc.checkFlow(guard, P[i], params)
            if not res:
                continue
        
        if isinstance(guard, LevelSet):
            # if current guard set is a level set, only level set method applies
            # MATLAB: Rguard{i} = guardIntersect_levelSet(loc,P{i},guard);
            Rguard[i] = loc.guardIntersect_levelSet(P[i], guard)
            continue
        
        # selected method for the calculation of the intersection
        method = options.get('guardIntersect', 'zonoGirard')
        
        if method == 'polytope':
            # compute intersection with the method in [1]
            # MATLAB: Rguard{i} = guardIntersect_polytope(loc,P{i},guard,options);
            Rguard[i] = loc.guardIntersect_polytope(P[i], guard, options)
            
        elif method == 'conZonotope':
            # compute intersection using constrained zonotopes
            # calculate orthogonal basis with the methods in Sec. V.A in [6]
            # MATLAB: B = calcBasis(loc,P{i},guard,options,params);
            B = loc.calcBasis(P[i], guard, options, params)
            # MATLAB: Rguard{i} = guardIntersect_conZonotope(loc,P{i},guard,B,options);
            Rguard[i] = loc.guardIntersect_conZonotope(P[i], guard, B, options)
            
        elif method == 'zonoGirard':
            # compute intersection with the method in [2]
            # calc. orthogonal basis with the methods described in Sec. V.A in [6]
            # MATLAB: B = calcBasis(loc,P{i},guard,options,params);
            B = loc.calcBasis(P[i], guard, options, params)
            # MATLAB: Rguard{i} = guardIntersect_zonoGirard(loc,P{i},guard,B);
            Rguard[i] = loc.guardIntersect_zonoGirard(P[i], guard, B)
            
        elif method == 'hyperplaneMap':
            # compute intersection with the method in [3]
            R0 = _aux_getInitialSet(Rtp, minInd[i])
            # MATLAB: Rguard{i} = guardIntersect_hyperplaneMap(loc,guard,R0,params,options);
            Rguard[i] = loc.guardIntersect_hyperplaneMap(guard, R0, params, options)
            
        elif method == 'pancake':
            # compute intersection with the method in [4]
            R0 = _aux_getInitialSet(Rtp, minInd[i])
            # MATLAB: Rguard{i} = guardIntersect_pancake(loc,R0,guard,actGuards(i),params,options);
            Rguard[i] = loc.guardIntersect_pancake(R0, guard, actGuards[i], params, options)
            
        elif method == 'nondetGuard':
            # compute intersection with method for nondeterministic guards
            # calc. orthogonal basis with the methods described in Sec. V.A in [6]
            # MATLAB: B = calcBasis(loc,P{i},guard,options,params);
            B = loc.calcBasis(P[i], guard, options, params)
            # MATLAB: Rguard{i} = guardIntersect_nondetGuard(loc,P{i},guard,B);
            Rguard[i] = loc.guardIntersect_nondetGuard(P[i], guard, B)
            
        elif method == 'levelSet':
            # compute intersection with the method in [5]
            # MATLAB: if isa(guard,'polytope') && representsa_(guard,'conHyperplane',1e-12); guard = levelSet(guard); end
            if isinstance(guard, Polytope) and guard.representsa_('conHyperplane', 1e-12):
                from cora_python.contSet.levelSet import LevelSet
                guard = LevelSet(guard)
            # MATLAB: Rguard{i} = guardIntersect_levelSet(loc,P{i},guard);
            Rguard[i] = loc.guardIntersect_levelSet(P[i], guard)
            
        else:
            raise CORAerror('CORA:wrongFieldValue',
                          'options.guardIntersect',
                          {'polytope', 'conZonotope', 'zonoGirard',
                           'hyperplaneMap', 'pancake', 'nondetGuard', 'levelSet'})
    
    # remove all empty intersections
    Rguard, minInd, maxInd, actGuards = _aux_removeEmptySets(Rguard, minInd, maxInd, actGuards)
    
    # convert sets back to polynomial zonotopes
    if isinstance(params['R0'], PolyZonotope):
        for i in range(len(Rguard)):
            if isinstance(Rguard[i], Zonotope):
                Rguard[i] = PolyZonotope(Rguard[i])
    
    return Rguard, actGuards, minInd, maxInd


# Auxiliary functions -----------------------------------------------------

def _aux_groupSets(Pset: List[Any], guards: List[int], setIndices: List[int]) -> Tuple[List[int], List[int], List[List[Any]], List[int]]:
    """
    Group the reachable sets which intersect guard sets. The sets in one
    group all intersect the same guard set and are located next to each other
    
    Args:
        Pset: list of reachable sets
        guards: list of guard indices
        setIndices: list of set indices
        
    Returns:
        minInd: minimum indices for each group
        maxInd: maximum indices for each group
        P: grouped sets
        guards: guard indices for each group
    """
    
    # initialization
    guardInd = np.unique(guards)
    setIndicesGuards = [None] * len(guardInd)
    P = [None] * len(guardInd)
    
    # Step 1: group according to hit guard sets
    for i in range(len(guardInd)):
        ind = np.where(guards == guardInd[i])[0]
        setIndicesGuards[i] = [setIndices[j] for j in ind]
        P[i] = [Pset[j] for j in ind]
    
    # Step 2: group according to the location (neighbouring sets together)
    minInd, maxInd, Pint, guards_out = _aux_removeGaps(setIndicesGuards, guardInd, P)
    
    return minInd, maxInd, Pint, guards_out


def _aux_removeGaps(setIndicesGuards: List[List[int]], guards: np.ndarray, 
                    Pint: List[List[Any]]) -> Tuple[List[int], List[int], List[List[Any]], List[int]]:
    """
    Remove gaps in the set-index vector of each intersection
    
    Args:
        setIndicesGuards: list of set index lists for each guard
        guards: array of guard indices
        Pint: list of set lists for each guard
        
    Returns:
        minInd: minimum indices
        maxInd: maximum indices
        Pint: updated grouped sets
        guards: updated guard indices
    """
    
    # split all guard intersections with gaps between the set-indices into
    # multiple different intersections (needed if one guard set is hit
    # multiple times at different points in time)
    counter = 0
    
    while counter < len(guards):
        
        setIndices = setIndicesGuards[counter]
        
        for i in range(len(setIndices) - 1):
            
            # check if a gap occurs
            if setIndices[i+1] != setIndices[i] + 1:
                # add first part of the intersection (=gap free) to the beginning of the list
                setIndicesGuards.insert(0, setIndices[:i+1])
                Pint.insert(0, Pint[counter][:i+1])
                guards = np.concatenate([[guards[counter]], guards])
                
                # add second part of the intersection (possibly contains further gaps) 
                # to the part of the list that is not finished yet
                setIndicesGuards[counter+1] = setIndices[i+1:]
                Pint[counter+1] = Pint[counter+1][i+1:]
                
                break
        
        counter += 1
    
    # determine minimum and maximum set-index for each intersection
    minInd = [x[0] for x in setIndicesGuards]
    maxInd = [x[-1] for x in setIndicesGuards]
    
    return minInd, maxInd, Pint, guards.tolist()


def _aux_removeEmptySets(R: List[Any], minInd: List[int], maxInd: List[int], 
                         actGuards: List[int]) -> Tuple[List[Any], List[int], List[int], List[int]]:
    """
    Remove all sets for which the intersection with the guard set turned out to be empty
    
    Args:
        R: list of guard intersection sets
        minInd: minimum indices
        maxInd: maximum indices
        actGuards: active guard indices
        
    Returns:
        R: filtered list of non-empty sets
        minInd: filtered minimum indices
        maxInd: filtered maximum indices
        actGuards: filtered active guard indices
    """
    from cora_python.contSet.zonoBundle import ZonoBundle
    
    # get the indices of the non-empty sets
    ind = []
    
    for i in range(len(R)):
        if R[i] is None:
            continue
        if isinstance(R[i], ZonoBundle):
            if any(S.representsa_('emptySet', np.finfo(float).eps) for S in R[i].Z):
                continue
        elif R[i].representsa_('emptySet', np.finfo(float).eps):
            continue
        ind.append(i)
    
    # update variables
    R = [R[i] for i in ind]
    minInd = [minInd[i] for i in ind]
    maxInd = [maxInd[i] for i in ind]
    actGuards = [actGuards[i] for i in ind]
    
    return R, minInd, maxInd, actGuards


def _aux_getInitialSet(Rtp: List[Any], minInd: int) -> Any:
    """
    Get the initial set
    
    Args:
        Rtp: list of time-point reachable sets
        minInd: minimum index
        
    Returns:
        R0: initial set
        
    Raises:
        CORAerror: if initial set already intersects guard
    """
    
    if minInd == 1:
        raise CORAerror('CORA:specialError',
                      'The initial set already intersects the guard set!')
    else:
        R0 = Rtp[minInd - 1]
    
    return R0

