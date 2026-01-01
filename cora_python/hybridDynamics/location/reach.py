"""
reach - computes the reachable set of the system within a location and
   determines the intersections with the guard sets

Syntax:
    [R,Rjump_,res] = reach(loc,params,options)

Inputs:
    loc - location object
    params - model parameters
    options - struct containing the algorithm settings

Outputs:
    R - reachable set due to continuous evolution
    Rjump_ - list of guard set intersections with the corresponding sets
    res - true/false whether specifications are satisfied

See also: hybridAutomaton/reach

Authors:       Matthias Althoff, Niklas Kochdumper
Written:       07-May-2007 
Last update:   17-August-2007
               31-July-2016
               19-August-2016
               09-December-2019 (NK, integrated singleSetReach)
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import Any, Dict, List, Tuple, Optional
from cora_python.contSet.interval.interval import Interval
from cora_python.specification.specification.specification import Specification
from cora_python.specification.specification.add import add
from cora_python.specification.specification.check import check
# Methods are attached to Location class in location/__init__.py
# Use loc.potInt(), loc.potOut(), loc.guardIntersect() instead of standalone imports
from cora_python.g.classes.reachSet.reachSet import ReachSet
from cora_python.g.classes.reachSet.updateTime import updateTime


def reach(loc: Any, params: Dict[str, Any], options: Dict[str, Any]) -> Tuple[Any, List[Dict[str, Any]], bool]:
    """
    Computes the reachable set of the system within a location and
    determines the intersections with the guard sets
    
    Args:
        loc: location object
        params: model parameters
        options: struct containing the algorithm settings
        
    Returns:
        R: reachable set due to continuous evolution
        Rjump_: list of guard set intersections with the corresponding sets
        res: true/false whether specifications are satisfied
    """
    
    res = True
    Rjump = []
    tStart_interval = params['tStart']
    params['tStart'] = tStart_interval.infimum() if hasattr(tStart_interval, 'infimum') else float(tStart_interval)
    
    # adapt specifications
    specReach, specCheck = _aux_adaptSpecs(loc, options.get('specification', None))
    
    # since we require the reachable set for the guard intersection and not the
    # output set, we set the internal option 'compOutputSet' to false; the
    # output set will then be computed in hybridAutomaton/reach after all
    # computation in the location are finished
    options['compOutputSet'] = False
    
    # compute reachable set for the continuous dynamics until the reachable
    # set is fully located outside the invariant set
    R = loc.contDynamics.reach(params, options, specReach)
    
    # Handle case where R is a list (multiple branches)
    if isinstance(R, list):
        R_list = R
    else:
        R_list = [R]
    
    # loop over all reachable sets (the number of reachable sets may
    # increase if the sets are split during the computation)
    for i in range(len(R_list)):
        
        # determine all guard sets of the current location which any
        # reachable set intersects
        # MATLAB: [guards,setIndices,setType] = potInt(loc,R_list{i},params.finalLoc);
        guards, setIndices, setType = loc.potInt(R_list[i], params.get('finalLoc', None))
        
        # compute intersections with the guard sets
        # MATLAB: [Rguard,actGuards,minInd,maxInd] = guardIntersect(loc,guards,setIndices,setType,R_list{i},params,options);
        Rguard, actGuards, minInd, maxInd = loc.guardIntersect(
            guards, setIndices, setType, R_list[i], params, options)
        
        # compute reset and get target location
        Rjump_ = []
        
        for j in range(len(Rguard)):
            
            iGuard = actGuards[j]
            
            # compute reset
            reset_obj = loc.transition[iGuard].reset
            Rjump_set = reset_obj.evaluate(Rguard[j], params.get('U', None))
            
            # target location and parent reachable set
            target_loc = loc.transition[iGuard].target
            parent = R_list[i].parent + 1 if hasattr(R_list[i], 'parent') else 1
            
            # time interval for the guard intersection
            if setType == 'time-interval':
                tMin = R_list[i].timeInterval['time'][minInd[j] - 1].infimum() if hasattr(R_list[i].timeInterval['time'][minInd[j] - 1], 'infimum') else R_list[i].timeInterval['time'][minInd[j] - 1]
                tMax = R_list[i].timeInterval['time'][maxInd[j] - 1].supremum() if hasattr(R_list[i].timeInterval['time'][maxInd[j] - 1], 'supremum') else R_list[i].timeInterval['time'][maxInd[j] - 1]
                if hasattr(tStart_interval, 'rad'):
                    tMax = tMax + 2 * tStart_interval.rad()
                else:
                    tMax = tMax + 2 * ((tStart_interval.sup if hasattr(tStart_interval, 'sup') else tStart_interval[1]) - (tStart_interval.inf if hasattr(tStart_interval, 'inf') else tStart_interval[0])) / 2
            else:
                tMin = R_list[i].timePoint['time'][minInd[j] - 1]
                tMax = R_list[i].timePoint['time'][maxInd[j] - 1]
            
            Rjump_time = Interval(tMin, tMax)
            
            Rjump_.append({
                'set': Rjump_set,
                'time': Rjump_time,
                'loc': target_loc,
                'parent': parent
            })
        
        Rjump.extend(Rjump_)
        
        # remove the parts of the reachable sets outside the invariant
        # MATLAB: R_list{i} = potOut(loc,R_list{i},minInd,maxInd);
        if options.get('intersectInvariant', False):
            R_list[i] = loc.potOut(R_list[i], minInd, maxInd)
        
        # update times of the reachable set due to uncertain initial time
        R_list[i] = updateTime(R_list[i], tStart_interval)
        
        # check if specifications are violated
        if specCheck is not None and len(specCheck) > 0:
            res_check, _, _ = check(specCheck, R_list[i].timeInterval['set'][-1])
            if not res_check:
                res = False
                return R_list[0] if len(R_list) == 1 else R_list, Rjump, res
    
    return R_list[0] if len(R_list) == 1 else R_list, Rjump, res


# Auxiliary functions -----------------------------------------------------

def _aux_adaptSpecs(loc: Any, specs: Optional[List[Specification]]) -> Tuple[Specification, List[Specification]]:
    """
    Adapt specifications for location reachability analysis
    
    Args:
        loc: location object
        specs: list of specifications
        
    Returns:
        specReach: specifications for reachability (including invariant)
        specCheck: specifications for checking (excluding invariant)
    """
    
    # add the invariant as a specification so that reachability analysis of
    # continuous dynamics can exit once the reachable set fully leaves the
    # invariant
    specReach = Specification(loc.invariant, 'invariant')
    
    # add other specification to the list of specifications: note that unsafe
    # sets must intersect the invariant to be valid
    if specs is not None and len(specs) > 0:
        # double check because numel(specification()) == 1
        for i in range(len(specs)):
            if (specs[i].type != 'unsafeSet' or
                loc.invariant.isIntersecting_(specs[i].set, 'approx', 1e-8)):
                specReach = add(specReach, specs[i])
    
    # for the check, we skip the 'invariant' specification
    if isinstance(specReach, list) and len(specReach) > 1:
        specCheck = specReach[1:]
    elif isinstance(specReach, Specification):
        specCheck = []
    else:
        specCheck = specReach[1:] if len(specReach) > 1 else []
    
    return specReach, specCheck

