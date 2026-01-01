"""
reach - computes the reachable set of a hybrid automaton

Syntax:
    R = reach(HA,params,options)
    [R,res] = reach(HA,params,options,spec)

Inputs:
    HA - hybridAutomaton object
    params - parameter defining the reachability problem
    options - options for the computation of the reachable set
    spec - (optional) object of class specification

Outputs:
    R - reachSet object storing the reachable set
    res - true/false whether specifications are satisfied

See also: location/reach

Authors:       Matthias Althoff, Niklas Kochdumper
Written:       07-May-2007 
Last update:   16-August-2007
               20-August-2013
               30-October-2015
               22-August-2016
               19-December-2019 (NK, restructured the algorithm)
               13-October-2021 (MP, location-specific specifications)
               27-November-2022 (MW, restructure specification syntax)
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import Any, Dict, List, Tuple, Optional
from cora_python.contSet.interval.interval import Interval
from cora_python.contSet.fullspace.fullspace import Fullspace
from cora_python.specification.specification.specification import Specification
from cora_python.specification.specification.add import add
from cora_python.g.classes.reachSet.reachSet import ReachSet
from cora_python.g.functions.matlab.validate.preprocessing.setDefaultValues import setDefaultValues
from cora_python.hybridDynamics.location.reach import reach as location_reach
from cora_python.hybridDynamics.hybridAutomaton.private.priv_flowDerivatives import priv_flowDerivatives
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def reach(HA: Any, params: Dict[str, Any], options: Dict[str, Any], *varargin) -> Tuple[List[ReachSet], bool]:
    """
    Computes the reachable set of a hybrid automaton
    
    Args:
        HA: hybridAutomaton object
        params: parameter defining the reachability problem
        options: options for the computation of the reachable set
        *varargin: optional specification object
        
    Returns:
        R: list of reachSet objects storing the reachable set
        res: true/false whether specifications are satisfied
    """
    
    res = True
    spec = setDefaultValues([None], list(varargin))[0]
    
    # options preprocessing
    # Note: validateOptions may not be fully implemented yet
    # For now, we'll proceed assuming options are validated
    if 'validateOptions' in globals() or hasattr(HA, 'validateOptions'):
        if hasattr(HA, 'validateOptions'):
            params, options = HA.validateOptions(params, options)
        else:
            # Fallback: basic validation
            pass
    
    # compute derivatives for each location
    if 'paramInt' in params:
        # this is required for derivatives() call for nonlinParamSys
        options['paramInt'] = params['paramInt']
    priv_flowDerivatives(HA, options)
    
    # check specifications
    options['specificationLoc'] = _aux_check_flatHA_specification(HA, spec)
    
    # initialize reachable set: we use a fixed size to start with and then
    # double the size if the current size is exceeded; this process avoids
    # costly 'and-1' concatenation
    R = [None] * 10
    # index to append new reachSet objects to full list
    r = 0
    
    # initialize queue for reachable set computation (during the
    # computation, multiple branches of reachable set can emerge, requiring
    # to compute the successor reachable sets for all branches one after
    # the other; the queue handles this process)
    list_queue = [{
        'set': params['R0'],
        'loc': params['startLoc'],
        'time': Interval(params['tStart']) if not isinstance(params['tStart'], Interval) else params['tStart'],
        'parent': 0
    }]
    
    # display information on command window
    _aux_verbose_displayStart(options.get('verbose', False))
    
    # loop until the queue is empty or a specification is violated
    while len(list_queue) > 0 and res:
        
        # get location, initial set, start time, and parent branch for
        # reachable set computation of first element in the queue
        locID = list_queue[0]['loc']
        R0 = list_queue[0]['set']
        tStart = list_queue[0]['time']
        parent = list_queue[0]['parent']
        
        # get inputs and specification for the current location
        params_loc = params.copy()
        if 'Uloc' in params:
            params_loc['U'] = params['Uloc'][locID - 1]  # Convert to 0-based indexing
        if 'uloc' in params:
            params_loc['u'] = params['uloc'][locID - 1]
        if 'Wloc' in params:
            params_loc['W'] = params['Wloc'][locID - 1]
        if 'Vloc' in params:
            params_loc['V'] = params['Vloc'][locID - 1]
        options_loc = options.copy()
        if 'specificationLoc' in options:
            options_loc['specification'] = options['specificationLoc'][locID - 1]
        # get timeStep for the current location (unless adaptive)
        if options.get('linAlg', 'adaptive') != 'adaptive' and 'timeStepLoc' in options:
            options_loc['timeStep'] = options['timeStepLoc'][locID - 1]
        
        # check if current location has an instant transition
        instantTransition = []
        for trans in HA.location[locID - 1].transition:
            if isinstance(trans.guard, Fullspace) or (hasattr(trans.guard, 'representsa_') and trans.guard.representsa_('fullspace', 1e-12)):
                instantTransition.append(True)
            else:
                instantTransition.append(False)
        instantTransition = np.array(instantTransition)
        
        if np.any(instantTransition):
            
            # save reachable set to array
            temp = {
                'set': [R0],
                'time': [tStart]
            }
            if r == len(R):
                R.extend([None] * len(R))  # Double the size
            R[r] = ReachSet(temp, {}, parent, locID)
            # increment counter
            r += 1
            
            # here, we overwrite the first entry in list and continue the
            # reachability analysis with this set -- in contrast to below,
            # where the new sets are appended at the end of the list
            
            # compute derivatives of reset functions for current location
            HA = HA.derivatives(locID) if hasattr(HA, 'derivatives') else HA
            
            # find first instant transition
            instant_idx = np.where(instantTransition)[0][0]
            
            # append to the end of the list
            reset_obj = HA.location[locID - 1].transition[instant_idx].reset
            R0_reset = reset_obj.evaluate(R0, params_loc.get('U', None))
            target_loc = HA.location[locID - 1].transition[instant_idx].target
            
            list_queue[0] = {
                'set': R0_reset,
                'loc': target_loc,
                'parent': r
            }
            
            # print on command window that an instant transition has occurred
            _aux_verbose_displayInstantTransition(options.get('verbose', False), 
                                                  list_queue[0]['loc'], locID, tStart)
            continue
        
        else:
            _aux_verbose_displayReach(options.get('verbose', False), locID, tStart)
            
            # compute derivatives of reset functions for current location
            HA = HA.derivatives(locID) if hasattr(HA, 'derivatives') else HA
            
            # compute the reachable set within a location until either the
            # final time is reached or the reachable set hits a guard set
            # and the computation proceeds in another location
            params_loc['R0'] = R0
            params_loc['tStart'] = tStart
            Rtemp, Rjump, res = location_reach(HA.location[locID - 1], params_loc, options_loc)
            
            _aux_verbose_displayJump(options.get('verbose', False), Rtemp)
        
        # remove current element from the queue
        list_queue = list_queue[1:]
        
        # add the new branches of reachable sets to the queue
        for i in range(len(Rjump)):
            Rjump[i]['parent'] = Rjump[i]['parent'] + r
        list_queue.extend(Rjump)
        
        # display transitions on command window
        _aux_verbose_displayTransition(options.get('verbose', False), Rjump, locID)
        
        # store the computed reachable set
        # Handle case where Rtemp is a list or single object
        if isinstance(Rtemp, list):
            Rtemp_list = Rtemp
        else:
            Rtemp_list = [Rtemp]
        
        for i in range(len(Rtemp_list)):
            # compute output set
            Ytemp = _aux_outputSet(Rtemp_list[i], HA.location[locID - 1], params_loc, options_loc)
            # init reachSet object and append to full list
            temp = ReachSet(Ytemp['timePoint'], Ytemp['timeInterval'], parent, locID)
            if r == len(R):
                R.extend([None] * len(R))  # Double the size
            R[r] = temp
            # increment counter
            r += 1
    
    # truncate reachable set (empty entries at the end due to
    # pre-allocation of memory)
    R = R[:r]
    
    _aux_verbose_displayEnd(options.get('verbose', False))
    
    return R, res


# Auxiliary functions -----------------------------------------------------

def _aux_check_flatHA_specification(HA: Any, spec: Optional[List[Specification]]) -> List[List[Specification]]:
    """
    Rewrites specifications in the correct format
    """
    
    numLoc = len(HA.location)
    
    # initialize specifications with empty cells
    if spec is None or len(spec) == 0:
        return [[] for _ in range(numLoc)]
    
    # adjust specification for each location
    if not isinstance(spec, list) or (isinstance(spec, list) and len(spec) > 0 and isinstance(spec[0], Specification)):
        
        # number of specifications
        if isinstance(spec, Specification):
            nrSpecs = 1
            spec_list = [spec]
        else:
            nrSpecs = len(spec)
            spec_list = spec
        
        # checks
        for i in range(nrSpecs):
            spec_i = spec_list[i]
            # ensure that time information is not provided (unsupported)
            if (hasattr(spec_i, 'time') and spec_i.time is not None and
                hasattr(spec_i.time, 'representsa_') and 
                not spec_i.time.representsa_('emptySet', np.finfo(float).eps)):
                raise CORAerror('CORA:notSupported',
                              'Timed specifications are not yet supported for hybrid automata!')
            # ensure that no specification is active in a non-existing location
            if (hasattr(spec_i, 'location') and spec_i.location is not None and
                isinstance(spec_i.location, (list, np.ndarray)) and
                np.any(np.array(spec_i.location) > numLoc)):
                raise CORAerror('CORA:wrongValue', 'fourth',
                              'spec.location must not exceed the number of locations in the hybrid automaton.')
        
        specificationLoc = [[] for _ in range(numLoc)]
        # if spec.location = [], specification is assumed to be active in
        # all locations
        
        for i in range(numLoc):
            for j in range(nrSpecs):
                spec_j = spec_list[j]
                if (not hasattr(spec_j, 'location') or spec_j.location is None or
                    (isinstance(spec_j.location, (list, np.ndarray)) and len(spec_j.location) == 0) or
                    (isinstance(spec_j.location, (list, np.ndarray)) and (i + 1) in spec_j.location)):
                    specificationLoc[i] = add(specificationLoc[i], spec_j) if len(specificationLoc[i]) > 0 else [spec_j]
        
        return specificationLoc
    
    # copy specification for each location
    else:
        
        # check if the number of cells matches the number of locations
        if len(spec) != numLoc:
            raise CORAerror('CORA:notSupported',
                          'Input argument "spec" has the wrong format!')
        
        # check if time information is provided
        for i in range(len(spec)):
            for j in range(len(spec[i])):
                spec_ij = spec[i][j]
                if (hasattr(spec_ij, 'time') and spec_ij.time is not None and
                    hasattr(spec_ij.time, 'representsa_') and 
                    not spec_ij.time.representsa_('emptySet', np.finfo(float).eps)):
                    raise CORAerror('CORA:notSupported',
                                  'Timed specifications are not yet supported for hybrid automata!')
        
        # copy specifications
        return spec


def _aux_outputSet(Rtemp: Any, loc: Any, params: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
    """
    Since we require the reachable set in the entire computation due to guard
    intersections and the preparation of the start set for the next location,
    we only compute the output set at the end of the analysis of each
    location (using the outputSet-functions in contDynamics)
    """
    
    # rewrite options.u of current location to options.uTrans for outputSet()
    # TODO: do this in validateOptions?
    if 'u' in params:
        params_output = params.copy()
        params_output['uTrans'] = params['u']
    else:
        params_output = params.copy()
    
    # init
    Ytemp = {
        'timePoint': {},
        'timeInterval': {}
    }
    
    # time-point solution
    if hasattr(Rtemp, 'timePoint') and Rtemp.timePoint is not None and len(Rtemp.timePoint.get('set', [])) > 0:
        # time-point solution
        nrTimePointSets = len(Rtemp.timePoint['set'])
        Ytemp['timePoint']['set'] = [None] * nrTimePointSets
        Ytemp['timePoint']['time'] = Rtemp.timePoint['time']
        for i in range(len(Rtemp.timePoint['set'])):
            result = loc.contDynamics.outputSet(
                Rtemp.timePoint['set'][i], params_output, options)
            # Handle both single return and tuple return (Y, Verror)
            Ytemp['timePoint']['set'][i] = result[0] if isinstance(result, tuple) else result
    
    # time-interval solution
    if hasattr(Rtemp, 'timeInterval') and Rtemp.timeInterval is not None and len(Rtemp.timeInterval.get('set', [])) > 0:
        nrTimeIntervalSets = len(Rtemp.timeInterval['set'])
        Ytemp['timeInterval']['set'] = [None] * nrTimeIntervalSets
        Ytemp['timeInterval']['time'] = Rtemp.timeInterval['time']
        for i in range(len(Rtemp.timeInterval['set'])):
            result = loc.contDynamics.outputSet(
                Rtemp.timeInterval['set'][i], params_output, options)
            # Handle both single return and tuple return (Y, Verror)
            Ytemp['timeInterval']['set'][i] = result[0] if isinstance(result, tuple) else result
    
    # parent
    Ytemp['parent'] = Rtemp.parent if hasattr(Rtemp, 'parent') else 0
    
    return Ytemp


# logging functions below... (only print if options.verbose = true)

def _aux_verbose_displayStart(verbose: bool) -> None:
    """Display start message"""
    if not verbose:
        return
    print("Start analysis...")


def _aux_verbose_displayEnd(verbose: bool) -> None:
    """Display end message"""
    if not verbose:
        return
    print("...time horizon reached, analysis finished.\n")


def _aux_verbose_displayInstantTransition(verbose: bool, loc: int, locID: int, tStart: Any) -> None:
    """Display instant transition"""
    if not verbose:
        return
    print(f"  transition: location {locID} -> location {loc}... (time: {tStart})")


def _aux_verbose_displayReach(verbose: bool, locID: int, tStart: Any) -> None:
    """Display information about reachability analysis"""
    if not verbose:
        return
    print(f"Compute reachable set in location {locID}... (time: {tStart} to ", end='')


def _aux_verbose_displayJump(verbose: bool, Rtemp: Any) -> None:
    """Display information about guard intersection"""
    if not verbose:
        return
    if isinstance(Rtemp, list) and len(Rtemp) > 0:
        Rtemp_first = Rtemp[0]
    else:
        Rtemp_first = Rtemp
    if hasattr(Rtemp_first, 'timePoint') and Rtemp_first.timePoint is not None:
        time_end = Rtemp_first.timePoint['time'][-1] if len(Rtemp_first.timePoint.get('time', [])) > 0 else '?'
        print(f"{time_end})")
    else:
        print("?)")


def _aux_verbose_displayTransition(verbose: bool, list_transitions: List[Dict[str, Any]], locID: int) -> None:
    """Only if verbose = true: print outgoing transitions with target location
    identifier and time during which the transition has occurred
    """
    if not verbose:
        return
    
    if len(list_transitions) == 0:
        return
    
    elif len(list_transitions) == 1:
        print(f"  transition: location {locID} -> location {list_transitions[0]['loc']}... "
              f"(time: {list_transitions[0]['time']})")
    
    else:
        # print 'header'
        print("  transitions: ", end='')
        # indent not in first line
        indent = ""
        
        # loop over multiple transitions
        for i in range(len(list_transitions)):
            print(f"{indent}location {locID} -> location {list_transitions[i]['loc']}... "
                  f"(time: {list_transitions[i]['time']})")
            # add indent for vertical alignment to all other transitions
            indent = "               "

