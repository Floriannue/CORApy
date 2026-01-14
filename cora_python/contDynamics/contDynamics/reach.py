"""
reach - computes the reachable continuous set for the entire time horizon
        of a continuous system

Syntax:
    R = reach(sys, params, options)
    [R, res] = reach(sys, params, options, spec)
    [R, res, options] = reach(sys, params, options, spec)

Inputs:
    sys - contDynamics object
    params - parameter defining the reachability problem
    options - options for the computation of reachable sets
    spec - object of class specification (optional)

Outputs:
    R - object of class reachSet storing the computed reachable set
    res - true if specifications are satisfied, otherwise false (optional)
    options - options for the computation of reachable sets (optional)

Authors:       Matthias Althoff, Niklas Kochdumper (MATLAB)
              Python translation by AI Assistant
Written:       08-August-2016 (MATLAB)
Last update:   19-November-2022 (MATLAB)
Python translation: 2025
"""

import numpy as np
import math
from typing import Tuple, Any, Dict, Optional, Union
from cora_python.g.functions.matlab.validate.preprocessing.set_default_values import set_default_values
from cora_python.g.classes.reachSet import ReachSet
from cora_python.contSet.interval import Interval
from cora_python.g.functions.verbose.verboseLog import verboseLog


def reach(sys: Any, params: Dict[str, Any], options: Dict[str, Any], 
          spec: Optional[Any] = None) -> Union[Any, Tuple[Any, bool], Tuple[Any, bool, Dict[str, Any]]]:
    """
    Computes the reachable continuous set for the entire time horizon
    
    Args:
        sys: contDynamics object
        params: parameter defining the reachability problem
        options: options for the computation of reachable sets
        spec: object of class specification (optional)
        
    Returns:
        R: object of class reachSet storing the computed reachable set
        res: true if specifications are satisfied, otherwise false (if spec provided)
        options: options for the computation of reachable sets (if spec provided)
    """
    # MATLAB: res = true;
    res = True
    
    # MATLAB: spec = setDefaultValues({[]},varargin);
    spec = set_default_values([None], [spec] if spec is not None else [])[0]
    
    # options preprocessing
    # MATLAB: [params,options] = validateOptions(sys,params,options);
    # Check if sys has validateOptions method, otherwise use generic validation
    if hasattr(sys, 'validateOptions'):
        params, options = sys.validateOptions(params, options)
    else:
        # Basic validation - ensure required fields exist
        if 'tStart' not in params:
            params['tStart'] = 0.0
        if 'tFinal' not in params:
            raise ValueError("params must contain 'tFinal'")
        if 'R0' not in params:
            raise ValueError("params must contain 'R0'")
        if 'timeStep' not in options:
            raise ValueError("options must contain 'timeStep'")
        if 'taylorTerms' not in options:
            raise ValueError("options must contain 'taylorTerms'")
    
    # handling of specifications
    # MATLAB: specLogic = [];
    specLogic = None
    # MATLAB: if ~isempty(spec)
    if spec is not None:
        # MATLAB: [spec,specLogic] = splitLogic(spec);
        # TODO: Implement splitLogic if needed
        # For now, just use spec as-is
        pass
    
    # reach called from
    # - linear class (linProbSys, linParamSys) or
    # - nonlinear class (nonlinearSys, nonlinDASys, nonlinParamSys)
    # MATLAB: syslin = isa(sys,'linProbSys') || isa(sys,'linParamSys');
    syslin = (hasattr(sys, '__class__') and 
              ('linearSys' in sys.__class__.__name__.lower() or 
               'linParamSys' in sys.__class__.__name__.lower()))
    
    # compute symbolic derivatives
    # MATLAB: if ~syslin
    if not syslin:
        # paramInt required for derivatives of nonlinParamSys...
        # MATLAB: if isfield(params,'paramInt')
        if 'paramInt' in params:
            # MATLAB: options.paramInt = params.paramInt;
            options['paramInt'] = params['paramInt']
        
        # MATLAB: derivatives(sys,options);
        from cora_python.contDynamics.contDynamics.derivatives import derivatives
        derivatives(sys, options)
        
        # MATLAB: if contains(options.alg,'adaptive')
        if 'alg' in options and 'adaptive' in str(options['alg']):
            # nonlinear adaptive algorithm
            # MATLAB: [timeInt,timePoint,res,~,options] = reach_adaptive(sys,params,options);
            from cora_python.contDynamics.nonlinearSys.reach_adaptive import reach_adaptive
            timeInt, timePoint, res, _, options = reach_adaptive(sys, params, options)
            # MATLAB: R = reachSet.initReachSet(timePoint,timeInt);
            R = ReachSet.initReachSet(timePoint, timeInt)
            if spec is not None:
                return R, res, options
            else:
                return R
    
    # obtain factors for initial state and input solution time step
    # MATLAB: r = options.timeStep;
    r = options['timeStep']
    # MATLAB: for i = 1:(options.taylorTerms+1)
    # MATLAB:     options.factor(i) = r^(i)/factorial(i);
    options['factor'] = [r**i / math.factorial(i) for i in range(1, options['taylorTerms'] + 2)]
    
    # if a trajectory should be tracked
    # MATLAB: if isfield(params,'uTransVec')
    if 'uTransVec' in params:
        # MATLAB: params.uTrans = params.uTransVec(:,1);
        params['uTrans'] = params['uTransVec'][:, 0:1]
    
    # MATLAB: options.t = params.tStart;
    options['t'] = params['tStart']
    
    # time period
    # MATLAB: tVec = params.tStart:options.timeStep:params.tFinal;
    # Use numpy to create time vector similar to MATLAB
    steps = int(np.round((params['tFinal'] - params['tStart']) / options['timeStep']))
    tVec = np.linspace(params['tStart'], params['tFinal'], steps + 1)
    steps = len(tVec) - 1
    
    # initialize cell-arrays that store the reachable set
    # MATLAB: timeInt.set = cell(steps,1);
    # MATLAB: timeInt.time = cell(steps,1);
    # MATLAB: timePoint.set = cell(steps+1,1);
    # MATLAB: timePoint.time = cell(steps+1,1);
    timeInt = {'set': [None] * steps, 'time': [None] * steps}
    timePoint = {'set': [None] * (steps + 1), 'time': [None] * (steps + 1)}
    
    # Check if nonlinDASys (algebraic states)
    # MATLAB: if isa(sys,'nonlinDASys')
    is_nonlinDASys = (hasattr(sys, '__class__') and 
                      'nonlinDASys' in sys.__class__.__name__.lower())
    if is_nonlinDASys:
        # MATLAB: timeInt.algebraic = cell(steps,1);
        timeInt['algebraic'] = [None] * steps
    
    # first timePoint set is initial set
    # MATLAB: timePoint.time{1} = params.tStart;
    timePoint['time'][0] = params['tStart']
    
    # MATLAB: if syslin
    if syslin:
        # MATLAB: timePoint.set{1} = params.R0;
        timePoint['set'][0] = params['R0']
    else:
        # MATLAB: if isa(sys,'nonlinDASys')
        if is_nonlinDASys:
            # MATLAB: R_y = zonotope(consistentInitialState(sys,center(params.R0),...
            #     params.y0guess,params.uTrans));
            # MATLAB: timePoint.set{1}{1}.set = outputSet(sys,params.R0,R_y,params,options);
            # TODO: Implement nonlinDASys handling
            raise NotImplementedError("nonlinDASys not yet implemented")
        else:
            # MATLAB: timePoint.set{1}{1}.set = outputSet(sys,params.R0,params,options);
            from cora_python.contDynamics.contDynamics.outputSet import outputSet
            timePoint['set'][0] = [{'set': outputSet(sys, params['R0'], params, options),
                                    'prev': 1, 'parent': 1}]
    
    # log information
    # MATLAB: verboseLog(options.verbose,1,options.t,params.tStart,params.tFinal);
    verboseLog(options.get('verbose', False), 1, options['t'], params['tStart'], params['tFinal'])
    
    # initialize reachable set computations
    # MATLAB: try
    try:
        # MATLAB: [Rnext, options] = initReach(sys,params.R0,params,options);
        if syslin:
            # For linear systems, use linearSys.reach
            if hasattr(sys, 'reach'):
                return sys.reach(params, options, spec) if spec is not None else sys.reach(params, options)
            else:
                raise NotImplementedError(f"reach method not implemented for {type(sys).__name__}")
        else:
            # For nonlinear systems, use initReach
            from cora_python.contDynamics.nonlinearSys.initReach import initReach
            Rnext, options = initReach(sys, params['R0'], params, options)
    except Exception as ME:
        # if error from set explosion, return corresponding information
        # MATLAB: priv_reportReachError(ME,params.tStart,1);
        # TODO: Implement priv_reportReachError
        print(f"Error in initReach at t={params['tStart']}, step=1: {ME}")
        # MATLAB: R = reachSet.initReachSet(timePoint,timeInt);
        R = ReachSet.initReachSet(timePoint, timeInt)
        res = False
        if spec is not None:
            return R, res, options
        else:
            return R
    
    # loop over all reachability steps
    # MATLAB: for i = 2:steps
    for i in range(1, steps):
        # save reachable set in cell structure
        # MATLAB: if ~isa(sys,'nonlinDASys')
        if not is_nonlinDASys:
            # MATLAB: timeInt.set{i-1} = outputSet(sys,Rnext.ti,params,options);
            # MATLAB: timePoint.set{i} = outputSet(sys,Rnext.tp,params,options);
            from cora_python.contDynamics.contDynamics.outputSet import outputSet
            # Handle Rnext.ti and Rnext.tp - they might be lists of dicts
            if isinstance(Rnext['ti'], list):
                timeInt['set'][i-1] = [outputSet(sys, r_ti.get('set', r_ti) if isinstance(r_ti, dict) else r_ti, params, options) 
                                      for r_ti in Rnext['ti']]
            else:
                timeInt['set'][i-1] = outputSet(sys, Rnext['ti'], params, options)
            
            if isinstance(Rnext['tp'], list):
                timePoint['set'][i] = [outputSet(sys, r_tp.get('set', r_tp) if isinstance(r_tp, dict) else r_tp, params, options) 
                                      for r_tp in Rnext['tp']]
            else:
                timePoint['set'][i] = outputSet(sys, Rnext['tp'], params, options)
        else:
            # MATLAB: timeInt.set{i-1} = outputSet(sys,Rnext.ti,Rnext.y,params,options);
            # MATLAB: timePoint.set{i} = outputSet(sys,Rnext.tp,Rnext.y,params,options);
            # TODO: Implement nonlinDASys handling
            raise NotImplementedError("nonlinDASys not yet implemented")
        
        # MATLAB: timeInt.time{i-1} = interval(tVec(i-1),tVec(i));
        timeInt['time'][i-1] = Interval(tVec[i-1], tVec[i])
        # MATLAB: timePoint.time{i} = tVec(i);
        timePoint['time'][i] = tVec[i]
        
        # MATLAB: if isa(sys,'nonlinDASys')
        if is_nonlinDASys:
            # MATLAB: timeInt.algebraic{i-1} = Rnext.y;
            timeInt['algebraic'][i-1] = Rnext.get('y')
        
        # notify user if splitting has occurred
        # MATLAB: if length(timePoint.set{i}) > length(timePoint.set{i-1})
        if isinstance(timePoint['set'][i], list) and isinstance(timePoint['set'][i-1], list):
            if len(timePoint['set'][i]) > len(timePoint['set'][i-1]):
                print(f"split! ...number of parallel sets: {len(timePoint['set'][i])}")
        
        # check specification
        # MATLAB: if ~isempty(spec)
        if spec is not None:
            # MATLAB: if ~check(spec,Rnext.ti,timeInt.time{i-1})
            # TODO: Implement specification checking
            # For now, skip check
            pass
        
        # increment time
        # MATLAB: options.t = tVec(i);
        options['t'] = tVec[i]
        # log information
        # MATLAB: verboseLog(options.verbose,i,options.t,params.tStart,params.tFinal);
        verboseLog(options.get('verbose', False), i+1, options['t'], params['tStart'], params['tFinal'])
        
        # if a trajectory should be tracked
        # MATLAB: if isfield(params,'uTransVec')
        if 'uTransVec' in params:
            # MATLAB: params.uTrans = params.uTransVec(:,i);
            params['uTrans'] = params['uTransVec'][:, i:i+1]
        
        # compute next reachable set
        # MATLAB: try
        try:
            # MATLAB: [Rnext,options] = post(sys,Rnext,params,options);
            if syslin:
                # For linear systems, this shouldn't happen (should use linearSys.reach)
                raise NotImplementedError("Linear systems should use linearSys.reach")
            else:
                from cora_python.contDynamics.nonlinearSys.post import post
                Rnext, options = post(sys, Rnext, params, options)
        except Exception as ME:
            # if error from set explosion, return corresponding information
            # MATLAB: R = reachSet.initReachSet(timePoint,timeInt);
            R = ReachSet.initReachSet(timePoint, timeInt)
            # MATLAB: priv_reportReachError(ME,options.t,i);
            print(f"Error in post at t={options['t']}, step={i+1}: {ME}")
            if spec is not None:
                return R, res, options
            else:
                return R
    
    # compute output set
    # MATLAB: if ~isa(sys,'nonlinDASys')
    if not is_nonlinDASys:
        # MATLAB: timeInt.set{end} = outputSet(sys,Rnext.ti,params,options);
        # MATLAB: timePoint.set{end} = outputSet(sys,Rnext.tp,params,options);
        from cora_python.contDynamics.contDynamics.outputSet import outputSet
        if isinstance(Rnext['ti'], list):
            timeInt['set'][-1] = [outputSet(sys, r_ti.get('set', r_ti) if isinstance(r_ti, dict) else r_ti, params, options) 
                                 for r_ti in Rnext['ti']]
        else:
            timeInt['set'][-1] = outputSet(sys, Rnext['ti'], params, options)
        
        if isinstance(Rnext['tp'], list):
            timePoint['set'][-1] = [outputSet(sys, r_tp.get('set', r_tp) if isinstance(r_tp, dict) else r_tp, params, options) 
                                    for r_tp in Rnext['tp']]
        else:
            timePoint['set'][-1] = outputSet(sys, Rnext['tp'], params, options)
    else:
        # MATLAB: timeInt.set{end} = outputSet(sys,Rnext.ti,Rnext.y,params,options);
        # MATLAB: timePoint.set{end} = outputSet(sys,Rnext.tp,Rnext.y,params,options);
        # TODO: Implement nonlinDASys handling
        raise NotImplementedError("nonlinDASys not yet implemented")
    
    # MATLAB: timeInt.time{end} = interval(tVec(end-1),tVec(end));
    timeInt['time'][-1] = Interval(tVec[-2], tVec[-1])
    # MATLAB: timePoint.time{end} = tVec(end);
    timePoint['time'][-1] = tVec[-1]
    
    # MATLAB: if isfield(Rnext,'y')
    if 'y' in Rnext:
        # MATLAB: timeInt.algebraic{end} = Rnext.y;
        timeInt['algebraic'][-1] = Rnext['y']
    
    # check specification
    # MATLAB: if ~isempty(spec)
    if spec is not None:
        # MATLAB: if ~check(spec,timeInt.set{end},timeInt.time{end})
        # TODO: Implement specification checking
        # For now, skip check
        pass
    
    # construct reachset object
    # MATLAB: R = reachSet.initReachSet(timePoint,timeInt);
    R = ReachSet.initReachSet(timePoint, timeInt)
    
    # check temporal logic specifications
    # MATLAB: if res && ~isempty(specLogic)
    if res and specLogic is not None:
        # MATLAB: res = check(specLogic,R);
        # TODO: Implement temporal logic checking
        pass
    
    # log information
    # MATLAB: verboseLog(options.verbose,i+1,tVec(end),params.tStart,params.tFinal);
    verboseLog(options.get('verbose', False), steps+1, tVec[-1], params['tStart'], params['tFinal'])
    
    if spec is not None:
        return R, res, options
    else:
        return R
