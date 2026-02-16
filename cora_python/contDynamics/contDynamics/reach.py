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
        is_adaptive = ('alg' in options and 'adaptive' in str(options['alg']))
        if not is_adaptive:
            if 'timeStep' not in options:
                raise ValueError("options must contain 'timeStep'")
            if 'taylorTerms' not in options:
                raise ValueError("options must contain 'taylorTerms'")
        # Default options/params aligned with MATLAB validateOptions
        options.setdefault('reductionTechnique', 'girard')
        options.setdefault('reductionInterval', np.inf)
        options.setdefault('compOutputSet', True)
        options.setdefault('tensorOrderOutput', 2)
        # Defaults aligned with MATLAB validateOptions/postProcessing
        options.setdefault('tensorOrder', 2)
        options.setdefault('errorOrder', 1)
        options.setdefault('errorOrder3', 1)
        options.setdefault('intermediateOrder', 100)
        if is_adaptive:
            options.setdefault('verbose', False)
            options.setdefault('decrFactor', 0.90)
            options.setdefault('zetaTlin', 0.0005)
            options.setdefault('zetaTabs', 0.005)
            options.setdefault('thirdOrderTensorempty', False)
            options.setdefault('isHessianConst', False)
            options.setdefault('hessianCheck', False)
            if 'lin' in str(options.get('alg', '')):
                from cora_python.contSet.zonotope import Zonotope
                if not isinstance(params['R0'], Zonotope):
                    params['R0'] = Zonotope(params['R0'])
                options.setdefault('redFactor', 0.0005)
                options.setdefault('zetaK', 0.9)
                options.setdefault('zetaphi', [0.85, 0.76, 0.68])
            elif 'poly' in str(options.get('alg', '')):
                from cora_python.contSet.polyZonotope import PolyZonotope
                if not isinstance(params['R0'], PolyZonotope):
                    params['R0'] = PolyZonotope(params['R0'])
                options.setdefault('redFactor', 0.0001)
                options.setdefault('zetaphi', [0.80, 0.75, 0.63])
                options['tensorOrder'] = 3
                options.setdefault('polyZono', {})
                options['polyZono'].setdefault('volApproxMethod', 'interval')
                options['polyZono'].setdefault('maxDepGenOrder', 50)
                options['polyZono'].setdefault('maxPolyZonoRatio', 0.05)
                options['polyZono'].setdefault('restructureTechnique', 'reducePca')
    
    # MATLAB postProcessing (func='reach') for contDynamics: normalize input set U and u/uTrans
    # Ensures non-standard U (e.g. Interval) and params.u are correctly handled before subfunctions
    from cora_python.contSet.zonotope import Zonotope
    if 'U' not in params:
        params['U'] = Zonotope(np.zeros((sys.nr_of_inputs, 1)), np.array([]).reshape(sys.nr_of_inputs, 0))
    if 'U' in params:
        from cora_python.contSet.interval import Interval
        if isinstance(params['U'], Interval):
            params['U'] = Zonotope(params['U'])
    sys_class = getattr(sys, '__class__', None)
    sys_name = sys_class.__name__ if sys_class else ''
    if 'u' in params and 'nonlinearARX' not in sys_name:
        U = params['U']
        centerU = np.asarray(U.center())
        if centerU.ndim == 1:
            centerU = centerU.reshape(-1, 1)
        if np.any(centerU):
            u_val = np.asarray(params['u'])
            if u_val.ndim == 1:
                u_val = u_val.reshape(-1, 1)
            params['u'] = u_val + centerU
            params['U'] = U + (-centerU)
        input_traj_len = params['u'].shape[1] if np.asarray(params['u']).ndim >= 2 else 1
        if input_traj_len > 1:
            params['uTransVec'] = params['u']
        else:
            u_1 = np.asarray(params['u'])
            params['uTrans'] = u_1.reshape(-1, 1) if u_1.ndim == 1 else u_1
        del params['u']
    if 'uTrans' not in params and 'uTransVec' not in params and 'U' in params:
        params['uTrans'] = params['U'].center()
    
    # handling of specifications
    # MATLAB: specLogic = [];
    specLogic = None
    spec_non_logic = spec
    # MATLAB: if ~isempty(spec)
    if spec is not None:
        # MATLAB: [spec,specLogic] = splitLogic(spec);
        if hasattr(spec, 'splitLogic') and callable(getattr(spec, 'splitLogic')):
            spec_non_logic, specLogic = spec.splitLogic(spec)
        elif isinstance(spec, list) and len(spec) > 0 and hasattr(spec[0], 'splitLogic'):
            spec_non_logic, specLogic = spec[0].splitLogic(spec)
        else:
            # fallback to module-level splitLogic
            from cora_python.specification.specification import splitLogic
            spec_non_logic, specLogic = splitLogic(spec)
    
    # reach called from
    # - linear class (linProbSys, linParamSys) or
    # - nonlinear class (nonlinearSys, nonlinDASys, nonlinParamSys)
    # MATLAB: syslin = isa(sys,'linProbSys') || isa(sys,'linParamSys');
    if hasattr(sys, '__class__'):
        sys_class = sys.__class__.__name__.lower()
        # MATLAB: only linear classes (linProbSys, linParamSys, linearSys)
        syslin = sys_class.startswith('lin') and not sys_class.startswith('nonlin')
    else:
        syslin = False
    
    # compute symbolic derivatives
    # MATLAB: if ~syslin
    if not syslin:
        # paramInt required for derivatives of nonlinParamSys...
        # MATLAB: if isfield(params,'paramInt')
        if 'paramInt' in params:
            # MATLAB: options.paramInt = params.paramInt;
            options['paramInt'] = params['paramInt']
        
        # MATLAB: derivatives(sys,options);
        # Use object.method() pattern; method attached in contDynamics.__init__.py
        sys.derivatives(options)
        
        # MATLAB: if contains(options.alg,'adaptive')
        if 'alg' in options and 'adaptive' in str(options['alg']):
            # nonlinear adaptive algorithm
            # MATLAB: [timeInt,timePoint,res,~,options] = reach_adaptive(sys,params,options);
            timeInt, timePoint, res, _, options = sys.reach_adaptive(params, options)
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
    tStart = params['tStart']
    tFinal = params['tFinal']
    timeStep = options['timeStep']
    # Avoid floating overshoot: build with floor, then append tFinal if needed
    steps = int(np.floor((tFinal - tStart) / timeStep + 1e-12))
    tVec = tStart + np.arange(steps + 1) * timeStep
    if tVec.size == 0:
        tVec = np.array([tStart])
    if tVec[-1] < tFinal - 1e-10:
        tVec = np.append(tVec, tFinal)
    elif tVec[-1] > tFinal + 1e-10:
        tVec[-1] = tFinal
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
            # Use object.method() pattern; method attached in contDynamics.__init__.py
            output0 = sys.outputSet(params['R0'], params, options)
            if isinstance(output0, tuple):
                output0 = output0[0]
            timePoint['set'][0] = [{'set': output0,
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
        from cora_python.contDynamics.contDynamics.private.priv_reportReachError import priv_reportReachError
        priv_reportReachError(ME, params['tStart'], 1)
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
            # Handle Rnext.ti and Rnext.tp - they might be lists of dicts
            if isinstance(Rnext['ti'], list):
                timeInt_sets = []
                for r_ti in Rnext['ti']:
                    r_val = r_ti.get('set', r_ti) if isinstance(r_ti, dict) else r_ti
                    out_val = sys.outputSet(r_val, params, options)
                    timeInt_sets.append(out_val[0] if isinstance(out_val, tuple) else out_val)
                timeInt['set'][i-1] = timeInt_sets[0] if len(timeInt_sets) == 1 else timeInt_sets
            else:
                out_val = sys.outputSet(Rnext['ti'], params, options)
                timeInt['set'][i-1] = out_val[0] if isinstance(out_val, tuple) else out_val
            
            if isinstance(Rnext['tp'], list):
                timePoint_sets = []
                for r_tp in Rnext['tp']:
                    r_val = r_tp.get('set', r_tp) if isinstance(r_tp, dict) else r_tp
                    out_val = sys.outputSet(r_val, params, options)
                    timePoint_sets.append(out_val[0] if isinstance(out_val, tuple) else out_val)
                timePoint['set'][i] = timePoint_sets[0] if len(timePoint_sets) == 1 else timePoint_sets
            else:
                out_val = sys.outputSet(Rnext['tp'], params, options)
                timePoint['set'][i] = out_val[0] if isinstance(out_val, tuple) else out_val
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
        if spec_non_logic is not None:
            # MATLAB: if ~check(spec,Rnext.ti,timeInt.time{i-1})
            res_check = _check_spec(spec_non_logic, Rnext['ti'], timeInt['time'][i-1])
            if not res_check:
                res = False
                R = ReachSet.initReachSet(timePoint, timeInt)
                if spec is not None:
                    return R, res, options
                return R
        
        # increment time
        # MATLAB: options.t = tVec(i);
        options['t'] = tVec[i]
        # log information
        # MATLAB: verboseLog(options.verbose,i,options.t,params.tStart,params.tFinal);
        verboseLog(options.get('verbose', False), i+1, options['t'], params['tStart'], params['tFinal'])
        
        # optional progress logging (helps diagnose non-termination)
        if options.get('progress', False):
            interval = int(options.get('progressInterval', 10))
            if i == 1 or (interval > 0 and (i + 1) % interval == 0):
                time_step = options.get('timeStep', np.nan)
                progress_pct = (options['t'] - params['tStart']) / (params['tFinal'] - params['tStart']) * 100
                remaining = params['tFinal'] - options['t']
                print(f"[reach] step={i+1}/{steps} t={options['t']:.6g} dt={time_step:.6g} "
                      f"({progress_pct:.1f}% complete, {remaining:.6g} remaining)", flush=True)
        
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
            from cora_python.contDynamics.contDynamics.private.priv_reportReachError import priv_reportReachError
            priv_reportReachError(ME, options['t'], i + 1)
            if spec is not None:
                return R, res, options
            else:
                return R
    
    # compute output set
    # MATLAB: if ~isa(sys,'nonlinDASys')
    if not is_nonlinDASys:
        # MATLAB: timeInt.set{end} = outputSet(sys,Rnext.ti,params,options);
        # MATLAB: timePoint.set{end} = outputSet(sys,Rnext.tp,params,options);
        if isinstance(Rnext['ti'], list):
            timeInt_sets = []
            for r_ti in Rnext['ti']:
                r_val = r_ti.get('set', r_ti) if isinstance(r_ti, dict) else r_ti
                out_val = sys.outputSet(r_val, params, options)
                timeInt_sets.append(out_val[0] if isinstance(out_val, tuple) else out_val)
            timeInt['set'][-1] = timeInt_sets[0] if len(timeInt_sets) == 1 else timeInt_sets
        else:
            out_val = sys.outputSet(Rnext['ti'], params, options)
            timeInt['set'][-1] = out_val[0] if isinstance(out_val, tuple) else out_val
        
        if isinstance(Rnext['tp'], list):
            timePoint_sets = []
            for r_tp in Rnext['tp']:
                r_val = r_tp.get('set', r_tp) if isinstance(r_tp, dict) else r_tp
                out_val = sys.outputSet(r_val, params, options)
                timePoint_sets.append(out_val[0] if isinstance(out_val, tuple) else out_val)
            timePoint['set'][-1] = timePoint_sets[0] if len(timePoint_sets) == 1 else timePoint_sets
        else:
            out_val = sys.outputSet(Rnext['tp'], params, options)
            timePoint['set'][-1] = out_val[0] if isinstance(out_val, tuple) else out_val
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
    if spec_non_logic is not None:
        # MATLAB: if ~check(spec,timeInt.set{end},timeInt.time{end})
        res_check = _check_spec(spec_non_logic, timeInt['set'][-1], timeInt['time'][-1])
        if not res_check:
            res = False
    
    # construct reachset object
    # MATLAB: R = reachSet.initReachSet(timePoint,timeInt);
    R = ReachSet.initReachSet(timePoint, timeInt)
    
    # check temporal logic specifications
    # MATLAB: if res && ~isempty(specLogic)
    if res and specLogic is not None:
        # MATLAB: res = check(specLogic,R);
        res = _check_spec(specLogic, R, None)
    
    # log information
    # MATLAB: verboseLog(options.verbose,i+1,tVec(end),params.tStart,params.tFinal);
    verboseLog(options.get('verbose', False), steps+1, tVec[-1], params['tStart'], params['tFinal'])
    
    if spec is not None:
        return R, res, options
    else:
        return R


def _check_spec(spec, S, time):
    if spec is None:
        return True
    if hasattr(spec, 'check') and callable(getattr(spec, 'check')):
        res = spec.check(S, time) if time is not None else spec.check(S)
    elif isinstance(spec, list) and len(spec) > 0 and hasattr(spec[0], 'check'):
        res = spec[0].check(spec, S, time) if time is not None else spec[0].check(spec, S)
    else:
        from cora_python.specification.specification import check
        res = check(spec, S, time) if time is not None else check(spec, S)
    if isinstance(res, tuple):
        return res[0]
    return res
