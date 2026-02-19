"""
linReach - computes the reachable set after linearization

Syntax:
    [Rti,Rtp,dimForSplit,options] = linReach(sys,Rstart,params,options)

Inputs:
    sys - nonlinearSys or nonlinParamSys object
    Rstart - initial reachable set
    params - model parameters
    options - struct with algorithm settings

Outputs:
    Rti - reachable set for time interval
    Rtp - reachable set for time point
    dimForSplit - dimension that is split to reduce the lin. error
    options - struct with algorithm settings

References: 
  [1] M. Althoff et al. "Reachability analysis of nonlinear systems with 
      uncertain parameters using conservative linearization"
  [2] M. Althoff. "Reachability analysis of nonlinear systems using 
      conservative polynomialization and non-convex sets"

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: initReach, post

Authors:       Matthias Althoff, Niklas Kochdumper, Mark Wetzlinger
Written:       17-January-2008
Last update:   29-June-2009
               23-July-2009
               10-July-2012
               18-September-2012
               09-August-2016
               12-September-2017
               02-January-2020 (NK, restructured the function)
               22-April-2020 (MW, simplification)
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import Tuple, Any, Dict, Optional, List
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.interval import Interval
from cora_python.contDynamics.contDynamics.private.priv_abstrerr_lin import priv_abstrerr_lin
from cora_python.contDynamics.contDynamics.private.priv_abstrerr_poly import priv_abstrerr_poly
from cora_python.contDynamics.contDynamics.private.priv_precompStatError import priv_precompStatError
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

# Import polyZonotope and conPolyZono for type checking
from cora_python.contSet.polyZonotope import PolyZonotope
from cora_python.contSet.conPolyZono import ConPolyZono



def linReach(sys: Any, Rstart: Dict[str, Any], params: Dict[str, Any], 
             options: Dict[str, Any]) -> Tuple[Any, Any, Optional[int], Dict[str, Any]]:
    """
    Computes the reachable set after linearization
    
    Args:
        sys: nonlinearSys or nonlinParamSys object
        Rstart: initial reachable set (dict with 'set' and 'error' keys)
        params: model parameters
        options: struct with algorithm settings
        
    Returns:
        Rti: reachable set for time interval
        Rtp: reachable set for time point (dict with 'set' and 'error' keys)
        dimForSplit: dimension that is split to reduce the lin. error (None if no splitting)
        options: struct with algorithm settings
    """
    
    # extract initial set and abstraction error
    # MATLAB: Rinit = Rstart.set;
    Rinit = Rstart['set']
    
    # MATLAB: abstrerr = Rstart.error;
    abstrerr = Rstart['error']
    
    # necessary to update part of abstraction that is dependent on x0 when
    # linearization remainder is not computed
    # MATLAB: if isfield(options,'updateInitFnc')
    if 'updateInitFnc' in options:
        # MATLAB: currentStep = round((options.t-params.tStart)/options.timeStep)+1;
        currentStep = int(round((options['t'] - params['tStart']) / options['timeStep']) + 1)
        # MATLAB: Rinit = options.updateInitFnc(Rinit,currentStep);
        Rinit = options['updateInitFnc'](Rinit, currentStep)
    
    # linearize the nonlinear system
    # MATLAB: [sys,linsys,linParams,linOptions] = linearize(sys,Rinit,params,options);
    # linearize is implemented in cora_python.contDynamics.nonlinearSys.linearize
    from cora_python.contDynamics.nonlinearSys.linearize import linearize
    sys, linsys, linParams, linOptions = linearize(sys, Rinit, params, options)
    
    # translate Rinit by linearization point
    # MATLAB: Rdelta = Rinit + (-sys.linError.p.x);
    Rdelta = Rinit + (-sys.linError.p.x)
    import os
    if os.getenv('CORA_DEBUG_LINREACH') == '1':
        try:
            t_now = options.get('t')
            if t_now is not None and abs(t_now - 312.0) < 1e-9:
                Rdelta_int = Rdelta.interval()
                print("DEBUG Rdelta.inf =", Rdelta_int.infimum().reshape(-1))
                print("DEBUG Rdelta.sup =", Rdelta_int.supremum().reshape(-1))
        except Exception:
            pass
    # Debug Rdelta interval for MATLAB comparison
    import os
    if os.getenv('CORA_DEBUG_LINREACH') == '1' and options.get('t') in (4.0, 8.0, 12.0, 16.0, 308.0, 312.0):
        try:
            Rdelta_int = Rdelta.interval()
            print("DEBUG Rdelta t =", options.get('t'))
            print("DEBUG Rdelta.inf =", Rdelta_int.infimum().reshape(-1))
            print("DEBUG Rdelta.sup =", Rdelta_int.supremum().reshape(-1))
        except Exception:
            pass
    
    # compute reachable set of the linearized system
    # MATLAB: if isa(sys,'nonlinParamSys') && isa(params.paramInt,'interval')
    is_nonlinParamSys = hasattr(sys, '__class__') and 'nonlinParamSys' in sys.__class__.__name__.lower()
    is_paramInt_interval = ('paramInt' in params and 
                           hasattr(params['paramInt'], '__class__') and 
                           'interval' in params['paramInt'].__class__.__name__.lower())
    
    if is_nonlinParamSys and is_paramInt_interval:
        # MATLAB: [linsys,R] = initReach_inputDependence(linsys,Rdelta,linParams,linOptions);
        from cora_python.contDynamics.linearParamSys.initReach_inputDependence import initReach_inputDependence
        linsys, R, _ = initReach_inputDependence(linsys, Rdelta, linParams, linOptions)
        Rtp = R['tp']
        Rti = R['ti']
    # MATLAB: elseif isa(linsys,'linParamSys')
    elif _is_linear_param_sys(linsys):
        # MATLAB: R = initReach(linsys,Rdelta,linParams,linOptions);
        # NOTE: linParamSys.initReach needs to be translated
        from cora_python.contDynamics.linearParamSys.initReach import initReach as linParamSys_initReach
        R = linParamSys_initReach(linsys, Rdelta, linParams, linOptions)
        Rtp = R['tp']
        Rti = R['ti']
    else:
        # MATLAB: [Rtp,Rti,~,~,PU,Pu,~,C_input] = oneStep(linsys,Rdelta,...
        #        linParams.U,linParams.uTrans,options.timeStep,options.taylorTerms);
        from cora_python.contDynamics.linearSys.oneStep import oneStep
        Rtp, Rti, Htp, Hti, PU, Pu, C_state, C_input = oneStep(
            linsys, Rdelta, linParams['U'], linParams['uTrans'], 
            options['timeStep'], options['taylorTerms']
        )
        # Debug oneStep components for MATLAB comparison
        import os
        if os.getenv('CORA_DEBUG_LINREACH') == '1' and options.get('t') == 312.0:
            def _interval_bounds(S):
                try:
                    IH = S.interval()
                    return IH.infimum().reshape(-1), IH.supremum().reshape(-1)
                except Exception:
                    return None, None
            try:
                from cora_python.contDynamics.linearSys.private.priv_correctionMatrixState import priv_correctionMatrixState
                F_dbg = priv_correctionMatrixState(linsys, options['timeStep'], options['taylorTerms'])
                print("DEBUG oneStep F.inf =", F_dbg.int.inf.reshape(-1))
                print("DEBUG oneStep F.sup =", F_dbg.int.sup.reshape(-1))
            except Exception as exc:
                print("DEBUG oneStep F.error =", repr(exc))
            Hti_inf, Hti_sup = _interval_bounds(Hti)
            PU_inf, PU_sup = _interval_bounds(PU)
            Pu_inf, Pu_sup = _interval_bounds(Pu)
            Cstate_inf, Cstate_sup = _interval_bounds(C_state)
            Cinput_inf, Cinput_sup = _interval_bounds(C_input)
            if Hti_inf is not None:
                print("DEBUG oneStep Hti.inf =", Hti_inf)
                print("DEBUG oneStep Hti.sup =", Hti_sup)
            if PU_inf is not None:
                print("DEBUG oneStep PU.inf =", PU_inf)
                print("DEBUG oneStep PU.sup =", PU_sup)
            if Pu_inf is not None:
                print("DEBUG oneStep Pu.inf =", Pu_inf)
                print("DEBUG oneStep Pu.sup =", Pu_sup)
            if Cstate_inf is not None:
                print("DEBUG oneStep C_state.inf =", Cstate_inf)
                print("DEBUG oneStep C_state.sup =", Cstate_sup)
            if Cinput_inf is not None:
                print("DEBUG oneStep C_input.inf =", Cinput_inf)
                print("DEBUG oneStep C_input.sup =", Cinput_sup)
        
        # MATLAB: if strcmp(options.alg,'poly')
        if options['alg'] == 'poly':
            # pre-compute set of state differences
            # MATLAB: Rdiff = aux_deltaReach(linsys,Rdelta,PU,Pu,C_input,...
            #    options.timeStep,options.taylorTerms,...
            #    options.reductionTechnique,options.intermediateOrder);
            Rdiff = aux_deltaReach(linsys, Rdelta, PU, Pu, C_input,
                                  options['timeStep'], options['taylorTerms'],
                                  options['reductionTechnique'], options['intermediateOrder'])
            
            # pre-compute static abstraction error
            # MATLAB: if options.tensorOrder > 2
            if options['tensorOrder'] > 2:
                # MATLAB: [H,Zdelta,errorStat,T,ind3,Zdelta3] = ...
                #            priv_precompStatError(sys,Rdelta,params,options);
                H, Zdelta, errorStat, T, ind3, Zdelta3 = priv_precompStatError(sys, Rdelta, params, options)
            else:
                H = None
                Zdelta = None
                errorStat = None
                T = None
                ind3 = None
                Zdelta3 = None
        else:
            Rdiff = None
            H = None
            Zdelta = None
            errorStat = None
            T = None
            ind3 = None
            Zdelta3 = None
    
    # MATLAB: if isfield(options,'approxDepOnly') && options.approxDepOnly
    if 'approxDepOnly' in options and options['approxDepOnly']:
        # MATLAB: if ~exist('errorStat','var')
        if errorStat is None:
            errorStat = None
        # MATLAB: R.tp = Rtp; R.ti = Rti;
        R = {'tp': Rtp, 'ti': Rti}
        # MATLAB: [Rtp,Rti,dimForSplit,options] = aux_approxDepReachOnly(linsys,sys,R,options,errorStat);
        Rtp, Rti, dimForSplit, options = aux_approxDepReachOnly(linsys, sys, R, options, errorStat)
        return Rti, Rtp, dimForSplit, options
    
    # compute reachable set of the abstracted system including the
    # abstraction error using the selected algorithm
    # MATLAB: if strcmp(options.alg,'linRem')
    if options['alg'] == 'linRem':
        # MATLAB: [Rtp,Rti,perfInd] = aux_linReach_linRem(sys,R,Rinit,Rdelta,params,options);
        R = {'tp': Rtp, 'ti': Rti}
        Rtp, Rti, perfInd = aux_linReach_linRem(sys, R, Rinit, Rdelta, params, options)
    else:
        
        # loop until the actual abstraction error is smaller than the 
        # estimated linearization error
        # MATLAB: perfIndCurr = Inf; perfInd = 0;
        perfIndCurr = np.inf
        perfInd = 0
        
        # used in AROC for reachsetOptimalControl (reachSet with previous
        # linearization error)
        # MATLAB: if isfield(options,'prevErr')
        if 'prevErr' in options:
            # MATLAB: perfIndCurr = 1;
            perfIndCurr = 1
            # MATLAB: if isfield(options,'prevErrScale')
            if 'prevErrScale' in options:
                # MATLAB: scale = options.prevErrScale;
                scale = options['prevErrScale']
            else:
                # MATLAB: scale = 0.8;
                scale = 0.8
            # MATLAB: Rerror = scale*options.prevErr;
            Rerror = scale * options['prevErr']
        else:
            Rerror = None
        
        # MATLAB: while perfIndCurr > 1 && perfInd <= 1
        while perfIndCurr > 1 and perfInd <= 1:
            # estimate the abstraction error 
            # MATLAB: appliedError = 1.1*abstrerr;
            appliedError = 1.1 * abstrerr
            
            # MATLAB: Verror = zonotope(0*appliedError,diag(appliedError));
            Verror = Zonotope(np.zeros_like(appliedError), np.diag(appliedError.flatten()))
            
            # MATLAB: if isa(linsys,'linParamSys')
            is_linParamSys = hasattr(linsys, '__class__') and 'linParamSys' in linsys.__class__.__name__.lower()
            
            if is_linParamSys:
                # MATLAB: RallError = errorSolution(linsys,options,Verror);
                # NOTE: errorSolution needs to be translated
                from cora_python.contDynamics.linearParamSys.errorSolution import errorSolution
                RallError = errorSolution(linsys, options, Verror)
            else:
                # MATLAB: RallError = particularSolution_timeVarying(linsys,...
                #    Verror,options.timeStep,options.taylorTerms);
                from cora_python.contDynamics.linearSys.particularSolution_timeVarying import particularSolution_timeVarying
                RallError = particularSolution_timeVarying(linsys, Verror, options['timeStep'], options['taylorTerms'])
            
            # compute the abstraction error using the conservative
            # linearization approach described in [1]
            # MATLAB: if strcmp(options.alg,'lin')
            if options['alg'] == 'lin':
                
                # compute overall reachable set including linearization error
                # MATLAB: Rmax = Rti+RallError;
                Rmax = Rti + RallError
                # Debug logging for interval growth (controlled via env var)
                import os
                if os.getenv('CORA_DEBUG_LINREACH') == '1':
                    try:
                        t_now = options.get('t')
                        Rti_int = Rti.interval()
                        Rall_int = RallError.interval()
                        Rmax_int = Rmax.interval()
                        print("DEBUG linReach t =", t_now)
                        print("DEBUG p.x =", sys.linError.p.x.reshape(-1))
                        print("DEBUG Rti.inf =", Rti_int.infimum().reshape(-1))
                        print("DEBUG Rti.sup =", Rti_int.supremum().reshape(-1))
                        print("DEBUG RallError.inf =", Rall_int.infimum().reshape(-1))
                        print("DEBUG RallError.sup =", Rall_int.supremum().reshape(-1))
                        print("DEBUG Rmax.inf =", Rmax_int.infimum().reshape(-1))
                        print("DEBUG Rmax.sup =", Rmax_int.supremum().reshape(-1))
                        if t_now is not None and abs(t_now - 312.0) < 1e-9:
                            print("DEBUG maxError =", options.get('maxError'))
                            try:
                                F_dbg = None
                                if hasattr(linsys, 'taylor') and hasattr(linsys.taylor, '_F_cache'):
                                    F_dbg = linsys.taylor._F_cache.get(options['timeStep'])
                                if F_dbg is not None and hasattr(F_dbg, 'inf') and hasattr(F_dbg, 'sup'):
                                    print("DEBUG oneStep F.inf =", F_dbg.inf.reshape(-1))
                                    print("DEBUG oneStep F.sup =", F_dbg.sup.reshape(-1))
                            except Exception:
                                pass
                    except Exception:
                        pass
                
                # compute linearization error
                # MATLAB: [trueError,VerrorDyn] = priv_abstrerr_lin(sys,Rmax,params,options);
                trueError, VerrorDyn = priv_abstrerr_lin(sys, Rmax, params, options)
                if os.getenv('CORA_DEBUG_LINREACH') == '1':
                    try:
                        print("DEBUG appliedError =", appliedError.reshape(-1))
                        print("DEBUG trueError =", trueError.reshape(-1))
                    except Exception:
                        pass
                
                # MATLAB: VerrorStat = zeros(sys.nrOfDims,1);
                VerrorStat = np.zeros((sys.nr_of_dims, 1))
                
            # compute the abstraction error using the conservative
            # polynomialization approach described in [2]    
            else:
                
                # compute overall reachable set including linearization error
                # MATLAB: Rmax = Rdelta+zonotope(Rdiff)+RallError;
                Rdiff_zono = Zonotope(Rdiff) if not isinstance(Rdiff, Zonotope) else Rdiff
                Rmax = Rdelta + Rdiff_zono + RallError
                
                # compute abstraction error
                # MATLAB: [trueError,VerrorDyn,VerrorStat] = ...
                #            priv_abstrerr_poly(sys,Rmax,Rdiff+RallError,params,options, ...
                #                                H,Zdelta,errorStat,T,ind3,Zdelta3);
                Rdiff_RallError = Rdiff + RallError if Rdiff is not None else RallError
                trueError, VerrorDyn, VerrorStat = priv_abstrerr_poly(
                    sys, Rmax, Rdiff_RallError, params, options,
                    H, Zdelta, errorStat, T, ind3, Zdelta3
                )
            
            # compare linearization error with the maximum allowed error
            # MATLAB: perfIndCurr = max(trueError./appliedError);
            with np.errstate(divide='ignore', invalid='ignore'):
                perfIndCurr = np.max(trueError / appliedError)
            if os.getenv('CORA_DEBUG_LINREACH') == '1':
                try:
                    t_now = options.get('t')
                    if t_now is not None and abs(t_now - 312.0) < 1e-9:
                        print("DEBUG perfIndCurr =", perfIndCurr)
                except Exception:
                    pass
            
            # MATLAB: perfInd = max(trueError./options.maxError);
            with np.errstate(divide='ignore', invalid='ignore'):
                perfInd = np.max(trueError / options['maxError'])
            
            debug_split = (options.get('debugSplit', False) or os.getenv('CORA_DEBUG_SPLIT') == '1')
            if debug_split:
                try:
                    t_now = options.get('t', None)
                    debug_max_t = options.get('debugSplitMaxT', None)
                    if debug_max_t is None:
                        debug_max_t_env = os.getenv('CORA_DEBUG_SPLIT_MAX_T')
                        debug_max_t = float(debug_max_t_env) if debug_max_t_env else None
                    if debug_max_t is None or (t_now is not None and t_now <= debug_max_t):
                        print("DEBUG split: t =", t_now)
                        print("DEBUG split: appliedError =", np.asarray(appliedError).reshape(-1))
                        print("DEBUG split: trueError =", np.asarray(trueError).reshape(-1))
                        print("DEBUG split: maxError =", np.asarray(options['maxError']).reshape(-1))
                        print("DEBUG split: perfIndCurr =", perfIndCurr, "perfInd =", perfInd, flush=True)
                except Exception:
                    pass
            if os.getenv('CORA_DEBUG_LINREACH') == '1':
                try:
                    t_now = options.get('t')
                    if t_now is not None and abs(t_now - 312.0) < 1e-9:
                        print("DEBUG perfInd =", perfInd)
                except Exception:
                    pass
            
            # MATLAB: abstrerr = trueError;
            abstrerr = trueError
            
            # clean exit in case of set explosion
            # MATLAB: if any(abstrerr > 1e+100)
            if np.any(abstrerr > 1e+100):
                raise CORAerror('CORA:reachSetExplosion')
        
        # translate reachable sets by linearization point
        # MATLAB: Rti = Rti+sys.linError.p.x;
        Rti = Rti + sys.linError.p.x
        
        # MATLAB: Rtp = Rtp+sys.linError.p.x;
        Rtp = Rtp + sys.linError.p.x
        
        # compute the reachable set due to the linearization error
        # MATLAB: if ~exist('Rerror','var')
        if Rerror is None:
            # MATLAB: if isa(linsys,'linParamSys')
            if is_linParamSys:
                # MATLAB: Rerror = errorSolution(linsys,options,VerrorDyn);
                from cora_python.contDynamics.linearParamSys.errorSolution import errorSolution
                Rerror = errorSolution(linsys, options, VerrorDyn)
            else:
                # MATLAB: Rerror_dyn = particularSolution_timeVarying(linsys,...
                #    VerrorDyn,options.timeStep,options.taylorTerms);
                from cora_python.contDynamics.linearSys.particularSolution_timeVarying import particularSolution_timeVarying
                Rerror_dyn = particularSolution_timeVarying(linsys, VerrorDyn, options['timeStep'], options['taylorTerms'])
                
                # MATLAB: Rerror_stat = particularSolution_constant(linsys,...
                #    VerrorStat,options.timeStep,options.taylorTerms);
                from cora_python.contDynamics.linearSys.particularSolution_constant import particularSolution_constant
                Rerror_stat, _, _ = particularSolution_constant(linsys, VerrorStat, options['timeStep'], options['taylorTerms'])
                # Ensure Rerror_stat is a zonotope (not a vector)
                if not isinstance(Rerror_stat, Zonotope):
                    Rerror_stat = Zonotope(Rerror_stat) if isinstance(Rerror_stat, np.ndarray) else Rerror_stat
                
                # MATLAB: Rerror = Rerror_dyn + Rerror_stat;
                Rerror = Rerror_dyn + Rerror_stat
            
            # MATLAB: if isfield(options,'approxErr') && options.approxErr
            if 'approxErr' in options and options['approxErr']:
                # MATLAB: options.prevErr = Rerror;
                options['prevErr'] = Rerror
        
        # add the abstraction error to the reachable sets
        # MATLAB: if strcmp(options.alg,'poly') && (isa(Rerror,'polyZonotope') || ...
        #                                      isa(Rerror,'conPolyZono'))
        # Check if Rerror is polyZonotope or conPolyZono
        is_poly_or_conpoly = False
        if PolyZonotope is not None:
            is_poly_or_conpoly = isinstance(Rerror, PolyZonotope)
        if not is_poly_or_conpoly and ConPolyZono is not None:
            is_poly_or_conpoly = isinstance(Rerror, ConPolyZono)
        
        if options['alg'] == 'poly' and is_poly_or_conpoly:
            # MATLAB: Rti=exactPlus(Rti,Rerror);
            Rti = Rti.exactPlus(Rerror)
            
            # MATLAB: Rtp=exactPlus(Rtp,Rerror);
            Rtp = Rtp.exactPlus(Rerror)
        else:
            # MATLAB: Rti=Rti+Rerror;
            Rti = Rti + Rerror
            
            # MATLAB: Rtp=Rtp+Rerror;
            Rtp = Rtp + Rerror
    
    # determine the best dimension to split the set in order to reduce the
    # linearization error
    # MATLAB: dimForSplit = [];
    dimForSplit = None
    
    # MATLAB: if perfInd > 1
    if perfInd > 1:
        # MATLAB: dimForSplit = priv_select(sys,Rstart,params,options);
        # NOTE: priv_select needs to be translated
        from cora_python.contDynamics.contDynamics.private.priv_select import priv_select
        dimForSplit = priv_select(sys, Rstart, params, options)
    
    # store the linearization error
    # MATLAB: Rtp_.set = Rtp;
    # MATLAB: Rtp_.error = abstrerr;
    # MATLAB: Rtp = Rtp_;
    Rtp_dict = {'set': Rtp, 'error': abstrerr}
    
    return Rti, Rtp_dict, dimForSplit, options


# Auxiliary functions -----------------------------------------------------

def aux_deltaReach(sys: Any, Rinit: Any, RV: Any, Rtrans: Any, inputCorr: Any,
                   timeStep: float, truncationOrder: int, reductionTechnique: str,
                   intermediateOrder: int) -> Any:
    """
    Compute delta reachable set (set of states differences): the notable
    different to linearSys functions is that we have (e^At - I)*Rinit and an
    enclose-call with the origin
    """
    
    # compute/read out helper variables
    # MATLAB: options = struct('timeStep',timeStep,'ithpower',truncationOrder);
    # Note: ithpower is not used for eAdt, only timeStep is needed
    # MATLAB: eAt = getTaylor(sys,'eAdt',options);
    # getTaylor is a method on linearSys objects
    if hasattr(sys, 'getTaylor'):
        eAt = sys.getTaylor('eAdt', timeStep=timeStep)
    elif hasattr(sys, 'taylor') and hasattr(sys.taylor, 'getTaylor'):
        eAt = sys.taylor.getTaylor('eAdt', timeStep=timeStep)
    else:
        raise AttributeError("System does not have getTaylor method")
    
    # MATLAB: F = readFieldForTimeStep(sys.taylor,'F',timeStep);
    # readFieldForTimeStep is a method on taylor objects
    if hasattr(sys, 'taylor') and hasattr(sys.taylor, 'readFieldForTimeStep'):
        F = sys.taylor.readFieldForTimeStep('F', timeStep)
        # If F is not computed yet, compute it using priv_correctionMatrixState
        if F is None:
            from cora_python.contDynamics.linearSys.private.priv_correctionMatrixState import priv_correctionMatrixState
            # F needs truncationOrder, which is passed to aux_deltaReach
            # For now, use a default truncationOrder (will be computed by priv_correctionMatrixState)
            F = priv_correctionMatrixState(sys, timeStep, float('inf'))
    else:
        raise AttributeError("System does not have readFieldForTimeStep method")
    
    # first time step homogeneous solution
    # MATLAB: n = sys.nrOfDims;
    n = sys.nr_of_dims
    
    # MATLAB: Rhom_tp_delta = (eAt - eye(n))*Rinit + Rtrans;
    eye_n = np.eye(n)
    Rhom_tp_delta = (eAt - eye_n) @ Rinit + Rtrans
    
    # MATLAB: if isa(Rinit,'zonotope')
    if isinstance(Rinit, Zonotope):
        # original computation
        # MATLAB: O = zonotope.origin(n);
        O = Zonotope.origin(n)
        
        # MATLAB: Rhom=enclose(O,Rhom_tp_delta)+F*Rinit+inputCorr;
        Rhom = O.enclose(Rhom_tp_delta) + F @ Rinit + inputCorr
    # MATLAB: elseif isa(Rinit,'polyZonotope') || isa(Rinit,'conPolyZono')
    elif (PolyZonotope is not None and isinstance(Rinit, PolyZonotope)) or \
         (ConPolyZono is not None and isinstance(Rinit, ConPolyZono)):
        # MATLAB: O = zeros(n)*Rhom_tp_delta;  % to retain dependencies!
        O = np.zeros((n, n)) @ Rhom_tp_delta  # to retain dependencies!
        
        # MATLAB: Rhom=enclose(O,Rhom_tp_delta)+F*zonotope(Rinit)+inputCorr;
        Rinit_zono = Zonotope(Rinit)
        Rhom = O.enclose(Rhom_tp_delta) + F @ Rinit_zono + inputCorr
    # MATLAB: elseif isa(Rinit,'zonoBundle')
    elif hasattr(Rinit, '__class__') and 'zonoBundle' in Rinit.__class__.__name__.lower():
        # MATLAB: O = zonoBundle.origin(n);
        from cora_python.contSet.zonoBundle import ZonoBundle
        O = ZonoBundle.origin(n)
        
        # MATLAB: Rhom=enclose(O,Rhom_tp_delta)+F*Rinit.Z{1}+inputCorr;
        Rhom = O.enclose(Rhom_tp_delta) + F @ Rinit.Z[0] + inputCorr
    else:
        Rhom = Rhom_tp_delta
    
    # reduce zonotope
    # MATLAB: Rhom=reduce(Rhom,reductionTechnique,intermediateOrder);
    Rhom = Rhom.reduce(reductionTechnique, intermediateOrder)
    
    # MATLAB: if ~isnumeric(RV)
    if not isinstance(RV, (int, float, np.number, np.ndarray)):
        # MATLAB: RV=reduce(RV,reductionTechnique,intermediateOrder);
        RV = RV.reduce(reductionTechnique, intermediateOrder)
    
    # total solution
    # MATLAB: if isa(Rinit,'polytope')
    if hasattr(Rinit, '__class__') and 'polytope' in Rinit.__class__.__name__.lower():
        # convert zonotopes to polytopes
        # MATLAB: Radd=polytope(RV);
        from cora_python.contSet.polytope import Polytope
        Radd = Polytope(RV)
        
        # MATLAB: Rdelta=Rhom+Radd;
        Rdelta = Rhom + Radd
    else:
        # original computation
        # MATLAB: Rdelta=Rhom+RV;
        Rdelta = Rhom + RV
    
    return Rdelta


def aux_linReach_linRem(sys: Any, R: Dict[str, Any], Rinit: Any, Rdelta: Any,
                       params: Dict[str, Any], options: Dict[str, Any]) -> Tuple[Any, Any, float]:
    """
    Compute the reachable set for the linearized system using an algorithm
    that is based on the linearization of the Lagrange remainder
    """
    
    # compute the reachable set for the linearized system
    # MATLAB: options.alg = 'lin';
    options_lin = options.copy()
    options_lin['alg'] = 'lin'
    
    # MATLAB: [sys,linsys,linParams,linOptions] = linearize(sys,Rinit,params,options);
    from cora_python.contDynamics.nonlinearSys.linearize import linearize
    sys, linsys, linParams, linOptions = linearize(sys, Rinit, params, options_lin)
    
    # MATLAB: if isa(linsys,'linParamSys')
    is_linParamSys = hasattr(linsys, '__class__') and 'linParamSys' in linsys.__class__.__name__.lower()
    
    if is_linParamSys:
        # MATLAB: linOptions.compTimePoint = true;
        linOptions['compTimePoint'] = True
    
    # MATLAB: if isa(sys,'nonlinParamSys') && isa(params.paramInt,'interval')
    is_nonlinParamSys = hasattr(sys, '__class__') and 'nonlinParamSys' in sys.__class__.__name__.lower()
    is_paramInt_interval = ('paramInt' in params and 
                           hasattr(params['paramInt'], '__class__') and 
                           'interval' in params['paramInt'].__class__.__name__.lower())
    
    if is_nonlinParamSys and is_paramInt_interval:
        # MATLAB: [~,Rlin] = initReach_inputDependence(linsys,Rdelta,linParams,linOptions);
        from cora_python.contDynamics.linearParamSys.initReach_inputDependence import initReach_inputDependence
        _, Rlin, _ = initReach_inputDependence(linsys, Rdelta, linParams, linOptions)
    elif is_linParamSys:
        # MATLAB: Rlin = initReach(linsys,Rdelta,linParams,linOptions);
        from cora_python.contDynamics.linearParamSys.initReach import initReach as linParamSys_initReach
        Rlin = linParamSys_initReach(linsys, Rdelta, linParams, linOptions)
        # MATLAB: Ro_int = interval(Rlin.ti);
        Ro_int = Rlin['ti'].interval()
    # MATLAB: elseif isa(linsys,'linearSys')
    elif hasattr(linsys, '__class__') and 'linearSys' in linsys.__class__.__name__.lower():
        # MATLAB: Rlinti = oneStep(linsys,Rdelta,linParams.U,...
        #    linParams.uTrans,options.timeStep,options.taylorTerms);
        from cora_python.contDynamics.linearSys.oneStep import oneStep
        Rlinti, _, _, _, _, _, _, _ = oneStep(
            linsys, Rdelta, linParams['U'], linParams['uTrans'],
            options['timeStep'], options['taylorTerms']
        )
        # MATLAB: Ro_int = interval(Rlinti);
        Ro_int = Rlinti.interval()
    
    # compare the computed reachable set to the reachable set of the
    # linearized system in order to decide if splitting is required
    # MATLAB: Rti_int = interval(R.ti);
    Rti_int = R['ti'].interval()
    
    # MATLAB: trueError = max(abs(Rti_int.inf-Ro_int.inf),abs(Rti_int.sup-Ro_int.sup));
    trueError = np.maximum(np.abs(Rti_int.inf - Ro_int.inf), np.abs(Rti_int.sup - Ro_int.sup))
    perfInd = np.max(trueError / options['maxError'])
    
    # translate reachable sets by linearization point
    # MATLAB: Rti = R.ti + sys.linError.p.x;
    Rti = R['ti'] + sys.linError.p.x
    
    # MATLAB: Rtp = R.tp + sys.linError.p.x;
    Rtp = R['tp'] + sys.linError.p.x
    
    return Rtp, Rti, perfInd


def aux_approxDepReachOnly(linsys: Any, nlnsys: Any, R: Dict[str, Any],
                           options: Dict[str, Any], errorStat: Any) -> Tuple[Any, Any, Optional[int], Dict[str, Any]]:
    """
    Computes an approximation of the reachable set for controller synthesis.
    Compared to the over-approximative reachability algorithm, the
    higher-order terms are only evaluated for the time-point reachable set
    (errorStat) and the Lagrange remainder is neglected.
    """
    
    # read tp and ti
    # MATLAB: R_tp = R.tp;
    R_tp = R['tp']
    
    # MATLAB: R_ti = R.ti;
    R_ti = R['ti']
    
    # MATLAB: if representsa_(errorStat,'emptySet',eps) || all(representsa_(errorStat,'origin',eps))
    from cora_python.contSet.zonotope.representsa_ import representsa_
    eps = np.finfo(float).eps
    
    if errorStat is None or representsa_(errorStat, 'emptySet', eps) or np.all(representsa_(errorStat, 'origin', eps)):
        # we do not need to consider errorStat then
        # MATLAB: R_tp = R_tp + nlnsys.linError.p.x;
        R_tp = R_tp + nlnsys.linError.p.x
        
        # MATLAB: R_ti = R_ti + nlnsys.linError.p.x;
        R_ti = R_ti + nlnsys.linError.p.x
    else:
        # consider errorStat
        # MATLAB: [id,~,ind] = unique(R_ti.id); E = zeros(length(id),size(R_ti.E,2));
        # NOTE: This is for polyZonotope - needs polyZonotope implementation
        from cora_python.contSet.polyZonotope import PolyZonotope
        if isinstance(R_ti, PolyZonotope):
            id_unique, ind = np.unique(R_ti.id, return_inverse=True)
            E = np.zeros((len(id_unique), R_ti.E.shape[1]))
            for i in range(len(ind)):
                E[ind[i], :] = E[ind[i], :] + R_ti.E[i, :]
            # MATLAB: R_ti = polyZonotope(R_ti.c,R_ti.G,R_ti.GI,E,id);
            R_ti = PolyZonotope(R_ti.c, R_ti.G, R_ti.GI, E, id_unique)
        
        # MATLAB: Asum = options.timeStep*eye(linsys.nrOfDims);
        Asum = options['timeStep'] * np.eye(linsys.nr_of_dims)
        
        # MATLAB: for i = 1:options.taylorTerms
        for i in range(options['taylorTerms']):
            # MATLAB: Asum = Asum + linsys.taylor.Apower{i}*linsys.taylor.dtoverfac{1}(i+1);
            Asum = Asum + linsys.taylor.Apower[i] * linsys.taylor.dtoverfac[0][i + 1]
        
        # MATLAB: eAtInt = Asum + linsys.taylor.E{1}*options.timeStep;
        # Ensure taylor.E is initialized as a list (MATLAB cell array E{1})
        from cora_python.contDynamics.linearSys.private.priv_expmRemainder import priv_expmRemainder
        from cora_python.matrixSet.intervalMatrix import IntervalMatrix
        
        # Check if E needs to be computed
        needs_E = False
        if not hasattr(linsys.taylor, 'E'):
            needs_E = True
        elif linsys.taylor.E is None:
            needs_E = True
        elif isinstance(linsys.taylor.E, list) and (len(linsys.taylor.E) == 0 or linsys.taylor.E[0] is None):
            needs_E = True
        
        if needs_E:
            E_interval = priv_expmRemainder(linsys, options['timeStep'], options['taylorTerms'])
            # Convert Interval to IntervalMatrix (MATLAB: interval(-W,W) creates interval matrix)
            # E_interval is an Interval with matrix bounds, convert to IntervalMatrix
            # Extract center and delta from interval: center = (inf + sup) / 2, delta = (sup - inf) / 2
            center = (E_interval.inf + E_interval.sup) / 2
            delta = (E_interval.sup - E_interval.inf) / 2
            E = IntervalMatrix(center, delta)
            linsys.taylor.E = [E]  # Store as list to match MATLAB cell array E{1}
        else:
            E = linsys.taylor.E[0]
        
        # MATLAB: eAtInt = Asum + linsys.taylor.E{1}*options.timeStep;
        # In MATLAB, numeric_matrix + interval_matrix creates an interval matrix
        # E is an IntervalMatrix, so we need to add Asum to it
        E_scaled = E * options['timeStep']  # Scale the interval matrix
        eAtInt = Asum + E_scaled  # This should use IntervalMatrix.plus
        
        # MATLAB: Rerror = eAtInt*errorStat;
        Rerror = eAtInt * errorStat
        
        # MATLAB: R_tp = exactPlus(R_tp,Rerror) + nlnsys.linError.p.x;
        R_tp = R_tp.exactPlus(Rerror) + nlnsys.linError.p.x
        
        # MATLAB: R_ti = exactPlus(R_ti,Rerror) + nlnsys.linError.p.x;
        R_ti = R_ti.exactPlus(Rerror) + nlnsys.linError.p.x
    
    # init output variables
    # MATLAB: R_tp = noIndep(reduce(R_tp,options.reductionTechnique,options.zonotopeOrder));
    # NOTE: noIndep removes independent generators from polyZonotope
    # For now, if it's a polyZonotope, use its noIndep method if available
    R_tp_reduced = R_tp.reduce(options['reductionTechnique'], options['zonotopeOrder'])
    if hasattr(R_tp_reduced, 'noIndep'):
        R_tp = R_tp_reduced.noIndep()
    else:
        R_tp = R_tp_reduced
    
    # MATLAB: R_ti = noIndep(reduce(R_ti,options.reductionTechnique,options.zonotopeOrder));
    R_ti_reduced = R_ti.reduce(options['reductionTechnique'], options['zonotopeOrder'])
    if hasattr(R_ti_reduced, 'noIndep'):
        R_ti = R_ti_reduced.noIndep()
    else:
        R_ti = R_ti_reduced
    
    # MATLAB: Rtp_.set = R_tp;
    Rtp_dict = {'set': R_tp}
    
    # MATLAB: Rti = R_ti;
    Rti = R_ti
    
    # MATLAB: Rtp_.error = zeros(length(R_tp.c),1);
    Rtp_dict['error'] = np.zeros((len(R_tp.c), 1))
    
    # MATLAB: Rtp = Rtp_;
    Rtp = Rtp_dict
    
    # MATLAB: dimForSplit = [];
    dimForSplit = None
    
    return Rtp, Rti, dimForSplit, options


def _is_linear_param_sys(linsys: Any) -> bool:
    """Robust check for LinearParamSys instances without string matching."""
    try:
        from cora_python.contDynamics.linearParamSys import LinearParamSys
        return isinstance(linsys, LinearParamSys)
    except Exception:
        return hasattr(linsys, '__class__') and 'linearparamsys' in linsys.__class__.__name__.lower()

