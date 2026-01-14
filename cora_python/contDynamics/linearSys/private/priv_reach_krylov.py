"""
priv_reach_krylov - computes the reachable set for linear systems using
   the krylov reachability algorithm for linear systems [1]

Syntax:
    [timeInt,timePoint,res] = priv_reach_krylov(linsys,params,options)

Inputs:
    linsys - linearSys object
    params - model parameters
    options - options for the computation of reachable sets

Outputs:
    timeInt - array of time-interval output sets
    timePoint - array of time-point output sets
    res - true/false (only if specification given)

Example:
    -

References:
    [1] M. Althoff. "Reachability analysis of large linear systems with
        uncertain inputs in the Krylov subspace", IEEE Transactions on
        Automatic Control 65 (2), pp. 477-492, 2020.

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Maximilian Perschl, Matthias Althoff, Mark Wetzlinger
Written:       26-June-2019
Last update:   19-November-2022 (MW, modularize specification check)
               23-April-2025 (MP, refactor including new error bound)
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import scipy.linalg
from typing import Any, Dict, Tuple
from cora_python.contSet.interval.interval import Interval
from cora_python.contSet.zonotope.zonotope import Zonotope
from cora_python.g.functions.verbose.verboseLog import verboseLog
from .priv_checkSpecification import priv_checkSpecification


def priv_reach_krylov(linsys: Any, params: Dict[str, Any], options: Dict[str, Any]) -> Tuple[Dict, Dict, bool]:
    """
    Computes the reachable set for linear systems using the Krylov reachability algorithm
    
    Args:
        linsys: linearSys object
        params: model parameters
        options: options for the computation of reachable sets
        
    Returns:
        timeInt: dict with 'set' and 'time' keys for time-interval output sets
        timePoint: dict with 'set' and 'time' keys for time-point output sets
        res: true/false whether specification satisfied
    """
    
    # if a trajectory should be tracked
    if 'uTransVec' in params:
        # MATLAB: params.uTrans = params.uTransVec(:,1); (1-based, first column)
        # Python: use index 0 for first column
        params['uTrans'] = params['uTransVec'][:, 0:1]
    
    # log information
    verboseLog(options.get('verbose', False), 1, params['tStart'], params['tStart'], params['tFinal'])
    
    # initialize reachable set computations
    from .priv_initReach_Krylov import priv_initReach_Krylov
    linsys, params, options = priv_initReach_Krylov(linsys, params, options)
    
    # time period
    # MATLAB: tVec = params.tStart:options.timeStep:params.tFinal;
    # Use similar approach to other reach functions for consistency
    steps = int(np.round((params['tFinal'] - params['tStart']) / options['timeStep']))
    tVec = np.linspace(params['tStart'], params['tFinal'], steps + 1)
    steps = len(tVec)
    
    # create options.t
    options['t'] = params['tStart']
    
    # initialize arguments for the output equation
    timeInt = {'set': [None] * (steps - 1), 'time': [None] * (steps - 1)}
    timePoint = {'set': [None] * steps, 'time': [None] * steps}
    
    # fill in values for initial set
    # MATLAB: timePoint.set{1} = linsys.C*params.R0;
    timePoint['set'][0] = linsys.C @ params['R0']
    timePoint['time'][0] = 0.0
    
    print(f"Krylov reachability: starting computation for {steps} steps")
    # loop over all reachability steps
    for k in range(1, steps):  # k from 1 to steps-1 (0-based: 1 to steps-1)
        
        # Print progress every 100 steps
        if k % 100 == 0:
            print(f"Krylov reachability: step {k}/{steps-1} ({(k/(steps-1)*100):.1f}%)")
        
        # increment time
        options['t'] = tVec[k]
        
        # propagate different sets using exponential matrix in Krylov subspace
        Rnext = _aux_propagation(linsys, params, options)
        
        # calculate the output set
        timeInt['set'][k - 1] = Rnext['ti']
        timeInt['time'][k - 1] = Interval([tVec[k - 1]], [tVec[k]])
        timePoint['set'][k] = Rnext['tp']
        timePoint['time'][k] = options['t']
        
        # safety property check
        if 'specification' in options:
            # MATLAB: priv_checkSpecification(...,k-1) where k goes from 2 to steps (1-based)
            # So k-1 goes from 1 to steps-1 (1-based index for time interval)
            # Python k goes from 1 to steps-1 (0-based), so k corresponds to MATLAB k-1
            # priv_checkSpecification expects 1-based index and converts internally with idx-1
            # So we pass k (which is 1-based equivalent: first iteration k=1 = first time interval)
            res, timeInt, timePoint = priv_checkSpecification(
                options['specification'], [], timeInt, timePoint, k)
            if not res:
                return timeInt, timePoint, res
        
        # log information
        verboseLog(options.get('verbose', False), k + 1, tVec[k], params['tStart'], params['tFinal'])
        
        # if a trajectory should be tracked
        if 'uTransVec' in params:
            # MATLAB: params.uTrans = params.uTransVec(:,k); where k goes from 2 to steps (1-based)
            # Python k goes from 1 to steps-1 (0-based)
            # MATLAB k=2 uses column 2 (index 1 in 0-based), Python k=1 should use column index 1
            # So Python k directly corresponds to the 0-based column index
            uTrans_idx = k if k < params['uTransVec'].shape[1] else params['uTransVec'].shape[1] - 1
            params['uTrans'] = params['uTransVec'][:, uTrans_idx:uTrans_idx+1]
            # input trajectory not yet implemented
    
    # specification fulfilled
    res = True
    return timeInt, timePoint, res


# Auxiliary functions -----------------------------------------------------

def _aux_propagation(sys: Any, params: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
    """
    Propagate all set solutions 
    
    Args:
        sys: linearSys object
        params: model parameters
        options: options
        
    Returns:
        Rnext: dict with keys:
            - tp: timepoint solution
            - ti: timeinterval solution
    """
    
    # retrieve reachable set
    U_0_sol = sys.krylov.get('Rpar_proj', None)
    Rhom_tp_prev = sys.krylov.get('Rhom_tp_prev', None)
    
    # propagate
    # homogeneous solution
    state = sys.krylov.get('state', {})
    c_sys_proj = state.get('c_sys_proj', None)
    g_sys_proj = state.get('g_sys_proj', [])
    Htp, TIE_new_proj = _aux_propagate_HomSol(
        c_sys_proj,
        g_sys_proj,
        options['timeStep'],
        options['t'] - options['timeStep'],
        params['R0'],
        options.get('taylorTerms', 10),
        options.get('krylovError', 0),
        sys.nr_of_outputs if hasattr(sys, 'nr_of_outputs') else sys.nrOfOutputs
    )
    
    # uncertain input solution
    from .priv_exponential_Krylov_projected_linSysInput import priv_exponential_Krylov_projected_linSysInput
    
    if not isinstance(U_0_sol, (int, float)) and U_0_sol is not None:
        input_krylov = sys.krylov.get('input', {})
        c_sys_proj_input = input_krylov.get('c_sys_proj', None)
        g_sys_proj_input = input_krylov.get('g_sys_proj', [])
        RV_0 = sys.krylov.get('RV_0', None)
        if RV_0 is not None:
            RV_proj, _ = priv_exponential_Krylov_projected_linSysInput(
                c_sys_proj_input,
                g_sys_proj_input,
                RV_0,
                sys.nr_of_outputs if hasattr(sys, 'nr_of_outputs') else sys.nrOfOutputs,
                options
            )
        else:
            RV_proj = Zonotope(np.zeros((sys.nr_of_outputs if hasattr(sys, 'nr_of_outputs') else sys.nrOfOutputs, 1)),
                              np.array([]).reshape(sys.nr_of_outputs if hasattr(sys, 'nr_of_outputs') else sys.nrOfOutputs, 0))
    else:
        # for the first step, we already computed the input solution in
        # init_reach_krylov
        RV_proj = sys.krylov.get('Rpar_proj_0', None)
    
    # constant input solution
    # dummy R0 to get norm correct
    uTrans_sys = sys.krylov.get('uTrans_sys', None)
    if uTrans_sys is not None:
        RTrans_new_proj, _ = priv_exponential_Krylov_projected_linSysInput(
            uTrans_sys,
            [],
            Zonotope(np.array([[1]]), np.array([]).reshape(1, 0)),
            sys.nr_of_outputs if hasattr(sys, 'nr_of_outputs') else sys.nrOfOutputs,
            options
        )
    else:
        RTrans_new_proj = Zonotope(np.zeros((sys.nr_of_outputs if hasattr(sys, 'nr_of_outputs') else sys.nrOfOutputs, 1)),
                                   np.array([]).reshape(sys.nr_of_outputs if hasattr(sys, 'nr_of_outputs') else sys.nrOfOutputs, 0))
    
    # compute new solutions for this timestep
    
    # we save U_0_sol as interval because otherwise we would have way too many
    # generators to justify the loss in precision
    # MATLAB: U_0_sol = U_0_sol + interval(RV_proj);
    # In MATLAB, Zonotope + Interval returns an Interval (to save generators)
    # In Python, Zonotope + Interval returns a Zonotope, so we convert to Interval to match MATLAB
    U_0_sol_temp = U_0_sol + RV_proj.interval()
    # Convert to Interval to match MATLAB behavior (saves generators)
    if isinstance(U_0_sol_temp, Zonotope):
        U_0_sol = U_0_sol_temp.interval()
    else:
        U_0_sol = U_0_sol_temp
    # MATLAB: U_0_sol = reduce(U_0_sol,options.reductionTechnique,options.zonotopeOrder);
    U_0_sol = U_0_sol.reduce(options.get('reductionTechnique', 'girard'), 
                             options.get('zonotopeOrder', 5))
    
    # next time step solution
    Rhom_tp_proj = Htp + RTrans_new_proj
    # MATLAB: R_tp_proj = Rhom_tp_proj + zonotope(U_0_sol);
    # In MATLAB, U_0_sol is an Interval, so zonotope(U_0_sol) converts it to Zonotope
    # Import at top level to avoid repeated imports in loop
    U_0_sol_zono = U_0_sol.zonotope()
    R_tp_proj = Rhom_tp_proj + U_0_sol_zono
    
    # affine time interval solution
    # MATLAB: TI_apx = enclose(Rhom_tp_prev,Rhom_tp_proj);
    TI_apx = Rhom_tp_prev.enclose(Rhom_tp_proj)
    
    # complete time-interval solution
    inputCorr = sys.krylov.get('inputCorr', Zonotope(np.zeros((sys.nr_of_outputs if hasattr(sys, 'nr_of_outputs') else sys.nrOfOutputs, 1)),
                                                     np.array([]).reshape(sys.nr_of_outputs if hasattr(sys, 'nr_of_outputs') else sys.nrOfOutputs, 0)))
    # MATLAB: R_ti_proj = TI_apx + U_0_sol + TIE_new_proj + sys.krylov.inputCorr;
    # In MATLAB:
    #   - TI_apx is a Zonotope (from enclose)
    #   - U_0_sol is an Interval (saved as interval to reduce generators)
    #   - TIE_new_proj (hom_tie) is an Interval (from aux_propagate_HomSol)
    #   - inputCorr is a Zonotope
    # MATLAB handles Zonotope + Interval automatically (converts Interval to Zonotope)
    # In Python, we need to convert Intervals to Zonotopes for addition
    # Convert U_0_sol (Interval) to Zonotope for addition
    if isinstance(U_0_sol, Interval):
        U_0_sol_zono = U_0_sol.zonotope()
    else:
        U_0_sol_zono = U_0_sol
    
    # Convert TIE_new_proj (Interval or numpy array) to Zonotope for addition
    if isinstance(TIE_new_proj, Interval):
        TIE_new_proj_zono = TIE_new_proj.zonotope()
    elif isinstance(TIE_new_proj, np.ndarray):
        # Convert numpy array to Zonotope (center only, no generators)
        if TIE_new_proj.ndim == 1:
            TIE_new_proj = TIE_new_proj.reshape(-1, 1)
        TIE_new_proj_zono = Zonotope(TIE_new_proj, np.array([]).reshape(TIE_new_proj.shape[0], 0))
    else:
        TIE_new_proj_zono = TIE_new_proj
    
    R_ti_proj = TI_apx + U_0_sol_zono + TIE_new_proj_zono + inputCorr
    
    # order reduction
    Rnext = {
        'ti': R_ti_proj.reduce(options.get('reductionTechnique', 'girard'),
                              options.get('zonotopeOrder', 5)),
        'tp': R_tp_proj.reduce(options.get('reductionTechnique', 'girard'),
                              options.get('zonotopeOrder', 5))
    }
    
    # save result
    sys.krylov['Rhom_tp_prev'] = Rhom_tp_proj
    sys.krylov['Rpar_proj'] = U_0_sol
    
    return Rnext


def _aux_propagate_HomSol(c_sys: Any, g_sys: list, timeStep: float, time: float,
                         R0: Any, taylorTerms: int, krylovError: float, dim_proj: int) -> Tuple[Any, Any]:
    """
    Propagate the homogeneous solution as well as the time-interval error
    stemming from the homogeneous solution
    
    Args:
        c_sys: center system (linearSys object or None)
        g_sys: list of generator systems (linearSys objects)
        timeStep: time step size
        time: current time
        R0: initial reachable set
        taylorTerms: number of Taylor terms
        krylovError: Krylov error bound
        dim_proj: dimension of projection (output dimension)
        
    Returns:
        hom_tp: homogeneous time-point solution
        hom_tie: homogeneous time-interval error
    """
    
    from scipy.sparse import csc_matrix
    from cora_python.contSet.zonotope.zonotope import Zonotope
    
    # MATLAB: c = sparse(center(R0));
    from scipy.sparse import issparse
    if hasattr(R0, 'center'):
        c = R0.center()
    elif hasattr(R0, 'c'):
        c = R0.c
    else:
        dim = R0.dim() if hasattr(R0, 'dim') else (R0.nr_of_dims if hasattr(R0, 'nr_of_dims') else 1)
        c = np.zeros((dim, 1))
    if issparse(c):
        c = c.toarray().flatten()
    else:
        c = np.asarray(c).flatten()
    
    # MATLAB: G = sparse(generators(R0));
    if hasattr(R0, 'generators'):
        G = R0.generators()
    elif hasattr(R0, 'G'):
        G = R0.G
    else:
        dim = R0.dim() if hasattr(R0, 'dim') else (R0.nr_of_dims if hasattr(R0, 'nr_of_dims') else 1)
        G = np.zeros((dim, 0))
    if issparse(G):
        G = G.toarray()
    else:
        G = np.asarray(G)
    
    # check if center is zero
    c_norm = np.linalg.norm(c)
    
    if c_sys is None or (hasattr(c_sys, '__len__') and len(c_sys) == 0):
        c_new = np.zeros((dim_proj, 1))
        # MATLAB: hom_tie = zeros(dim_proj,1);
        # In Python, we'll convert to Zonotope later for consistency
        hom_tie = np.zeros((dim_proj, 1))
    else:
        # Compute new center
        # MATLAB: eAtk = readFieldForTimeStep(c_sys.taylor,'eAdt',time);
        eAtk = c_sys.taylor.readFieldForTimeStep('eAdt', time)
        if eAtk is None or (hasattr(eAtk, 'size') and eAtk.size == 0):
            eAtk = scipy.linalg.expm(c_sys.A * time)
            c_sys.taylor.insertFieldTimeStep('eAdt', eAtk, time)
        
        # MATLAB: eAdt = readFieldForTimeStep(c_sys.taylor,'eAdt',timeStep);
        eAdt = c_sys.taylor.readFieldForTimeStep('eAdt', timeStep)
        if eAdt is None or (hasattr(eAdt, 'size') and eAdt.size == 0):
            eAdt = scipy.linalg.expm(c_sys.A * timeStep)
            c_sys.taylor.insertFieldTimeStep('eAdt', eAdt, timeStep)
        
        c_expMatrix = eAdt @ eAtk
        c_sys.taylor.insertFieldTimeStep('eAdt', c_expMatrix, time + timeStep)
        
        # MATLAB: c_new = c_norm*c_sys.B'*c_expMatrix(:,1);
        c_new = c_norm * (c_sys.B.T @ c_expMatrix[:, 0:1])
        
        # Compute new TIE
        # MATLAB: [~,F_c] = taylorMatrices(c_sys,timeStep,taylorTerms);
        # taylorMatrices returns (E, F, G), we only need F
        _, F_c, _ = c_sys.taylorMatrices(timeStep, taylorTerms)
        # MATLAB: hom_tie = c_norm*c_sys.B'*c_expMatrix*F_c(:,1);
        # F_c is an IntervalMatrix, F_c(:,1) extracts a column which should be an Interval
        # In MATLAB, F_c(:,1) returns an Interval (vector), not an IntervalMatrix
        # Extract the first column as an Interval (vector)
        from cora_python.matrixSet.intervalMatrix import IntervalMatrix
        from cora_python.contSet.interval import Interval
        if isinstance(F_c, IntervalMatrix):
            # Extract first column as Interval (vector)
            F_c_col_inf = F_c.infimum()[:, 0]
            F_c_col_sup = F_c.supremum()[:, 0]
            F_c_col = Interval(F_c_col_inf, F_c_col_sup)
            # c_expMatrix is a regular matrix, F_c_col is an Interval (vector)
            # c_expMatrix @ F_c_col should return an Interval (vector)
            from cora_python.contSet.interval.mtimes import mtimes as interval_mtimes
            temp = interval_mtimes(c_expMatrix, F_c_col)
            # c_sys.B.T is a regular matrix, temp is an Interval (vector)
            hom_tie = c_norm * interval_mtimes(c_sys.B.T, temp)
        else:
            hom_tie = c_norm * (c_sys.B.T @ c_expMatrix @ F_c[:, 0:1])
    
    # preallocation
    nrOfGens = G.shape[1] if G.ndim > 1 and G.shape[1] > 0 else 0
    
    # obtain generators using the Arnoldi iteration
    G_new = np.zeros((dim_proj, nrOfGens)) if nrOfGens > 0 else np.array([]).reshape(dim_proj, 0)
    
    if nrOfGens > 0:
        for iGen in range(nrOfGens):
            g_col = G[:, iGen]
            g_norm = np.linalg.norm(g_col)
            g_sys_i = g_sys[iGen]
            expMatrix = None  # Initialize for TIE computation
            
            if g_norm == 0:
                G_new[:, iGen] = g_col.flatten()[:dim_proj] if g_col.size >= dim_proj else np.pad(g_col.flatten(), (0, dim_proj - g_col.size))[:dim_proj]
            elif g_norm <= 1e-8:
                print("HI")  # Debug message from MATLAB
                # For small generators, we still need expMatrix for TIE computation
                eAtk = g_sys_i.taylor.readFieldForTimeStep('eAdt', time)
                if eAtk is None or (hasattr(eAtk, 'size') and eAtk.size == 0):
                    eAtk = scipy.linalg.expm(g_sys_i.A * time)
                    g_sys_i.taylor.insertFieldTimeStep('eAdt', eAtk, time)
                eAdt = g_sys_i.taylor.readFieldForTimeStep('eAdt', timeStep)
                if eAdt is None or (hasattr(eAdt, 'size') and eAdt.size == 0):
                    eAdt = scipy.linalg.expm(g_sys_i.A * timeStep)
                    g_sys_i.taylor.insertFieldTimeStep('eAdt', eAdt, timeStep)
                expMatrix = eAdt @ eAtk
                g_sys_i.taylor.insertFieldTimeStep('eAdt', expMatrix, time + timeStep)
            else:
                # Compute new generator
                # MATLAB: eAtk = readFieldForTimeStep(g_sys{iGen}.taylor,'eAdt',time);
                eAtk = g_sys_i.taylor.readFieldForTimeStep('eAdt', time)
                if eAtk is None or (hasattr(eAtk, 'size') and eAtk.size == 0):
                    eAtk = scipy.linalg.expm(g_sys_i.A * time)
                    g_sys_i.taylor.insertFieldTimeStep('eAdt', eAtk, time)
                
                # MATLAB: eAdt = readFieldForTimeStep(g_sys{iGen}.taylor,'eAdt',timeStep);
                eAdt = g_sys_i.taylor.readFieldForTimeStep('eAdt', timeStep)
                if eAdt is None or (hasattr(eAdt, 'size') and eAdt.size == 0):
                    eAdt = scipy.linalg.expm(g_sys_i.A * timeStep)
                    g_sys_i.taylor.insertFieldTimeStep('eAdt', eAdt, timeStep)
                
                expMatrix = eAdt @ eAtk
                g_sys_i.taylor.insertFieldTimeStep('eAdt', expMatrix, time + timeStep)
                
                # MATLAB: G_new(:,iGen) = g_norm*g_sys{iGen}.B'*expMatrix(:,1);
                G_new[:, iGen] = (g_norm * (g_sys_i.B.T @ expMatrix[:, 0:1])).flatten()
            
            # compute new TIE for this generator
            # MATLAB: This is computed for ALL generators (including zero and small ones)
            # For zero generators, g_norm=0 so the contribution is zero (0 * B' * expMatrix * F_g = 0)
            # For non-zero generators, we need expMatrix (already computed above)
            if g_norm > 0 and expMatrix is not None:
                # MATLAB: [~,F_g] = taylorMatrices(g_sys{iGen},timeStep,taylorTerms);
                # taylorMatrices returns (E, F, G), we only need F
                _, F_g, _ = g_sys_i.taylorMatrices(timeStep, taylorTerms)
                # Extract first column as Interval (vector) if F_g is IntervalMatrix
                if isinstance(F_g, IntervalMatrix):
                    F_g_col_inf = F_g.infimum()[:, 0]
                    F_g_col_sup = F_g.supremum()[:, 0]
                    F_g_col = Interval(F_g_col_inf, F_g_col_sup)
                    from cora_python.contSet.interval.mtimes import mtimes as interval_mtimes
                    temp = interval_mtimes(expMatrix, F_g_col)
                    F_g_term = g_norm * interval_mtimes(g_sys_i.B.T, temp)
                    hom_tie = hom_tie + F_g_term
                else:
                    hom_tie = hom_tie + g_norm * (g_sys_i.B.T @ expMatrix @ F_g[:, 0:1])
    else:
        G_new = np.array([]).reshape(dim_proj, 0)  # no generators
    
    if krylovError > 2 * np.finfo(float).eps:
        # Krylov error computation
        # +1 due to center
        if hasattr(R0, 'generators'):
            R0_G = R0.generators()
        elif hasattr(R0, 'G'):
            R0_G = R0.G
        else:
            dim = R0.dim() if hasattr(R0, 'dim') else (R0.nr_of_dims if hasattr(R0, 'nr_of_dims') else 1)
            R0_G = np.array([]).reshape(dim, 0)
        R0_G_cols = R0_G.shape[1] if R0_G.ndim > 1 and R0_G.shape[1] > 0 else 0
        error = krylovError * (R0_G_cols + 1)
        
        # initial-state-solution zonotope
        error_matrix = error * np.eye(dim_proj)
        hom_tp = Zonotope(c_new, np.hstack([G_new, error_matrix]) if G_new.size > 0 else error_matrix)
    else:
        hom_tp = Zonotope(c_new, G_new) if G_new.size > 0 else Zonotope(c_new, np.array([]).reshape(dim_proj, 0))
    
    # MATLAB: hom_tie can be a numeric array (zeros) or an Interval
    # In MATLAB, it's used directly in addition operations which handle type conversion
    # In Python, we keep it as-is and let the addition operations handle conversion
    # If it's a numpy array, it will be converted during addition
    # If it's an Interval, it will be converted to Zonotope during addition
    return hom_tp, hom_tie

