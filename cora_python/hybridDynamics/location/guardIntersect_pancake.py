"""
guardIntersect_pancake - implementation of the time scaling approach
   described in [1]

Syntax:
    R = guardIntersect_pancake(loc,R0,guard,guardID,params,options)

Inputs:
    loc - location object
    R0 - initial set (last reachable set not intersecting the guard set)
    guard - guard set (class: constrained hyperplane)
    guardID - ID of the guard set
    params - model parameters
    options - struct containing the algorithm settings

Outputs:
    R - set enclosing the guard intersection

References: 
  [1] S. Bak et al. "Time-Triggered Conversion of Guards for Reachability
      Analysis of Hybrid Automata"

Authors:       Niklas Kochdumper
Written:       05-November-2018             
Last update:   20-November-2019
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import os
from typing import Any, Dict, Tuple
from cora_python.contSet.polytope import Polytope
from cora_python.contSet.interval import Interval
from cora_python.specification.specification.specification import Specification
from cora_python.g.macros.CORAROOT import CORAROOT
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def guardIntersect_pancake(loc: Any, R0: Any, guard: Polytope, guardID: int,
                           params: Dict[str, Any], options: Dict[str, Any]) -> Any:
    """
    Implementation of the time scaling approach for guard intersection
    
    Args:
        loc: location object
        R0: initial set (last reachable set not intersecting guard)
        guard: guard set (polytope representing constrained hyperplane)
        guardID: ID of the guard set
        params: model parameters
        options: struct containing the algorithm settings
        
    Returns:
        R: set enclosing the guard intersection
    """
    
    # initialization
    sys = loc.contDynamics
    
    # check if guard set is a constrained hyperplane
    if not (isinstance(guard, Polytope) and guard.representsa_('conHyperplane', 1e-12)):
        raise CORAerror('CORA:specialError',
                       "The method 'pancake' only supports guards given as polytope objects that represent constrained hyperplanes!")
    
    # convert hyperplane to a halfspace that represents the outside of the
    # invariant set
    c = R0.center()
    P = Polytope(A_eq=guard.Ae, b_eq=guard.be)
    
    if P.contains_(c, 'exact', 1e-12, 0, False, False):
        P = Polytope(A_eq=-guard.Ae, b_eq=-guard.be)
    
    # set default options for nonlinear system reachability analysis
    optionsScaled = options.copy()
    
    if not (hasattr(sys, '__class__') and sys.__class__.__name__ == 'nonlinearSys'):
        optionsScaled = _aux_defaultOptions(options)
    
    # create system for the time-scaled system dynamics
    sys_scaled, params_scaled = _aux_scaledSystem(sys, P, R0, guardID, params)
    
    # compute the reachable set for the time scaled system
    R = _aux_reachTimeScaled(sys_scaled, P, R0, params_scaled, optionsScaled)
    
    # jump across the guard set in only one time step
    R = _aux_jump(sys, P, R, params, options)
    
    # project the reachable set onto the hyperplane
    R = R.projectOnHyperplane(guard)
    
    return R


# Auxiliary functions -----------------------------------------------------

def _aux_scaledSystem(sys: Any, P: Polytope, R0: Any, guardID: int, 
                      params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    """
    Scale the system dynamics using the distance to the hyperplane as a 
    scaling factor
    """
    
    # get maximum distance of initial set to hyperplane
    temp = P.Ae @ R0 + (-P.be)
    temp_interval = Interval(temp) if not isinstance(temp, Interval) else temp
    maxDist = temp_interval.supremum() if hasattr(temp_interval, 'supremum') else temp_interval.sup
    
    params_scaled = params.copy()
    params_scaled['paramInt'] = maxDist
    
    # define scaling function
    def g(x, p):
        return (P.Ae @ x - P.be) / p
    
    # get system dynamics
    n = sys.nr_of_dims if hasattr(sys, 'nr_of_dims') else sys.nrOfDims
    m = sys.nr_of_inputs if hasattr(sys, 'nr_of_inputs') else sys.nrOfInputs
    
    if hasattr(sys, '__class__') and sys.__class__.__name__ == 'linearSys':
        def f(x, u):
            return _aux_dynamicsLinSys(x, u, sys)
    else:
        f = sys.mFile
    
    # time scaled system dynamics
    def F(x, u, p):
        return g(x, p) * f(x, u)
    
    # create symbolic variables and generate function file
    import sympy as sp
    from cora_python.g.functions.matlab.matlabFunction import matlabFunction
    
    xSym = sp.Matrix(sp.symbols(f'x:{n}', real=True))
    uSym = sp.Matrix(sp.symbols(f'u:{m}', real=True))
    pSym = sp.Symbol('p', real=True)
    
    # create file path
    name = f'generated_{sys.name}_{guardID}_timeScaled'
    foldername = os.path.join(CORAROOT(), 'models', 'auxiliary')
    path = os.path.join(foldername, name)
    
    # create file for time scaled dynamics
    func = F(xSym, uSym, pSym)
    
    if not os.path.isdir(foldername):
        os.makedirs(foldername, exist_ok=True)
    
    funcHandle = matlabFunction(func, 'File', path, 'Vars', [xSym, uSym, pSym])
    
    # create time scaled system
    from cora_python.contDynamics.nonlinParamSys import nonlinParamSys
    sys_scaled = nonlinParamSys(funcHandle, n, m, 1)
    
    return sys_scaled, params_scaled


def _aux_reachTimeScaled(sys: Any, P: Polytope, R0: Any, params: Dict[str, Any],
                         options: Dict[str, Any]) -> Any:
    """
    Compute the reachable set of the scaled system such that the final
    reachable set until the scaled reachable set is very close to the
    hyperplane
    """
    
    # adapt options
    spec = Specification(P, 'unsafeSet')
    params_reach = params.copy()
    params_reach['R0'] = R0
    if 'maxError' in options:
        options_reach = options.copy()
        options_reach['maxError'] = np.inf * np.ones_like(options['maxError'])
    else:
        options_reach = options.copy()
    
    # prevent validateOptions error
    if (hasattr(sys, '__class__') and sys.__class__.__name__ == 'nonlinParamSys' and
        'intermediateTerms' not in options_reach):
        options_reach['intermediateTerms'] = 4
    
    # compute reachable set until
    R = sys.reach(params_reach, options_reach, spec)
    
    # get final reachable set
    Rfin = R.timePoint['set'][-1]
    
    return Rfin


def _aux_jump(sys: Any, P: Polytope, R0: Any, params: Dict[str, Any],
              options: Dict[str, Any]) -> Any:
    """
    Compute the reachable set in such a way that the reachable set jumps in 
    only one time step across the hyperplane
    """
    
    params_jump = params.copy()
    params_jump['R0'] = R0
    timeStep = options.get('timeStep', 0.01)
    timeStep_ = timeStep
    
    # compute reachable set
    options_jump = options.copy()
    options_jump['timeStep'] = timeStep
    params_jump['tFinal'] = timeStep
    params_jump['tStart'] = 0
    
    R = sys.reach(params_jump, options_jump)
    
    # check if located inside the invariant
    dist_ = R.timePoint['set'][-1].supportFunc_(P.Ae.T, 'upper', 'interval', 8, 1e-3) - P.be
    
    if dist_ < 0:
        # guard set crossed -> reduce time step size to get smaller set
        Rcont = R.timeInterval['set'][-1]
        distMin = R0.supportFunc_(P.Ae.T, 'lower', 'interval', 8, 1e-3) - P.be
        lb = 0.0
        ub = timeStep
        
        for i in range(10):
            
            # update time step
            timeStep = (ub - lb) / 2
            options_jump['timeStep'] = timeStep
            params_jump['tFinal'] = timeStep
            
            # compute reachable set
            R = sys.reach(params_jump, options_jump)
            
            # check if located inside the invariant
            dist = R.timePoint['set'][-1].supportFunc_(P.Ae.T, 'upper', 'interval', 8, 1e-3) - P.be
            
            if dist < 0:
                Rcont = R.timeInterval['set'][-1]
                ub = timeStep
                if abs(dist) <= distMin:
                    break
            else:
                lb = timeStep
    else:
        # guard set not crossed -> increase time interval
        while True:
            
            # update time step
            timeStep = timeStep + timeStep_
            options_jump['timeStep'] = timeStep
            params_jump['tFinal'] = timeStep
            
            # compute reachable set
            R = sys.reach(params_jump, options_jump)
            
            # check if located inside the invariant
            dist = R.timePoint['set'][-1].supportFunc_(P.Ae.T, 'upper', 'interval', 8, 1e-3) - P.be
            
            if dist < 0:
                Rcont = R.timeInterval['set'][-1]
                break
            elif dist > dist_:
                raise CORAerror('CORA:specialError', 'Pancake approach failed!')
            else:
                dist_ = dist
    
    return Rcont


def _aux_defaultOptions(options: Dict[str, Any]) -> Dict[str, Any]:
    """
    Set default options for nonlinear system reachability analysis (required
    if the continuous dynamics of the hybrid automaton is linear)
    """
    
    # define options and default values
    opts = ['alg', 'tensorOrder', 'errorOrder', 'intermediateOrder',
            'zonotopeOrder', 'taylorTerms', 'timeStep', 'intermediateTerms']
    defVal = ['lin', 3, 5, 50, 50, 10, 0.01, 4]
    
    # parse options
    options_default = {}
    for i, opt in enumerate(opts):
        if opt in options:
            options_default[opt] = options[opt]
        else:
            options_default[opt] = defVal[i]
    
    return options_default


def _aux_dynamicsLinSys(x: np.ndarray, u: np.ndarray, sys: Any) -> np.ndarray:
    """
    Dynamic function of a linear system
    """
    f = sys.A @ x + sys.B @ u
    if hasattr(sys, 'c') and sys.c is not None and sys.c.size > 0:
        f = f + sys.c
    return f

