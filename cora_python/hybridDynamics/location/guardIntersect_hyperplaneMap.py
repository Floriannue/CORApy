"""
guardIntersect_hyperplaneMap - implementation of the guard mapping
   approach described in [1]

Syntax:
    R = guardIntersect_hyperplaneMap(loc,guard,R0,params,options)

Inputs:
    loc - location object
    guard - guard set (class: polytope)
    R0 - initial set (last reachable set not intersecting the guard set)
    params - model parameters
    options - struct containing the algorithm settings

Outputs:
    R - reachable set mapped to the guard set

References: 
  [1] M. Althoff et al. "Avoiding Geometic Intersection Operations in 
      Reachability Analysis of Hybrid Systems"
  [2] M. Althoff et al. "Reachability Analysis of Nonlinear Systems with 
      Uncertain Parameters using Conservative Linearization"

Authors:       Matthias Althoff, Niklas Kochdumper
Written:       13-December-2019
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import Any, Dict, Tuple
from cora_python.contSet.polytope import Polytope
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.interval import Interval
from cora_python.matrixSet.matZonotope.matZonotope import matZonotope
from cora_python.specification.specification.specification import Specification
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def guardIntersect_hyperplaneMap(loc: Any, guard: Polytope, R0: Any, 
                                  params: Dict[str, Any], options: Dict[str, Any]) -> Any:
    """
    Implementation of the guard mapping approach
    
    Args:
        loc: location object
        guard: guard set (polytope representing constrained hyperplane)
        R0: initial set (last reachable set not intersecting guard)
        params: model parameters
        options: struct containing the algorithm settings
        
    Returns:
        R: reachable set mapped to the guard set
    """
    
    if not isinstance(guard, Polytope) or not guard.representsa_('conHyperplane', 1e-12):
        raise CORAerror('CORA:notSupported', 'Guard set must be a constrained hyperplane.')
    
    # refine the time interval at which the guard set is hit
    R0_refined, tmin, tmax, Rcont = _aux_refinedIntersectionTime(loc, guard, R0, params, options)
    tmax = tmax - tmin
    
    # average hitting time
    th = tmax / 2
    
    # system matrix A and set of uncertain inputs U
    A, U = _aux_systemParams(loc.contDynamics, Rcont, params)
    
    # constant part b of the flow \dot y = A*y0 + b (see Prop. 1 in [1])
    b = _aux_constantFlow(A, R0_refined, U, th, options.get('taylorTerms', 10))
    
    # reduce order of the initial set to speed up the computations
    R0_zonotope = Zonotope(R0_refined) if not isinstance(R0_refined, Zonotope) else R0_refined
    R0red = R0_zonotope.reduce(options.get('reductionTechnique', 'girard'), 
                               options.get('guardOrder', 2))
    
    R0red_ = R0red + (-R0red.center())
    R0_ = R0_refined + (-R0_refined.center())
    
    # error due to abstraction to state-dependent constant flow 
    # (see Sec. 5.3 in [1])
    err = _aux_abstractionError(A, U, R0red, th, tmax, options.get('taylorTerms', 10))
    
    # first part y_h of the mapped set (see Prop. 3 in [1])
    k, L, Q, phi = _aux_taylorSeriesParam(guard, A, b, R0red)
    
    res1 = k + L @ R0_ + 0.5 * phi * R0red_.quadMap(Q)
    
    # second part R_he of the mapped set (see (15) in [1])
    res2 = _aux_mappedSetError(guard, R0_refined, A, b, err)
    
    # overall mapped set
    R = res1 + res2
    
    # project set onto the hyperplane
    R = R.reduce(options.get('reductionTechnique', 'girard'), 
                options.get('zonotopeOrder', 2))
    
    R = R.projectOnHyperplane(guard)
    
    return R


# Auxiliary functions -----------------------------------------------------

def _aux_refinedIntersectionTime(loc: Any, guard: Polytope, R0: Any, 
                                  params: Dict[str, Any], options: Dict[str, Any]) -> Tuple[Any, float, float, Any]:
    """
    Compute the reachable set with a smaller time step to refine the time
    at which the reachable set intersects the guard set
    """
    
    # init halfspace representing the region inside the invariant
    P = Polytope(A_eq=guard.Ae, b_eq=guard.be)
    
    if not P.contains_(R0.center(), 'exact', 1e-12, 0, False, False):
        P = Polytope(A_eq=-guard.Ae, b_eq=-guard.be)
    
    spec = Specification(P, 'invariant')
    
    # adapt reachability options
    options_refined = options.copy()
    if hasattr(loc.contDynamics, '__class__') and loc.contDynamics.__class__.__name__ == 'linParamSys':
        options_refined['compTimePoint'] = True
    
    # set new parameters for next reach call
    options_refined['timeStep'] = 0.1 * options.get('timeStep', 0.01)
    params_refined = params.copy()
    params_refined['R0'] = R0
    params_refined['tStart'] = 0
    
    # compute reachable set until it fully crossed the hyperplane
    R = loc.contDynamics.reach(params_refined, options_refined, spec)
    
    # extract last time
    tmax = R.timePoint['time'][-1]
    
    # compute minimum time and set
    Rmin = R0
    tmin = 0.0
    found = False
    I = Interval.empty(R.timeInterval['set'][0].dim())
    
    for k in range(len(R.timeInterval['set'])):
        
        # check if start set of step intersects
        if not found and P.contains_(R.timePoint['set'][k], 'exact', 1e-12, 0, False, False):
            # update minimum time
            Rmin = R.timePoint['set'][k]
            tmin = R.timePoint['time'][k]
        else:
            # compute union of all sets that intersect the guard
            I_set = Interval(R.timeInterval['set'][k])
            I = I.or_op(I_set) if hasattr(I, 'or_op') else I | I_set
            found = True
    
    return Rmin, tmin, tmax, R


def _aux_abstractionError(A: np.ndarray, U: Any, R0: Zonotope, th: float, 
                          tmax: float, order: int) -> Zonotope:
    """
    Compute the set of abstractions errors due to the abstraction to state
    dependent constant flow according to Sec. 5.3 in [1]
    """
    
    # remainder for e^(At) due to finite taylor series (see (6) in [1])
    A_abs = np.abs(A)
    M = np.eye(len(A))
    M_ = M.copy()
    
    for i in range(1, order + 1):
        M_ = M_ @ A_abs * tmax / i
        M = M + M_
    
    from scipy.linalg import expm
    W = expm(A_abs * tmax) - M
    W = np.abs(W)
    
    e_hat = Interval(-W.flatten(), W.flatten())
    
    # compute powers of time
    tau_pow = [None] * order
    th_pow = [None] * order
    
    for i in range(order):
        th_pow[i] = th ** (i + 1)
        tau_pow[i] = Interval(0, tmax ** (i + 1))
    
    # split sets into center and remainder
    xi = R0.center()
    u = U.center() if hasattr(U, 'center') else np.zeros((A.shape[0], 1))
    U_ = U + (-u) if hasattr(U, '__add__') else U - u
    
    # first term of the error set (see (13) in [1])
    err1 = Zonotope(np.zeros((len(xi), 1)), np.array([]).reshape(len(xi), 0))
    M = A.copy()
    
    for i in range(2, order + 1):
        M = M @ A / i
        err1_term = M @ (tau_pow[i - 2] * R0 + (-th_pow[i - 2] * xi))
        err1 = err1 + err1_term
    
    # second term of the error set (see (13) in [1])
    err2 = Zonotope(np.zeros((len(xi), 1)), np.array([]).reshape(len(xi), 0))
    M = np.eye(len(A))
    
    for i in range(order):
        M = M @ A / (i + 2)
        err2 = err2 + M @ ((tau_pow[i] - th_pow[i]) * u)
    
    # third term of error set (due to set of uncertain inputs)
    M = np.eye(len(A))
    err3 = U_
    
    for i in range(order):
        M = M @ A / (i + 2)
        err3 = err3 + M @ (tau_pow[i] * U_)
    
    # overall error set (see (13) in [1])
    err = tau_pow[0] * (err1 + err2 + err3) + e_hat * R0 + e_hat * tmax * U
    
    return err


def _aux_systemParams(sys: Any, Rcont: Any, params: Dict[str, Any]) -> Tuple[np.ndarray, Any]:
    """
    Get the system matrix A and the set of uncertain inputs U
    """
    
    if hasattr(sys, '__class__') and sys.__class__.__name__ == 'linearSys':
        
        # extract system matrix + set of uncertain inputs
        A = sys.A
        U = sys.B @ params.get('U', Zonotope(np.zeros((sys.B.shape[1], 1)), np.array([]).reshape(sys.B.shape[1], 0)))
        
        if hasattr(sys, 'c') and sys.c is not None and sys.c.size > 0:
            U = U + sys.c
        
    elif hasattr(sys, '__class__') and sys.__class__.__name__ == 'nonlinearSys':
        
        # linearize the system
        c = Rcont.center()
        u = params.get('U', Zonotope(np.zeros((1, 1)), np.array([]).reshape(1, 0))).center() if hasattr(params.get('U', None), 'center') else np.zeros((1, 1))
        
        f = sys.mFile(c, u)
        A, B = sys.jacobian(c, u)
        
        # compute linearization error according to Prop. 1 in [2]
        int_x = Rcont
        int_x_ = int_x + (-c)
        
        int_u = Interval(params.get('U', Zonotope(np.zeros((1, 1)), np.array([]).reshape(1, 0))))
        int_u_ = Interval(params.get('U', Zonotope(np.zeros((1, 1)), np.array([]).reshape(1, 0)))) + (-u)
        
        # compute set of Hessians
        H = sys.hessian(int_x, int_u)
        
        dx = np.maximum(np.abs(int_x_.infimum() if hasattr(int_x_, 'infimum') else int_x_.inf), 
                       np.abs(int_x_.supremum() if hasattr(int_x_, 'supremum') else int_x_.sup))
        du = np.maximum(np.abs(int_u_.infimum() if hasattr(int_u_, 'infimum') else int_u_.inf), 
                       np.abs(int_u_.supremum() if hasattr(int_u_, 'supremum') else int_u_.sup))
        dz = np.concatenate([dx, du])
        
        # initialize linearization error
        linError = np.zeros(len(H))
        
        # compute linearization error
        for i in range(len(H)):
            H_ = np.abs(H[i])
            H_ = np.maximum(H_.infimum() if hasattr(H_, 'infimum') else H_.inf, 
                           H_.supremum() if hasattr(H_, 'supremum') else H_.sup)
            linError[i] = 0.5 * dz.T @ H_ @ dz
        
        linError = Zonotope(np.zeros_like(linError), np.diag(linError))
        
        # add linearization error to the set of uncertain inputs
        U = B @ params.get('U', Zonotope(np.zeros((B.shape[1], 1)), np.array([]).reshape(B.shape[1], 0))) + (f - A @ c) + linError
        
    else:
        raise CORAerror('CORA:specialError',
                       'Hyperplane mapping is only implemented for the classes "linearSys" and "nonlinearSys".')
    
    return A, U


def _aux_mappedSetError(guard: Polytope, R0: Any, A: np.ndarray, b: np.ndarray, err: Any) -> Any:
    """
    Compute the second part R_he of the mapped set (see (15) in [1])
    """
    
    # obtain object properties
    n = guard.Ae.T  # hyperplane normal vector (transpose to get column vector)
    
    # interval enclosure of the fraction
    temp = A @ R0 + b
    I = Interval(-n @ err) / Interval(n @ temp)
    
    # overall set (see (15) in [1])
    R = I * temp + err
    
    return R


def _aux_constantFlow(A: np.ndarray, R0: Any, U: Any, th: float, order: int) -> np.ndarray:
    """
    Compute constant part b of the flow \dot y = A*y_0 + b according to 
    Prop. 1 in [1]
    """
    
    # compute matrix Theta(th)
    Theta = np.zeros_like(A)
    M = A.copy()
    
    for i in range(2, order + 1):
        M = M @ A * th / i
        Theta = Theta + M
    
    # compute matrix Gamma(th)
    Gamma = np.eye(len(A))
    M = np.eye(len(A))
    
    for i in range(1, order + 1):
        M = M @ A * th / (i + 1)
        Gamma = Gamma + M
    
    # compute constant flow vector
    x0 = R0.center()
    u = U.center() if hasattr(U, 'center') else np.zeros((A.shape[0], 1))
    b = Theta @ x0 + Gamma @ u
    
    return b


def _aux_taylorSeriesParam(guard: Polytope, A: np.ndarray, b: np.ndarray, R0: Zonotope) -> Tuple[np.ndarray, np.ndarray, list, Interval]:
    """
    Computes the coefficients of the second order taylor series 
    
      y_i \in k_i + L_i (x-x*) + 0.5*phi*(x-x*)'*Q_i*(x-x*)
    
    according to Prop. 3 in [1]
    """
    
    # obtain object properties
    n = guard.Ae.T  # hyperplane normal vector (transpose to get column vector)
    d = guard.be  # hyperplane offset
    
    # auxiliary variables (see Prop. 2 in [1])
    x0 = R0.center()
    
    Lambda = n.T @ (A @ x0 + b)
    Upsilon = (n.T @ A).T
    Theta = -n @ Lambda - (d - n.T @ x0) * Upsilon
    Omega = -n @ Upsilon.T + Upsilon @ n.T
    
    # interval enclosure of Theta/Lambda (see (19) in [1])
    Lambda_zono = n.T @ (A @ R0 + b)
    Lambda_int = Interval(Lambda_zono)
    Theta_int = Interval(-n @ (n.T @ (A @ R0 + b)) + (-1) * (d + (-1) * (n.T @ R0)) * Upsilon)
    
    temp = Theta_int / Lambda_int
    
    psi_c = temp.center() if hasattr(temp, 'center') else (temp.inf + temp.sup) / 2
    psi_g = temp.rad() if hasattr(temp, 'rad') else (temp.sup - temp.inf) / 2
    
    # interval enclosure phi of the set 1/Lambda^2 (see Prop. 3 in [1])
    phi = 1 / (Lambda_int ** 2)
    
    # matrix zonotope for Theta (see (18) in [1])
    Lambda_zono = n.T @ (A @ R0 + b)
    Theta_aux_zono = (-1) * (d + (-1) * (n.T @ R0)) * Upsilon
    
    Theta_aux_mat = np.hstack([Theta_aux_zono.c.reshape(-1, 1), Theta_aux_zono.G])
    Lambda_mat = np.hstack([Lambda_zono.c.reshape(-1, 1), Lambda_zono.G])
    
    Theta_mat = -n @ Lambda_mat + Theta_aux_mat
    
    # constant vector k
    k = x0 + (A @ x0 + b) * (d - n.T @ x0) / Lambda
    
    # linear map L
    L = np.eye(len(A)) + \
        A * (d - n.T @ x0) / Lambda + \
        (A @ x0 + b) @ Theta.T / (Lambda ** 2)
    
    # quadratic map Q(i,l,m) (see Prop. 3 in [1])
    temp_zono = A @ R0 + b
    c = temp_zono.center()
    G = temp_zono.G
    gens = G.shape[1]
    
    Q = [None] * len(b)
    
    for i in range(len(b)):
        
        # compute center matrix
        c_Q = A[i, :].T @ Theta_mat[:, 0].T + Theta_mat[:, 0] @ A[i, :].T + \
              c[i] * (Omega - psi_c * 2 * Upsilon.T)
        c_Q_rad = c[i] * (-psi_g * 2 * Upsilon.T)
        
        # compute generator matrices
        g_Q = np.zeros((c_Q.shape[0], c_Q.shape[1], gens))
        g_Q_rad = np.zeros((c_Q.shape[0], c_Q.shape[1], gens))
        
        for iGen in range(gens):
            g_Q[:, :, iGen] = A[i, :].T @ Theta_mat[:, 1 + iGen].T + \
                              Theta_mat[:, 1 + iGen] @ A[i, :].T + \
                              G[i, iGen] * (Omega - psi_c * 2 * Upsilon.T)
        
        for iGen in range(gens):
            g_Q_rad[:, :, iGen] = G[i, iGen] * (-psi_g * 2 * Upsilon.T)
        
        Q_prep = np.concatenate([c_Q_rad[:, :, np.newaxis], g_Q, g_Q_rad[:, :, np.newaxis]], axis=2)
        
        # generate matrix zonotope
        Q[i] = matZonotope(c_Q, Q_prep)
    
    return k, L, Q, phi

