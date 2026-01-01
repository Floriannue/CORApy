"""
checkFlow - remove all intersections for which the flow of the system
   does not point toward the guard set (see Sec. 4.4.4 in [1])

Syntax:
    [res,R] = checkFlow(loc,guard,R,params)

Inputs:
    loc - location object
    guard - guard set (class: polytope, levelSet)
    R - list of reachable sets
    params - model parameters

Outputs:
    res - true/false whether flow points toward the guard for any of the
          intersecting reachable sets
    R - updated list of reachable sets

References:
   [1] N. Kochdumper. "Extensions to Polynomial Zonotopes and their
       Application to Verifcation of Cyber-Physical Systems", PhD Thesis 

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: location/guardIntersect

Authors:       Niklas Kochdumper
Written:       23-December-2019             
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import Any, List, Tuple, Optional
from cora_python.contSet.polytope import Polytope
from cora_python.contSet.levelSet import LevelSet
from cora_python.contSet.contSet.projectOnHyperplane import projectOnHyperplane
from cora_python.g.functions.matlab.init.gramSchmidt import gramSchmidt
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def checkFlow(loc: Any, guard: Any, R: List[Any], params: dict) -> Tuple[bool, List[Any]]:
    """
    Remove all intersections for which the flow of the system does not point toward the guard set
    
    Args:
        loc: location object
        guard: guard set (polytope or levelSet)
        R: list of reachable sets
        params: model parameters
        
    Returns:
        res: true/false whether flow points toward the guard for any of the intersecting reachable sets
        R: updated list of reachable sets
    """
    
    # read out continuous dynamics
    sys = loc.contDynamics
    
    # adapt the guard set such that the normal vector of the hyperplane
    # points toward the outside of the invariant set
    outside = None
    if isinstance(guard, Polytope) and guard.representsa_('conHyperplane', 1e-12):
        guard = _aux_adaptGuard(loc, guard, R)
    elif isinstance(guard, LevelSet):
        outside = _aux_getOutside(loc, guard, R)
    else:
        raise CORAerror('CORA:notSupported',
                      'Requires levelSet or polytope representing constrained hyperplane.')
    
    # loop over all reachable sets
    R_ = [None] * len(R)
    counter = 0
    
    for i in range(len(R)):
        
        # check if the flow points toward the guard set
        if isinstance(guard, Polytope) and guard.representsa_('conHyperplane', 1e-12):
            res = _aux_flowInDirection(sys, guard, R[i], params)
        elif isinstance(guard, LevelSet):
            res = _aux_flowInDirectionLevelSet(sys, guard, R[i], outside, params)
        else:
            res = False
        
        if res:
            R_[counter] = R[i]
            counter += 1
    
    # check if all sets are empty
    if counter > 0:
        res = True
        R = R_[:counter]
    else:
        res = False
        R = []
    
    return res, R


# Auxiliary functions -----------------------------------------------------

def _aux_flowInDirection(sys: Any, guard: Polytope, R: Any, params: dict) -> bool:
    """
    Check if the flow of the system points in the direction of the guard set
    
    Args:
        sys: continuous dynamics system
        guard: guard set (polytope representing constrained hyperplane)
        R: reachable set
        params: model parameters
        
    Returns:
        res: true if flow points toward guard, false otherwise
    """
    from cora_python.contSet.zonotope import Zonotope
    from cora_python.contSet.zonoBundle import ZonoBundle
    from cora_python.contSet.interval import Interval
    from cora_python.contDynamics.linearSys import LinearSys
    
    # get hyperplane normal direction
    c = guard.Ae / np.linalg.norm(guard.Ae)
    
    # project reachable set onto the hyperplane
    Rproj = projectOnHyperplane(R, guard)
    
    # fast check: check if the flow of the center points towards the guard
    params_fast = params.copy()
    params_fast['u'] = params['U'].center()
    params_fast['w'] = params.get('W', Zonotope(np.zeros((sys.nr_of_outputs, 1)))).center()
    
    fcnHan = sys.getfcn(params_fast)
    flow = fcnHan(0, Rproj.center())
    flow = c @ flow
    
    if flow > 0:
        return True
    
    # compute interval enclosure of the reachable set
    B = gramSchmidt(guard.Ae.T)
    
    if isinstance(Rproj, ZonoBundle):
        Rproj = Rproj.Z[0]
    
    if not isinstance(Rproj, Zonotope):
        Rproj = Zonotope(Rproj)
    
    R_reduced = B @ (B.T @ Rproj).reduce('pca', 1)
    
    # compute flow in hyperplane normal direction
    if isinstance(sys, LinearSys):
        cen = R_reduced.center()
        G = R_reduced.generators()
        I = Interval(-np.ones((G.shape[1], 1)), np.ones((G.shape[1], 1)))
        cA = c @ sys.A
        U_int = Interval(params['U'])
        flow = cA @ cen + (cA @ G) @ I + (c @ sys.B) @ U_int + sys.c
    else:
        # For nonlinear systems, use Taylor models
        from cora_python.contSet.zonotope.taylm import taylm
        from cora_python.contSet.taylm import Taylm
        
        tay = taylm(R_reduced)
        params_tay = params.copy()
        U_int = Interval(params['U'])
        params_tay['u'] = Taylm(U_int, 4, 'u')
        fcnHan = sys.getfcn(params_tay)
        flow = fcnHan(0, tay)
        flow = c @ flow
    
    # check if the flow points in the direction of the guard set
    flow_int = Interval(flow) if not isinstance(flow, Interval) else flow
    res = flow_int.supremum() > 0
    
    return res


def _aux_flowInDirectionLevelSet(sys: Any, guard: LevelSet, R: Any, outside: int, params: dict) -> bool:
    """
    Check if the flow of the system points in the direction of the guard set (level set)
    
    Args:
        sys: continuous dynamics system
        guard: guard set (levelSet)
        R: reachable set
        outside: direction indicator (1 or -1)
        params: model parameters
        
    Returns:
        res: true if flow points toward guard, false otherwise
    """
    from cora_python.contSet.zonotope import Zonotope
    from cora_python.contSet.interval import Interval
    from cora_python.contSet.zonotope.taylm import taylm
    from cora_python.contSet.taylm import Taylm
    
    # fast check: check if the flow of the center points towards the guard
    params_fast = params.copy()
    params_fast['u'] = params['U'].center()
    params_fast['w'] = params.get('W', Zonotope(np.zeros((sys.nr_of_outputs, 1)))).center()
    
    c = R.center()
    fcnHan = sys.getfcn(params_fast)
    flow = outside * guard.der.grad(c).T @ fcnHan(0, c)
    
    if flow > 0:
        return True
    
    # compute interval enclosure of the reachable set
    if not isinstance(R, Zonotope):
        R = Zonotope(R)
    
    R_reduced = R.reduce('pca', 1)
    
    # compute flow in hyperplane normal direction
    tay = taylm(R_reduced)
    params_tay = params.copy()
    U_int = Interval(params['U'])
    params_tay['u'] = Taylm(U_int, 4, 'u')
    fcnHan = sys.getfcn(params_tay)
    flow = outside * (guard.der.grad(tay).T @ fcnHan(0, tay))
    
    # check if the flow points in the direction of the guard set
    flow_int = Interval(flow) if not isinstance(flow, Interval) else flow
    res = flow_int.supremum() > 0
    
    return res


def _aux_adaptGuard(loc: Any, guard: Polytope, R: List[Any]) -> Polytope:
    """
    Adapt the guard set such that the normal vector of the hyperplane
    points toward the outside of the invariant set
    
    Args:
        loc: location object
        guard: guard set (polytope)
        R: list of reachable sets
        
    Returns:
        guard: adapted guard set
    """
    from cora_python.contSet.polytope import Polytope
    
    P = Polytope(A_eq=guard.Ae, b_eq=guard.be)
    inv = loc.invariant
    
    # get center of the first reachable set
    if isinstance(R[0], list):
        c = R[0][0].center()
    else:
        c = R[0].center()
    
    # check if center is on the correct side of the hyperplane
    if inv.contains_(c, 'exact', 100*np.finfo(float).eps, 0, False, False):
        if not P.contains_(c, 'exact', 100*np.finfo(float).eps, 0, False, False):
            guard = Polytope(A_eq=-guard.Ae, b_eq=-guard.be)
    else:
        if P.contains_(c, 'exact', 100*np.finfo(float).eps, 0, False, False):
            guard = Polytope(A_eq=-guard.Ae, b_eq=-guard.be)
    
    return guard


def _aux_getOutside(loc: Any, guard: LevelSet, R: List[Any]) -> int:
    """
    Determine which side of the guard set is the outside of the invariant set
    g(x) >= 0 is outside => outside = 1
    g(x) < 0 is outside => outside = -1
    
    Args:
        loc: location object
        guard: guard set (levelSet)
        R: list of reachable sets
        
    Returns:
        outside: direction indicator (1 or -1)
    """
    inv = loc.invariant
    outside = 1
    
    # get center of the first reachable set
    if isinstance(R[0], list):
        c = R[0][0].center()
    else:
        c = R[0].center()
    
    # check if center is on the correct side of the hyperplane
    if inv.contains_(c, 'exact', 100*np.finfo(float).eps, 0, False, False):
        if guard.funHan(c) > 0:
            outside = -1
    else:
        if guard.funHan(c) <= 0:
            outside = -1
    
    return outside

