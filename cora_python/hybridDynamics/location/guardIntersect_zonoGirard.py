"""
guardIntersect_zonoGirard - implementation of the zonotope-hyperplane
   intersection approach described in [1]

Syntax:
    R = guardIntersect_zonoGirard(loc,R,guard,B)

Inputs:
    loc - location object
    R - list of intersections between the reachable set and the guard
    guard - guard set (class: constrained hyperplane)
    B - basis

Outputs:
    R - set enclosing the guard intersection

References: 
  [1] A. Girard et al. "Zonotope/Hyperplane Intersection for Hybrid 
      Systems Reachablity Analysis"
  [2] M. Althoff et al. "Zonotope bundles for the efficient computation 
      of reachable sets", 2011

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Stefan Liu, Niklas Kochdumper
Written:       19-December-2016 
Last update:   18-May-2018 (NK, integration into CORA)
               19-December-2019 (NK, restructured the code)
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import Any, List, Optional, Tuple
from cora_python.contSet.polytope import Polytope
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.zonoBundle import ZonoBundle
from cora_python.contSet.interval import Interval
from cora_python.contSet.conZonotope import ConZonotope
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def guardIntersect_zonoGirard(loc: Any, R: List[Any], guard: Polytope, 
                              B: List[np.ndarray]) -> Any:
    """
    Implementation of the zonotope-hyperplane intersection approach described in [1]
    
    Args:
        loc: location object
        R: list of intersections between the reachable set and the guard
        guard: guard set (polytope representing constrained hyperplane)
        B: list of basis matrices
        
    Returns:
        R: set enclosing the guard intersection (zonotope, zonoBundle, or empty)
    """
    
    # check if guard set is a constrained hyperplane
    if not (isinstance(guard, Polytope) and guard.representsa_('conHyperplane', 1e-12)):
        raise CORAerror('CORA:specialError',
                      "The method 'zonoGirard' only supports guards given as polytope objects that represent constrained hyperplanes!")
    
    # construct polytope from guard set inequality constraints C*x <= d
    if guard.b.size > 0:
        poly = Polytope(A=guard.A, b=guard.b)
    else:
        poly = None
    
    # loop over all calculated basis
    Z = [None] * len(B)
    
    for i in range(len(B)):
        
        # loop over all reachable sets
        I = None
        for j in range(len(R)):
            
            # interval enclosure in the transformed space according to [1]
            intTemp = _aux_enclosingInterval(guard, B[i], R[j])
            
            # unite all intervals
            if j == 0:
                I = intTemp
            else:
                # MATLAB: I = I | intTemp
                I = I.or_op(intTemp) if hasattr(I, 'or_op') else I | intTemp
        
        # set for one basis is empty -> overall set is empty
        if I.representsa_('emptySet', np.finfo(float).eps):
            return []
        
        # transform back to original space
        # MATLAB: Z{i} = B{i}*zonotope(I);
        Z_i = Zonotope(I)
        Z[i] = B[i] @ Z_i
        
        # remove parts outside the guard sets inequality constraints
        if poly is not None and not poly.representsa_('emptySet', np.finfo(float).eps):
            Z[i] = _aux_tightenSet(Z[i], poly)
    
    # construct the enclosing zonotope bundle object
    Z = [z for z in Z if z is not None]
    
    if len(Z) == 0:
        return []
    elif len(Z) == 1:
        return Z[0]
    else:
        return ZonoBundle(Z)


# Auxiliary functions -----------------------------------------------------

def _aux_enclosingInterval(guard: Polytope, B: np.ndarray, Z: Any) -> Interval:
    """
    Implementation of Algorithm 2 in reference paper [1]
    
    Args:
        guard: guard set (polytope representing constrained hyperplane)
        B: basis matrix
        Z: reachable set (zonotope, zonoBundle, or other set)
        
    Returns:
        I: interval enclosure in transformed space
    """
    from cora_python.contSet.zonoBundle import ZonoBundle
    
    # enclose the set with a zonotope
    if not isinstance(Z, (Zonotope, ZonoBundle)):
        Z = Zonotope(Z)
    
    # get hyperplane normal vector and offset
    n = guard.Ae.T  # Transpose to get column vector
    gamma = guard.be
    
    # initialization
    lb = -np.inf * np.ones((len(n), 1))
    ub = np.inf * np.ones((len(n), 1))
    
    # loop over all basis vectors l
    for i in range(B.shape[1]):
        
        if isinstance(Z, ZonoBundle):
            # loop over all parallel sets
            for k in range(len(Z.Z)):
                # Generate two-dimensional zonogon
                ZZ = np.hstack([Z.Z[k].center(), Z.Z[k].generators()])
                SZ = np.vstack([(ZZ.T @ n).T, (ZZ.T @ B[:, i]).T])
                SZ[np.abs(SZ) < np.finfo(float).eps] = 0
                Z_2D = Zonotope(SZ)
                
                # Interval of intersection
                m, M = _aux_bound_intersect_2D(Z_2D, gamma)
                
                lb[i] = max(m, lb[i])
                ub[i] = min(M, ub[i])
        else:
            # Generate two-dimensional zonogon
            ZZ = np.hstack([Z.center(), Z.generators()])
            SZ = np.vstack([(ZZ.T @ n).T, (ZZ.T @ B[:, i]).T])
            SZ[np.abs(SZ) < np.finfo(float).eps] = 0
            Z_2D = Zonotope(SZ)
            
            # Interval of intersection
            m, M = _aux_bound_intersect_2D(Z_2D, gamma)
            
            lb[i] = m
            ub[i] = M
    
    # Convert to Zonotope
    lb, ub = _aux_robustProjection(B, n, gamma, lb, ub)
    
    # Increase robustness by equaling out small differences
    test = ub >= lb
    
    if not np.all(test):
        ind = np.where(test == 0)[0]
        ub[ind] = ub[ind] + np.finfo(float).eps * np.ones((len(ind), 1))
    
    if np.all(lb <= ub):
        I = Interval(lb, ub)
    else:
        I = Interval.empty(len(lb))
    
    return I


def _aux_bound_intersect_2D(Z: Zonotope, L: float) -> Tuple[float, float]:
    """
    Implementation of Algorithm 3 in [1]
    
    Args:
        Z: 2D zonotope
        L: vertical line parameter (L = {x,y: x = gamma})
        
    Returns:
        m: lower bound
        M: upper bound
    """
    
    Z = Z.compact_('zeros', np.finfo(float).eps)
    
    c = Z.center()
    g = Z.generators()
    
    r = g.shape[1]
    gamma = L  # L = {x,y: x = gamma}
    
    # Lower Bound ---------------------------------------------------------
    
    P = c.copy()  # current Position
    
    for i in range(r):
        if (g[1, i] < 0) or ((g[1, i] == 0) and g[0, i] < 0):
            g[:, i] = -g[:, i]  # ensure all generators are pointing upward
        P = P - g[:, i:i+1]  # drives P toward the lowest vertex of Z
    
    # Direction of
    if P[0] < gamma:
        dir_val = 1
        G = _aux_sort_trig(g, dir_val)  # we should look right
    else:
        dir_val = -1
        G = _aux_sort_trig(g, dir_val)  # or left
    dir_val = -dir_val
    
    s = np.sum(2 * G, axis=1, keepdims=True)
    
    m = _aux_dichotomicSearch(P, G, s, gamma, dir_val, Z)
    
    # Upper bound ---------------------------------------------------------
    
    P = c.copy()  # current Position
    g = Z.generators()  # Reset g
    
    for i in range(r):
        if (g[1, i] < 0) or ((g[1, i] == 0) and g[0, i] < 0):
            g[:, i] = -g[:, i]  # ensure all generators are pointing upward
        P = P + g[:, i:i+1]  # drives P toward the uppest vertex of Z
    
    if P[0] > gamma:
        dir_val = 1
        G = -1 * _aux_sort_trig(g, dir_val)  # we should look left
    else:
        dir_val = -1
        G = -1 * _aux_sort_trig(g, dir_val)  # or right
    
    s = np.sum(2 * G, axis=1, keepdims=True)
    
    M = _aux_dichotomicSearch(P, G, s, gamma, dir_val, Z)
    
    if (abs(m) < 1e-12) and (abs(M) < 1e-12):  # prevent numerical error
        m = 0
        M = 0
    elif abs(m - M) < 1e-12:
        M = m
    
    return m, M


def _aux_sort_trig(g: np.ndarray, dir_val: int) -> np.ndarray:
    """
    Sort g according to angle of every vector
    
    Args:
        g: generator matrix (2 x n)
        dir_val: direction (1 or -1)
        
    Returns:
        G: sorted generator matrix
    """
    # MATLAB: theta = cart2pol(g(1,:),g(2,:))
    # cart2pol(x, y) returns angle in radians: atan2(y, x)
    theta = np.arctan2(g[1, :], g[0, :])
    
    if dir_val == 1:
        I = np.argsort(theta)
    elif dir_val == -1:
        I = np.argsort(theta)[::-1]  # descending order
    
    G = g[:, I]
    return G


def _aux_split_pivot(G: np.ndarray, s: np.ndarray, dir_val: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split the set of generators at the pivot element (= angle)
    
    Args:
        G: generator matrix
        s: sum vector
        dir_val: direction
        
    Returns:
        G1: first part of generators
        G2: second part of generators
    """
    
    cos_G = dir_val * G[0, :] / np.sqrt(G[0, :]**2 + G[1, :]**2)
    pivot = dir_val * s[0] / np.sqrt(s[0]**2 + s[1]**2)
    
    G1 = G[:, cos_G <= pivot]
    G2 = G[:, cos_G > pivot]
    
    return G1, G2


def _aux_dichotomicSearch(P_0: np.ndarray, G_0: np.ndarray, s_0: np.ndarray, 
                          gamma: float, dir_val: int, Z: Zonotope, 
                          queue: Optional[List] = None) -> float:
    """
    Search the intersection between a 2D-zonotope and a line
    
    Args:
        P_0: initial position
        G_0: initial generators
        s_0: initial sum vector
        gamma: vertical line parameter
        dir_val: direction
        Z: zonotope
        queue: search queue (optional)
        
    Returns:
        m: intersection value
    """
    
    if queue is None:
        queue = []
    else:
        if len(queue) > 0:
            queue = queue[1:]  # Remove first element
    
    P = P_0.copy()
    G = G_0.copy()
    s = s_0.copy()
    
    counter = 0
    max_counter = G.shape[1]
    
    while G.shape[1] > 1 and counter < max_counter:
        G1, G2 = _aux_split_pivot(G, s, dir_val)
        s1 = np.sum(2 * G1, axis=1, keepdims=True)
        line_int_y = _aux_lineIntersect2D(P, P + s1, gamma)
        
        if line_int_y is not None:  # exist intersection
            G = G1
            s = s1
        else:
            # save data for search
            new_entry = [P.copy(), G1.copy(), s1.copy()]
            queue.append(new_entry)
            
            G = G2
            s = s - s1
            P = P + s1
        
        counter += 1
    
    # only one generator remains
    m = _aux_lineIntersect2D(P, P + s, gamma)
    
    if m is None:
        if len(queue) > 0:
            m, _, _ = _aux_dichotomicSearch(
                queue[0][0], queue[0][1], queue[0][2], gamma, dir_val, Z, queue)
        else:
            # No intersection found
            return np.nan
    else:
        # overapproximation and underapproximation are not used in the return
        # but computed for consistency with MATLAB
        m_over = P[1]
        m_unde = P[1] + s[1]
    
    return m


def _aux_lineIntersect2D(p1: np.ndarray, p2: np.ndarray, gamma: float) -> Optional[float]:
    """
    Calculate the intersection between a line segment and a vertical line
    
    Args:
        p1: first point (2D)
        p2: second point (2D)
        gamma: vertical line parameter (x = gamma)
        
    Returns:
        y: y-coordinate of intersection, or None if no intersection
    """
    eps = np.finfo(float).eps
    
    # check if gamma is one of the points
    if abs(p1[0] - gamma) < eps:
        return float(p1[1])
    elif abs(p2[0] - gamma) < eps:
        return float(p2[1])
    
    # check if line is vertical
    if p1[0] == p2[0]:
        return None
    
    # check if line is horizontal
    if p1[1] == p2[1]:
        if gamma > min(p1[0], p2[0]) and gamma < max(p1[0], p2[0]):
            return float(p2[1])
        else:
            return None
    
    # calculate parameter of line y = a*x + b
    a = (p1[1] - p2[1]) / (p1[0] - p2[0])
    b = p1[1] - a * p1[0]
    
    # calculate intersection with vertical line x = gamma
    y = a * gamma + b
    
    # check if the intersection is located inside the line segment
    if y > max(p1[1], p2[1]) or y < min(p1[1], p2[1]):
        return None
    
    return float(y)


def _aux_robustProjection(D: np.ndarray, n: np.ndarray, gamma: float, 
                          lb: np.ndarray, ub: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    If the basis D is orthogonal to the hyperplane, set the interval value in
    the corresponding dimension to the hyperplane parameter to increase the
    numerical stability
    
    Args:
        D: basis matrix
        n: hyperplane normal vector
        gamma: hyperplane offset
        lb: lower bounds
        ub: upper bounds
        
    Returns:
        lb: updated lower bounds
        ub: updated upper bounds
    """
    
    # project to modified space
    n_ = D.T @ n
    
    # check if the basis is orthogonal to the hyperplane (with tolerance)
    ind = np.where(np.abs(n_) >= np.finfo(float).eps)[0]
    
    # hyperplane normal vector perpendicular to orthogonal basis
    # -> set the corresponding interval dimension to the hyperplane value
    if len(ind) == 1:
        val = gamma * n_[ind[0]]
        lb[ind[0]] = val
        ub[ind[0]] = val
    
    return lb, ub


def _aux_tightenSet(Z: Zonotope, P: Polytope) -> Optional[Zonotope]:
    """
    Remove the parts of the interval that are located outside the inequality
    constraints C*x <= d of the constrained hyperplane
    
    Args:
        Z: zonotope
        P: polytope with inequality constraints
        
    Returns:
        Z: tightened zonotope, or None if empty
    """
    
    # check if the zonotope fulfills the constraints
    if not P.isIntersecting_(Z, 'approx', 1e-8):
        return None
    
    # convert the zonotope to a constrained zonotope
    cZ = ConZonotope(Z)
    
    # intersect the constrained zonotope with the polytope
    cZ = cZ.and_(P, 'exact')
    
    # tighten the domain of the zonotope factors
    # MATLAB: try cZ = rescale(cZ,'iter'); catch cZ = rescale(cZ,'optimize'); end
    try:
        cZ = cZ.rescale('iter')
    except:
        cZ = cZ.rescale('optimize')
    
    # extract the rescaled zonotope from the constrained zonotope
    Z = Zonotope(cZ.c, cZ.G)
    
    return Z

