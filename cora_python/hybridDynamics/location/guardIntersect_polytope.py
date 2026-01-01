"""
guardIntersect_polytope - enclosure of guard intersections based on
   using a combination of zonotopes and polytopes as described in [1]

Syntax:
    R = guardIntersect_polytope(loc,R,options)

Inputs:
    loc - location object
    R - list of intersections between the reachable set and the guard
    guard - guard set
    options - struct containing the algorithm settings

Outputs:
    R - enclosure of the guard intersection

References: 
  [1] M. Althoff et al. "Computing Reachable Sets of Hybrid Systems Using 
      a Combination of Zonotopes and Polytopes", 2009
  [2] M. Althoff et al. "Zonotope bundles for the efficient computation 
      of reachable sets", 2011

Authors:       Matthias Althoff, Niklas Kochdumper
Written:       19-December-2019
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import Any, List, Dict
from cora_python.contSet.polytope import Polytope
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.zonoBundle import ZonoBundle
from cora_python.contSet.interval import Interval
from cora_python.g.functions.matlab.init.gramSchmidt import gramSchmidt
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def guardIntersect_polytope(loc: Any, R: List[Any], guard: Polytope, 
                           options: Dict[str, Any]) -> Any:
    """
    Enclosure of guard intersections based on using a combination of zonotopes and polytopes
    
    Args:
        loc: location object
        R: list of intersections between the reachable set and the guard
        guard: guard set (polytope)
        options: struct containing the algorithm settings (must contain 'enclose')
        
    Returns:
        R: enclosure of the guard intersection (zonotope or zonoBundle)
    """
    
    # enclose all relevant reachable sets by polytopes
    for i in range(len(R)):
        R[i] = _aux_conv2polytope(R[i])
    
    # intersect the reachable sets with the guard set
    for i in range(len(R)):
        # MATLAB: R{i} = and_(R{i},guard,'exact');
        R[i] = R[i].and_(guard, 'exact')
    
    # compute vertices
    V = np.array([]).reshape(0, 0)
    for i in range(len(R)):
        # MATLAB: vert = vertices(R{i});
        if hasattr(R[i], 'vertices'):
            vert = R[i].vertices()
        else:
            from cora_python.contSet.contSet.vertices import vertices
            vert = vertices(R[i])
        
        # MATLAB: V = [V,vert];
        if V.size == 0:
            V = vert
        else:
            V = np.hstack([V, vert])
    
    # enclose vertices with the methods described in Section V.A in [2]
    m = len(options['enclose'])
    Z = [None] * m
    
    for i in range(m):
        
        enclose_method = options['enclose'][i]
        
        if enclose_method == 'box':
            # Box method as described in Section V.A.a) in [2]
            # MATLAB: Z{i} = zonotope(interval.enclosePoints(V));
            I = Interval.enclosePoints(V)
            Z[i] = Zonotope(I)
            
        elif enclose_method == 'pca':
            # PCA method as described in Section V.A.b) in [2]
            # MATLAB: Z{i} = zonotope.enclosePoints(V,'stursberg');
            Z[i] = Zonotope.enclosePoints(V, 'stursberg')
            
        elif enclose_method == 'flow':
            # Flow method as described in Section V.A.d) in [2]
            # MATLAB: Z{i} = aux_flowEnclosure(loc.contDynamics,V,options);
            Z[i] = _aux_flowEnclosure(loc.contDynamics, V, options)
            
        else:
            raise CORAerror('CORA:wrongFieldValue', 'options.enclose',
                          ['box', 'pca', 'flow'])
    
    # construct the enclosing zonotope or zonotope bundle object
    if len(Z) == 1:
        R = Z[0]
    else:
        R = ZonoBundle(Z)
    
    return R


# Auxiliary functions -----------------------------------------------------

def _aux_flowEnclosure(sys: Any, V: np.ndarray, options: Dict[str, Any]) -> Zonotope:
    """
    Flow method for enclosing a set of vertices with a zonotope as described 
    in Section V.A.d) in [1]
    
    Args:
        sys: continuous dynamics system object
        V: matrix of vertices (each column is a vertex)
        options: struct containing algorithm settings
        
    Returns:
        Z: zonotope enclosing the vertices
    """
    
    # compute flow at center of vertices
    # MATLAB: c = mean(V,2);
    c = np.mean(V, axis=1, keepdims=True)
    
    # MATLAB: options.u = options.uTrans;
    # Create options copy with u set
    options_with_u = options.copy()
    options_with_u['u'] = options.get('uTrans', np.zeros((sys.B.shape[1] if hasattr(sys, 'B') else 0, 1)))
    
    # MATLAB: fcnHan = getfcn(sys,options);
    # MATLAB: dir = fcnHan(0,c);
    # Get function handle from system
    if hasattr(sys, 'getfcn'):
        fcnHan = sys.getfcn(options_with_u)
    else:
        # Fallback: try to construct function handle manually
        # For linearSys: f(t,x) = A*x + B*u + c + E*w
        if hasattr(sys, 'A') and hasattr(sys, 'B'):
            u = options_with_u.get('u', np.zeros((sys.B.shape[1], 1)))
            w = options_with_u.get('w', np.zeros((sys.E.shape[1] if hasattr(sys, 'E') and sys.E.size > 0 else 0, 1)))
            def fcnHan(t, x):
                x_arr = np.asarray(x).reshape(-1, 1)
                result = sys.A @ x_arr
                if sys.B.size > 0:
                    result += sys.B @ u
                if hasattr(sys, 'c') and sys.c.size > 0:
                    result += sys.c.reshape(-1, 1)
                if hasattr(sys, 'E') and sys.E.size > 0:
                    result += sys.E @ w
                return result.flatten()
        elif hasattr(sys, 'mFile'):
            # For nonlinearSys: f(t,x) = mFile(x, u)
            u = options_with_u.get('u', None)
            def fcnHan(t, x):
                return sys.mFile(x, u)
        else:
            raise CORAerror('CORA:notSupported', 'Cannot get function handle from system')
    
    dir_vec = np.asarray(fcnHan(0, c.flatten()))
    dir_vec = dir_vec.reshape(-1, 1)  # Ensure column vector
    
    # MATLAB: dir = dir./norm(dir);
    dir_vec = dir_vec / np.linalg.norm(dir_vec)
    
    # get basis that is orthogonal to the flow direction
    # MATLAB: B = gramSchmidt(dir);
    B = gramSchmidt(dir_vec)
    
    # compute box enclosure in transformed space
    # MATLAB: Z = B*zonotope(interval.enclosePoints(B'*V));
    V_transformed = B.T @ V
    I_transformed = Interval.enclosePoints(V_transformed)
    Z_transformed = Zonotope(I_transformed)
    Z = B @ Z_transformed
    
    return Z


def _aux_conv2polytope(R: Any) -> Polytope:
    """
    Compute an enclosing polytope for the given set
    
    Args:
        R: set object (zonotope, interval, etc.)
        
    Returns:
        P: polytope enclosing the set
    """
    
    # enclose set with zonotope
    # MATLAB: if ~isa(R,'zonotope')
    # MATLAB:     R = zonotope(R);
    if not isinstance(R, Zonotope):
        R = Zonotope(R)
    
    # reduce the given set with different methods
    # MATLAB: Zred1 = reduce(R,'pca',1);
    Zred1 = R.reduce('pca', 1)
    
    # MATLAB: Zred2 = zonotope(interval(R));
    if hasattr(R, 'interval'):
        I_R = R.interval()
    else:
        I_R = Interval(R)
    Zred2 = Zonotope(I_R)
    
    # compute intersection of the two reduced sets
    # MATLAB: P = and_(polytope(Zred1),polytope(Zred2),'exact');
    if hasattr(Zred1, 'polytope'):
        P1 = Zred1.polytope()
    else:
        from cora_python.contSet.zonotope.polytope import polytope
        P1 = polytope(Zred1)
    
    if hasattr(Zred2, 'polytope'):
        P2 = Zred2.polytope()
    else:
        from cora_python.contSet.zonotope.polytope import polytope
        P2 = polytope(Zred2)
    
    P = P1.and_(P2, 'exact')
    
    return P

