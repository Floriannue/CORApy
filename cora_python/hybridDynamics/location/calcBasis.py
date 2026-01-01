"""
calcBasis - calculate orthogonal basis with the methods from [1]

Syntax:
    B = calcBasis(loc,R,guard,options,params)

Inputs:
    loc - object of class location
    R - list of intersections between the reachable set and the guard
    guard - guard set (class: constrained hyperplane)
    options - struct containing the algorithm settings
    params - model parameters for evaluating dynamic equation
             (only options.enclose = 'flow')

Outputs:
    B - list containing the calculated basis

References: 
  [1] M. Althoff et al. "Zonotope bundles for the efficient computation 
      of reachable sets", 2011

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: location/guardIntersect

Authors:       Niklas Kochdumper
Written:       05-November-2018             
Last update:   20-November-2019
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import Any, Dict, List, Optional
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.polytope import Polytope
from cora_python.contSet.contSet.projectOnHyperplane import projectOnHyperplane
from cora_python.g.functions.matlab.init.gramSchmidt import gramSchmidt
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def calcBasis(loc: Any, R: List[Any], guard: Any, options: Dict[str, Any], 
              params: Optional[Dict[str, Any]] = None) -> List[np.ndarray]:
    """
    Calculate orthogonal basis with the methods from [1]
    
    Args:
        loc: location object
        R: list of intersections between the reachable set and the guard
        guard: guard set (polytope representing constrained hyperplane)
        options: struct containing the algorithm settings (must contain 'enclose' key)
        params: model parameters for evaluating dynamic equation (only for 'flow' method)
        
    Returns:
        B: list containing the calculated basis matrices
    """
    
    # initialization
    sys = loc.contDynamics
    m = len(options['enclose'])
    B = [None] * m
    
    # loop over all selected methods
    for i in range(m):
        
        method = options['enclose'][i]
        
        if method == 'box':
            # Box method as described in Section V.A.a) in [1]
            B[i] = np.eye(sys.nr_of_dims)
            
        elif method == 'pca':
            # PCA method as described in Section V.A.b) in [1]
            
            # concatenate all generators
            G = _aux_extractGenerators(R)
            
            # project the generators onto the hyperplane
            if isinstance(guard, Polytope) and guard.representsa_('conHyperplane', 1e-12):
                Z = Zonotope(np.zeros((sys.nr_of_dims, 1)), G)
                Z_ = projectOnHyperplane(Z, guard)
                G = Z_.generators()
            
            # limit maximum number of generators to 500
            if G.shape[1] > 500:
                ind = np.argsort(np.sum(G**2, axis=0))[::-1]  # descending order
                G = G[:, ind[:500]]
            
            # calculate an orthogonal transformation matrix using PCA
            # MATLAB: [B{i},~,~] = svd([-G,G]);
            U, _, _ = np.linalg.svd(np.hstack([-G, G]), full_matrices=False)
            B[i] = U
            
        elif method == 'flow':
            # Flow method as described in Section V.A.d) in [1]
            
            if params is None:
                raise CORAerror('CORA:wrongInput', 'params', 
                              'params must be provided for flow method')
            
            # compute projected center of the union of all sets
            c = _aux_extractCenter(R)
            
            if isinstance(guard, Polytope) and guard.representsa_('conHyperplane', 1e-12):
                Z = Zonotope(c)
                Z_ = projectOnHyperplane(Z, guard)
                c = Z_.center()
            
            # compute flow at the center
            params_flow = params.copy()
            params_flow['u'] = params.get('uTrans', np.zeros((sys.nr_of_inputs, 1)))
            fcnHan = sys.getfcn(params_flow)
            dir_vec = fcnHan(0, c)
            
            dir_vec = dir_vec / np.linalg.norm(dir_vec)
            
            # get basis that is orthogonal to the flow direction
            B[i] = gramSchmidt(dir_vec)
            
        else:
            raise CORAerror('CORA:wrongFieldValue', 'options.enclose',
                          {'box', 'pca', 'flow'})
    
    return B


# Auxiliary functions -----------------------------------------------------

def _aux_extractGenerators(R: List[Any]) -> np.ndarray:
    """
    Extract all generator vectors from the sets that are over-approximated
    
    Args:
        R: list of reachable sets
        
    Returns:
        G: matrix of concatenated generators
    """
    from cora_python.contSet.zonoBundle import ZonoBundle
    from cora_python.contSet.polyZonotope import PolyZonotope
    from cora_python.contSet.conZonotope import ConZonotope
    
    G_list = []
    
    # loop over all sets
    for j in range(len(R)):
        
        Rj = R[j]
        
        # different types of sets
        if isinstance(Rj, ZonoBundle):
            for i in range(Rj.parallelSets):
                G_list.append(Rj.Z[i].generators())
        elif isinstance(Rj, PolyZonotope):
            temp = Zonotope(Rj)
            G_list.append(temp.generators())
        elif isinstance(Rj, Zonotope) or isinstance(Rj, ConZonotope):
            G_list.append(Rj.generators())
        else:
            # Fallback: assume it has a Z attribute with generators
            if hasattr(Rj, 'Z') and Rj.Z.shape[1] > 1:
                G_list.append(Rj.Z[:, 1:])  # Skip first column (center)
    
    if len(G_list) == 0:
        return np.array([]).reshape((0, 0))
    
    # Concatenate all generators
    G = np.hstack(G_list)
    return G


def _aux_extractCenter(R: List[Any]) -> np.ndarray:
    """
    Extract center from all sets and compute mean center
    
    Args:
        R: list of reachable sets
        
    Returns:
        c: mean center of all sets
    """
    from cora_python.contSet.zonoBundle import ZonoBundle
    
    c_list = []
    
    # loop over all sets
    for j in range(len(R)):
        
        Rj = R[j]
        
        # different types of sets
        if isinstance(Rj, ZonoBundle):
            for i in range(Rj.parallelSets):
                c_list.append(Rj.Z[i].center())
        else:
            c_list.append(Rj.center())
    
    if len(c_list) == 0:
        raise ValueError("No sets provided to extract center")
    
    # compute center of all centers
    c = np.mean(np.column_stack(c_list), axis=1, keepdims=True)
    return c

