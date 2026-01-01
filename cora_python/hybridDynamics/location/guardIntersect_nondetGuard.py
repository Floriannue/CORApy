"""
guardIntersect_nondetGuard - enclosure of guard intersections for 
   non-deterministic guard sets with large uncertainty

Syntax:
    R = guardIntersect_nondetGuard(loc,R,guard,B)

Inputs:
    loc - location object
    R - list of intersections between the reachable set and the guard
    guard - guard set (class: constrained hyperplane)
    B - basis

Outputs:
    R - set enclosing the guard intersection

Authors:       Niklas Kochdumper
Written:       19-December-2019
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import Any, List
from cora_python.contSet.polytope import Polytope
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.zonoBundle import ZonoBundle
from cora_python.contSet.interval import Interval
from cora_python.contSet.conZonotope import ConZonotope
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def guardIntersect_nondetGuard(loc: Any, R: List[Any], guard: Polytope, 
                               B: List[np.ndarray]) -> Any:
    """
    Enclosure of guard intersections for non-deterministic guard sets with large uncertainty
    
    Args:
        loc: location object
        R: list of intersections between the reachable set and the guard
        guard: guard set (polytope representing constrained hyperplane)
        B: list of basis matrices
        
    Returns:
        R: set enclosing the guard intersection (zonotope, zonoBundle, or empty)
    """
    
    # loop over all basis
    Z = [None] * len(B)
    
    for i in range(len(B)):
        
        # enclose all reachable set with an interval in transformed space
        I = Interval.empty(B[i].shape[0])
        
        for j in range(len(R)):
            # MATLAB: I_new = interval(B{i}'*R{j});
            # Transform set and convert to interval
            R_transformed = B[i].T @ R[j]
            if hasattr(R_transformed, 'interval'):
                I_new = R_transformed.interval()
            else:
                I_new = Interval(R_transformed)
            
            # MATLAB: if representsa_(I_new,'emptySet',eps) && representsa_(I,'emptySet',eps)
            if I_new.representsa_('emptySet', np.finfo(float).eps) and I.representsa_('emptySet', np.finfo(float).eps):
                # MATLAB: I = I | B{i}'*interval(R{j});
                # Get interval of R[j], then transform
                if hasattr(R[j], 'interval'):
                    R_j_interval = R[j].interval()
                else:
                    R_j_interval = Interval(R[j])
                R_j_transformed = B[i].T @ R_j_interval
                if hasattr(R_j_transformed, 'interval'):
                    R_j_transformed_interval = R_j_transformed.interval()
                else:
                    R_j_transformed_interval = Interval(R_j_transformed)
                I = I.or_op(R_j_transformed_interval) if hasattr(I, 'or_op') else I | R_j_transformed_interval
            else:
                # MATLAB: I = I | I_new;
                I = I.or_op(I_new) if hasattr(I, 'or_op') else I | I_new
        
        # backtransformation to the original space
        # MATLAB: Z_trans = B{i} * zonotope(I);
        Z_trans = B[i] @ Zonotope(I)
        
        # convert the resulting set to a constrained zonotope
        cZ = ConZonotope(Z_trans)
        
        # intersect the set with the guard set
        # MATLAB: cZ = and_(cZ,guard,'exact');
        cZ = cZ.and_(guard, 'exact')
        
        # enclose the intersection with an interval in transformed space
        # MATLAB: Z{i} = B{i} * zonotope(interval(B{i}'*cZ));
        cZ_transformed = B[i].T @ cZ
        if hasattr(cZ_transformed, 'interval'):
            cZ_transformed_interval = cZ_transformed.interval()
        else:
            cZ_transformed_interval = Interval(cZ_transformed)
        Z[i] = B[i] @ Zonotope(cZ_transformed_interval)
    
    # construct set enclosing the intersection
    if len(Z) == 1:
        R = Z[0]
    else:
        R = ZonoBundle(Z)
    
    return R

