"""
guardIntersect_conZonotope - constrained zonotope based enclosure of 
   guard intersections

Syntax:
    R = guardIntersect_conZonotope(loc,R,guard,B,options)

Inputs:
    loc - location object
    R - list of intersections between the reachable set and the guard
    guard - guard set (class: constrained hyperplane)
    B - basis
    options - required algorithm parameters: .reductionTechnique, .guardOrder

Outputs:
    R - set enclosing the guard intersection

Authors:       Niklas Kochdumper
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
from cora_python.contSet.polyZonotope import PolyZonotope
from cora_python.contSet.conZonotope import ConZonotope
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def guardIntersect_conZonotope(loc: Any, R: List[Any], guard: Polytope, 
                               B: List[np.ndarray], options: Dict[str, Any]) -> Any:
    """
    Constrained zonotope based enclosure of guard intersections
    
    Args:
        loc: location object
        R: list of intersections between the reachable set and the guard
        guard: guard set (polytope representing constrained hyperplane)
        B: list of basis matrices
        options: required algorithm parameters (must contain 'reductionTechnique' and 'guardOrder')
        
    Returns:
        R: set enclosing the guard intersection (zonotope or zonoBundle)
    """
    
    # convert all relevant reachable sets to constrained zonotopes
    for i in range(len(R)):
        # MATLAB: if isa(R{i},'polyZonotope')
        if isinstance(R[i], PolyZonotope):
            # MATLAB: R{i} = zonotope(R{i});
            if hasattr(R[i], 'zonotope'):
                R[i] = R[i].zonotope()
            else:
                R[i] = Zonotope(R[i])
        
        # MATLAB: R_reduce = reduce(R{i},options.reductionTechnique,options.guardOrder);
        if hasattr(R[i], 'reduce'):
            R_reduce = R[i].reduce(options['reductionTechnique'], options['guardOrder'])
        else:
            from cora_python.contSet.contSet.reduce import reduce
            R_reduce = reduce(R[i], options['reductionTechnique'], options['guardOrder'])
        
        # MATLAB: R{i} = conZonotope(R_reduce);
        if hasattr(R_reduce, 'conZonotope'):
            R[i] = R_reduce.conZonotope()
        else:
            R[i] = ConZonotope(R_reduce)
    
    # intersect the reachable sets with the guard set
    for i in range(len(R)):
        # MATLAB: R{i} = and_(R{i},guard,'exact');
        R[i] = R[i].and_(guard, 'exact')
    
    # MATLAB: R = R(~cellfun('isempty',R));
    R = [r for r in R if r is not None and not (hasattr(r, 'representsa_') and r.representsa_('emptySet', np.finfo(float).eps))]
    
    # loop over all calculated basis
    Z = [None] * len(B)
    
    for i in range(len(B)):
        
        # MATLAB: I = interval.empty(size(B{i},1));
        I = Interval.empty(B[i].shape[0])
        
        # loop over all reachable sets
        for j in range(len(R)):
            
            # interval enclosure in the transformed space
            # MATLAB: intTemp = interval(B{i}'*R{j});
            R_transformed = B[i].T @ R[j]
            if hasattr(R_transformed, 'interval'):
                intTemp = R_transformed.interval()
            else:
                intTemp = Interval(R_transformed)
            
            # unite all intervals
            # MATLAB: I = I | intTemp;
            I = I.or_op(intTemp) if hasattr(I, 'or_op') else I | intTemp
        
        # backtransformation to the original space
        # MATLAB: Z{i} = B{i} * zonotope(I);
        Z_i = Zonotope(I)
        Z[i] = B[i] @ Z_i
    
    # construct the enclosing zonotope bundle object
    if len(Z) == 1:
        R = Z[0]
    else:
        R = ZonoBundle(Z)
    
    return R

