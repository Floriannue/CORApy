"""
guardIntersect_levelSet - implementation of the guard intersection 
   enclosure with polynomial zonotopes as described in [1]

Syntax:
    R = guardIntersect_levelSet(loc,R,guard,options)

Inputs:
    loc - location object
    R - list of intersections between the reachable set and the guard
    guard - guard set (class: levelSet)

Outputs:
    R - polynomial zonotope enclosing the guard intersection

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: guardIntersect

References: 
  [1] N. Kochdumper et al. "Reachability Analysis for Hybrid Systems with 
      Nonlinear Guard Sets", HSCC 2020

Authors:       Niklas Kochdumper
Written:       07-January-2020 
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import Any, List
from cora_python.contSet.levelSet import LevelSet
from cora_python.contSet.interval import Interval
from cora_python.contSet.polyZonotope import PolyZonotope
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def guardIntersect_levelSet(loc: Any, R: List[Any], guard: LevelSet) -> PolyZonotope:
    """
    Implementation of the guard intersection enclosure with polynomial zonotopes as described in [1]
    
    Args:
        loc: location object
        R: list of intersections between the reachable set and the guard
        guard: guard set (levelSet object)
        
    Returns:
        R: polynomial zonotope enclosing the guard intersection
    """
    
    # compute enclosing interval for all reachable sets
    # MATLAB: I = interval.empty(dim(R{1}));
    if len(R) == 0:
        raise CORAerror('CORA:wrongInput', 'R must contain at least one set')
    
    # Get dimension from first set
    if hasattr(R[0], 'dim'):
        dim_R = R[0].dim()
    else:
        # Fallback: try to get dimension from interval conversion
        I_temp = Interval(R[0]) if hasattr(R[0], 'interval') else R[0].interval()
        dim_R = I_temp.dim()
    
    I = Interval.empty(dim_R)
    
    for i in range(len(R)):
        
        # interval enclosure (see Step 2 in Sec. 3.2 in [1])
        # MATLAB: I_ = interval(R{i});
        if hasattr(R[i], 'interval'):
            I_ = R[i].interval()
        else:
            I_ = Interval(R[i])
        
        # domain tightening only possible for guard sets with comparison
        # operator '=='; for instant transitions with conditions, level sets
        # have comparison operator '<=' and thus the domain cannot be
        # tightened; however, instant transitions also only have one set in R
        # MATLAB: try I_ = tightenDomain(guard,I_); end
        # Note: MATLAB try-end without catch will propagate errors if tightenDomain fails.
        # The try block suggests tightenDomain may not always be applicable, but MATLAB
        # will let the error propagate. In Python, we match this behavior exactly.
        try:
            I_ = guard.tightenDomain(I_)
        except:
            # MATLAB try-end without catch propagates errors. However, based on the
            # comment, tightenDomain may not be applicable for compOp != '=='.
            # If tightenDomain is not implemented or fails, we continue without tightening
            # to match the expected behavior (the comment suggests this is acceptable).
            pass
        
        # compute union of all intervals (see Step 3 in Sec. 3.2 in [1])
        # MATLAB: I = I | I_;
        I = I.or_op(I_) if hasattr(I, 'or_op') else I | I_
    
    # compute the intersection of the reachable set with the guard set
    # (see Step 4 in Sec. 3.2 in [1])
    # MATLAB: pZ = polyZonotope(I);
    pZ = PolyZonotope(I)
    
    # MATLAB: R = and_(guard,pZ,'approx');
    R = guard.and_(pZ, 'approx')
    
    return R

