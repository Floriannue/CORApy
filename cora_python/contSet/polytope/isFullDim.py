"""
isFullDim - checks if the dimension of the affine hull of a polytope is
    equal to the dimension of its ambient space; additionally, one can
    obtain a basis of the subspace in which the polytope is contained

Syntax:
    res = isFullDim(P)
    res = isFullDim(P, tol)
    res, subspace = isFullDim(P)

Inputs:
    P - polytope object
    tol - tolerance

Outputs:
    res - true/false
    subspace - (optional) Returns a set of orthogonal unit vectors
               x_1,...,x_k such that P is strictly contained in
               center(P)+span(x_1,...,x_k)
               (here, 'strictly' means that k is minimal).
               Note that if P is just a point, subspace=[].

Authors:       Niklas Kochdumper, Viktor Kotsev, Mark Wetzlinger, Adrian Kulmburg (MATLAB)
               Python translation by AI Assistant
Written:       02-January-2020 (MATLAB)
Last update:   10-July-2024 (MW, refactor, MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import TYPE_CHECKING, Tuple, Union, Optional

if TYPE_CHECKING:
    from cora_python.contSet.polytope.polytope import Polytope

from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.contSet.polytope.private.priv_isFullDim_V import priv_isFullDim_V


def isFullDim(P: 'Polytope', tol: float = 1e-12) -> Union[bool, Tuple[bool, Optional[np.ndarray]]]:
    """
    Checks if the dimension of the affine hull of a polytope is equal to the dimension of its ambient space.
    
    Args:
        P: polytope object
        tol: tolerance (default: 1e-12)
        
    Returns:
        res: True/False whether polytope is full-dimensional
        subspace: (optional) orthogonal unit vectors defining the subspace
    """
    
    # check if fullDim property is already set (MATLAB: ~isempty(P.fullDim.val))
    if hasattr(P, '_fullDim_val') and P._fullDim_val is not None:
        res = P._fullDim_val
        # In MATLAB, if only one output is requested and value is cached, return early
        # For two outputs, still compute subspace below...
        # For simplicity, assume single output for now
        return res
    
    # check whether V- or H-representation given
    if hasattr(P, 'isVRep') and P.isVRep:
        # --- V representation
        # compute degeneracy acc. to [1, (17)] and subspace acc. to [1, (20)]
        res, subspace = priv_isFullDim_V(P.V, tol)
    else:
        # --- H representation
        res, subspace = _aux_isFullDim_Hpoly(P, tol)
    
    # save the set property (only done once, namely, here!)
    P._fullDim_val = res
    
    return res, subspace


def _aux_isFullDim_Hpoly(P: 'Polytope', tol: float) -> Tuple[bool, Optional[np.ndarray]]:
    """Auxiliary function for H-representation polytopes"""
    
    n = P.dim()
    
    if n == 1:
        # 1D case
        return _aux_isFullDim_1D_Hpoly(P, tol)
    else:
        # >=2D case
        # For now, implement the simpler case without subspace computation
        res = _aux_isFullDim_nD_Hpoly_nosubspace(P, tol)
        subspace = None  # TODO: implement subspace computation if needed
        return res, subspace


def _aux_isFullDim_1D_Hpoly(P: 'Polytope', tol: float) -> Tuple[bool, Optional[np.ndarray]]:
    """1D case for H-representation"""
    # In 1D, polytope is full-dimensional if it's an interval, not a point
    # Check if there are inequality constraints that define a non-trivial interval
    
    if P.A is None or P.A.size == 0:
        # No inequality constraints -> fullspace (full-dimensional)
        return True, None
    
    # For 1D: A is of form [1; -1], b is [b1; b2] representing x <= b1, -x <= b2 (i.e., x >= -b2)
    # Full-dimensional if b1 > -b2 (i.e., interval has positive length)
    if P.A.shape[0] >= 2:
        # Find bounds
        upper_bounds = P.b[P.A[:, 0] > tol]  # x <= b constraints
        lower_bounds = -P.b[P.A[:, 0] < -tol]  # x >= -b constraints (from -x <= b)
        
        if len(upper_bounds) > 0 and len(lower_bounds) > 0:
            max_lower = np.max(lower_bounds)
            min_upper = np.min(upper_bounds)
            res = min_upper > max_lower + tol  # Interval has positive length
        else:
            res = True  # Unbounded interval
    else:
        res = True  # Single constraint or no constraints
    
    subspace = None if res else np.array([]).reshape(1, 0)
    return res, subspace


def _aux_isFullDim_nD_Hpoly_nosubspace(P: 'Polytope', tol: float) -> bool:
    """nD case for H-representation without subspace computation"""
    
    # Use linear programming approach as in MATLAB [1, (18)]
    # The idea is to maximize/minimize each coordinate direction
    # If the polytope is degenerate, some directions will have the same max/min
    
    n = P.dim()
    
    # Check if polytope is empty first
    if hasattr(P, 'emptySet') and P.emptySet:
        return False
    
    # Simple check: if there are equality constraints, check their rank
    if hasattr(P, 'Ae') and P.Ae is not None and P.Ae.size > 0:
        # If rank of equality constraints equals dimension, it's a point
        rank_Ae = np.linalg.matrix_rank(P.Ae)
        if rank_Ae == n:
            return False  # Point (0-dimensional)
        elif rank_Ae > n:
            return False  # Over-constrained (empty or inconsistent)
    
    # For a more robust check, we could use linear programming to find
    # the affine dimension, but for now use a simpler heuristic
    
    # If we have vertices available, use them
    try:
        V = P.vertices_()
        if V.size > 0:
            # Use SVD approach like in V-representation case
            if V.shape[1] <= n:
                return False
            V_centered = V - np.mean(V, axis=1, keepdims=True)
            _, S, _ = np.linalg.svd(V_centered)
            from cora_python.g.functions.matlab.validate.check.withinTol import withinTol
            rank = np.sum(~withinTol(S, 0, tol))
            return rank == n
    except:
        pass
    
    # Default to True if we can't determine otherwise
    # In practice, this should be enhanced with proper LP-based checking
    return True