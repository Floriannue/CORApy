import numpy as np
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cora_python.contSet.polytope.polytope import Polytope

def priv_isFullDim(P: 'Polytope') -> bool:
    """
    Checks if a polytope is full-dimensional.

    Args:
        P: Polytope object.

    Returns:
        bool: True if the polytope is full-dimensional, False otherwise.
    """
    n = P.dim()

    if P.isVRep:
        if n == 0:
            return True # A 0-dimensional point is full-dimensional
        if n == 1:
            # 1D: full-dimensional if it's an interval, not a single point.
            # Check if it has infinite vertices or more than one unique vertex.
            if np.any(np.isinf(P.V)):
                return True
            else:
                # Check if it's a true interval or just a point
                return P.V.shape[1] > 1 or (P.V.shape[1] == 1 and not withinTol(P.V[:,0], P.V[:,1], np.finfo(float).eps))
        else:
            # nD V-representation
            if P.V.shape[1] <= n:
                # Full-dimensionality requires at least n+1 affinely independent vertices.
                # If num_vertices <= n, it cannot be full-dimensional.
                return False
            else:
                # Use SVD to check affine dimension (rank of centered vertices)
                # V is (dim x num_vertices)
                # Subtract mean from each column (vertex) to center them
                V_centered = P.V - np.mean(P.V, axis=1, keepdims=True)
                
                # SVD returns singular values in decreasing order. Count non-zero singular values.
                # Compare with tolerance to account for floating point inaccuracies.
                _, S, _ = np.linalg.svd(V_centered)
                
                # Number of non-zero singular values equals the rank (affine dimension).
                # Polytope is full-dimensional if its affine dimension is equal to n.
                return n == np.sum(~withinTol(S, 0, 1e-12)) # 1e-12 from MATLAB's aux_computeHiddenProperties
    
    elif P.isHRep:
        # Determine dimension from equality constraints
        # If Ae is empty, effective_dim is n. Otherwise, it's n - rank(Ae)
        if P.Ae.size > 0:
            # The dimension of the affine hull is n - rank(Ae)
            rank_Ae = np.linalg.matrix_rank(P.Ae, tol=1e-12)
            effective_dim = n - rank_Ae

            # If effective_dim is less than ambient dimension n, it's not full-dimensional in R^n
            if effective_dim < n:
                return False
            
            # If effective_dim equals n, and it's not empty (from inequality constraints),
            # then it is full-dimensional.
            # We also need to consider the case where the inequalities make it empty or reduce dimension
            # in the effective_dim subspace. This is a complex check and usually handled by an emptiness check.
            # For now, if effective_dim is n, and it's not empty, we consider it full-dimensional.
            # P.emptySet property will handle the emptiness check for us.
            return not P.emptySet # If it's not empty and effective dim is n, then full dim
        else:
            # No equality constraints (Ae is empty). Full-dimensionality depends on inequalities.
            # If it's not empty and inequalities don't make it degenerate, it's full-dimensional.
            # The P.emptySet property should correctly determine if inequalities make it empty.
            # For a general H-representation, if it's not empty and has no equalities, it's full-dimensional.
            return not P.emptySet

    else:
        # This case implies an empty polytope or a 0-dimensional point without defined representation
        # Empty set is not full-dimensional.
        # For a 0-dim point, if it's not VRep, it might be an empty point.
        return False 