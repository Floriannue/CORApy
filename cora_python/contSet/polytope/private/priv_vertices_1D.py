import numpy as np
from typing import Optional
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol

def priv_vertices_1D(A: Optional[np.ndarray], b: Optional[np.ndarray], 
                     Ae: Optional[np.ndarray], be: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """
    Compute vertices for 1D polytope.
    
    Args:
        A: inequality constraint matrix (n_ineq x 1)
        b: inequality constraint offset (n_ineq,)
        Ae: equality constraint matrix (n_eq x 1)
        be: equality constraint offset (n_eq,)
        
    Returns:
        V: vertices as (1 x n_vertices) array, or None if empty
    """
    # Initialize bounds
    lb = -np.inf
    ub = np.inf
    
    # Process equality constraints first
    if Ae is not None and Ae.size > 0 and be is not None and be.size > 0:
        # For 1D, equality constraint Ae*x = be means x = be/Ae
        # Check if all equality constraints are consistent
        for i in range(len(be)):
            if abs(Ae[i, 0]) < 1e-12:  # Ae[i] ≈ 0
                if abs(be[i]) > 1e-12:  # be[i] ≠ 0
                    # Constraint 0*x = be[i] with be[i] ≠ 0 -> infeasible
                    return None
                # else: constraint 0*x = 0 -> always satisfied, ignore
            else:
                # x = be[i] / Ae[i, 0]
                x_eq = be[i] / Ae[i, 0]
                # All equality constraints must give the same x value
                if lb == -np.inf and ub == np.inf:
                    lb = ub = x_eq
                elif not withinTol(x_eq, lb, 1e-12):
                    # Inconsistent equality constraints
                    return None
    
    # Process inequality constraints
    if A is not None and A.size > 0 and b is not None and b.size > 0:
        for i in range(len(b)):
            if abs(A[i, 0]) < 1e-12:  # A[i] ≈ 0
                if b[i] < -1e-12:  # b[i] < 0
                    # Constraint 0*x <= b[i] with b[i] < 0 -> infeasible
                    return None
                # else: constraint 0*x <= b[i] with b[i] >= 0 -> always satisfied, ignore
            else:
                # A[i]*x <= b[i] -> x <= b[i]/A[i] (if A[i] > 0) or x >= b[i]/A[i] (if A[i] < 0)
                if A[i, 0] > 0:
                    # x <= b[i] / A[i, 0]
                    ub = min(ub, b[i] / A[i, 0])
                else:
                    # x >= b[i] / A[i, 0]
                    lb = max(lb, b[i] / A[i, 0])
    
    # Check feasibility
    if lb > ub + 1e-12:
        return None  # Empty set
    
    # Construct vertices
    if lb == ub:
        # Single point
        return np.array([[lb]])
    elif lb == -np.inf and ub == np.inf:
        # Unbounded in both directions -> fullspace in 1D
        return np.array([[-np.inf, np.inf]])
    elif lb == -np.inf:
        # Unbounded below, bounded above
        return np.array([[-np.inf, ub]])
    elif ub == np.inf:
        # Bounded below, unbounded above
        return np.array([[lb, np.inf]])
    else:
        # Bounded interval
        return np.array([[lb, ub]]) 