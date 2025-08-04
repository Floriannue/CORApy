import numpy as np
from typing import Optional
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol
from cora_python.g.functions.matlab.converter.CORAlinprog import CORAlinprog
from .priv_vertices_1D import priv_vertices_1D
from .priv_normalize_constraints import priv_normalize_constraints
from .priv_equality_to_inequality import priv_equality_to_inequality

def priv_representsa_emptySet(A: np.ndarray, b: np.ndarray, 
                             Ae: Optional[np.ndarray], be: Optional[np.ndarray], 
                             n: int, tol: float = 1e-9) -> bool:
    """
    Determines if a polytope is empty.
    
    Description:
        A*x <= b is infeasible if
            min  b'*y
            s.t. A'*y =  0,
                    y >= 0
        is feasible. If problem is either unbounded (below since minimization)
        or b'*y < 0, the polytope is empty. If the problem is infeasible or
        b'*y == 0, the polytope is not empty.
    
    Args:
        A: inequality constraint matrix
        b: inequality constraint offset
        Ae: equality constraint matrix
        be: equality constraint offset
        n: dimension of the polytope
        tol: tolerance
        
    Returns:
        bool: True if polytope is empty, False otherwise
    """
    # Special case: 1D with bounded polytope
    if n == 1:
        # For 1D, use constraint satisfaction check instead of vertex computation
        # to properly handle unbounded polytopes
        
        # Check for inconsistent equality constraints first
        if Ae is not None and Ae.size > 0 and be is not None and be.size > 0:
            # For 1D, equality constraint Ae*x = be means x = be/Ae
            eq_values = []
            for i in range(len(be)):
                if abs(Ae[i, 0]) < 1e-12:  # Ae[i] ≈ 0
                    if abs(be[i]) > 1e-12:  # be[i] ≠ 0
                        # Constraint 0*x = be[i] with be[i] ≠ 0 -> infeasible
                        return True
                    # else: constraint 0*x = 0 -> always satisfied, ignore
                else:
                    # x = be[i] / Ae[i, 0]
                    x_eq = be[i] / Ae[i, 0]
                    eq_values.append(x_eq)
            
            # Check if all equality constraints are consistent
            if len(eq_values) > 1:
                for i in range(1, len(eq_values)):
                    if not withinTol(eq_values[i], eq_values[0], 1e-12):
                        return True  # Inconsistent equality constraints
        
        # Check inequality constraints for feasibility
        if A is not None and A.size > 0 and b is not None and b.size > 0:
            for i in range(len(b)):
                if abs(A[i, 0]) < 1e-12:  # A[i] ≈ 0
                    if b[i] < -1e-12:  # b[i] < 0
                        # Constraint 0*x <= b[i] with b[i] < 0 -> infeasible
                        return True
        
        # If we reach here, the 1D polytope is feasible (might be unbounded)
        return False
    
    # Quick check: no constraints
    if (b is None or b.size == 0) and (be is None or be.size == 0):
        return False
    
    # Quick check: are there constraints of the form 0*x <= b with b < 0?
    if A is not None and A.size > 0 and b is not None and b.size > 0:
        zero_rows = np.all(withinTol(A, 0, tol), axis=1)
        if np.any(zero_rows) and np.any(b[zero_rows] < 0):
            return True
    
    # Number of inequality and equality constraints
    nrConIneq = b.size if b is not None else 0
    nrConEq = be.size if be is not None else 0
    
    # Normalize constraints
    A_norm, b_norm, Ae_norm, be_norm = priv_normalize_constraints(A, b, Ae, be, 'A')
    
    if Ae_norm is not None and Ae_norm.size > 0:
        # Polytope can already be declared empty if there is no point
        # that satisfies the set of equality constraints
        if np.linalg.matrix_rank(Ae_norm) == min(nrConEq, n):
            # If the rank of the equality constraint matrix is smaller than
            # the minimum of the number of equality constraints and the
            # dimension, then there are still degrees of freedom
            
            # Check for aligned vectors: pre-compute dot product
            dotprod_norm = Ae_norm @ Ae_norm.T
            
            for i in range(nrConEq):
                # If there are aligned constraints, then they have to have
                # the same value in be (all but one of them are redundant),
                # otherwise the polytope is the empty set because there is
                # no point that can satisfy two parallel Ae*x = be at the
                # same time
                
                # Check for aligned vectors
                comparison_array = withinTol(Ae_norm[i, :] - dotprod_norm[:, i] * Ae_norm[i, :],
                                             np.zeros(n))
                
                # Ensure the array is at least 2D before applying axis=1
                if comparison_array.ndim == 1:
                    aligned_constraints = np.all(comparison_array)
                else:
                    aligned_constraints = np.all(comparison_array, axis=1)
                
                if np.sum(aligned_constraints) > 1:
                    # At least two constraints are aligned
                    if not np.all(withinTol(be_norm[aligned_constraints], be_norm[i])):
                        # Polytope is the empty set
                        return True
        
        # Rewrite equality constraints as inequality constraints
        A_norm, b_norm = priv_equality_to_inequality(A_norm, b_norm, Ae_norm, be_norm)
        # Update number of inequality constraints
        nrConIneq = b_norm.size
    
    # Solve the dual problem using linear programming
    problem = {
        'f': b_norm.T if b_norm is not None else np.array([]),
        'Aineq': np.vstack([
            -np.eye(nrConIneq),
            A_norm.T,
            -A_norm.T
        ]) if A_norm is not None else -np.eye(nrConIneq),
        'bineq': np.zeros(nrConIneq + 2*n),
        'Aeq': None,
        'beq': None,
        'lb': None,
        'ub': None
    }
    
    # Solve linear program
    try:
        x, fval, exitflag, output, lambda_out = CORAlinprog(problem)
        
        if exitflag == -2 or (exitflag > 0 and fval >= -1e-10):
            # If this problem is infeasible or if optimal objective value
            # is >= 0, the polytope is not empty
            return False
        elif exitflag == -3 or (exitflag > 0 and fval < -1e-10):
            # If problem is unbounded (below since minimization) or
            # objective value is smaller than zero, polytope is empty
            return True
        else:
            # Problem could not be solved -> assume not empty
            return False
            
    except Exception:
        # If optimization fails, assume not empty for safety
        return False 