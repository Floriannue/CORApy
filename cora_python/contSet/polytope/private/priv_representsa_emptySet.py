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
    # Special case: 1D
    if n == 1:
        # Compute vertices (fast for 1D) and check whether empty
        V = priv_vertices_1D(A, b, Ae, be)
        return V is None or V.size == 0
    
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
                aligned_constraints = np.all(
                    withinTol(Ae_norm[i, :] - dotprod_norm[:, i] * Ae_norm[i, :],
                             np.zeros(n)), axis=1)
                
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