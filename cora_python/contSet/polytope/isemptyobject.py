"""
isemptyobject - check if a polytope represents an empty set

Syntax:
    res = isemptyobject(P)

Inputs:
    P - polytope object

Outputs:
    res - true/false

Example:
    A = [1; -1];
    b = [-1; -1];
    P = polytope(A, b);
    res = isemptyobject(P)

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 16-September-2019 (MATLAB)
Last update: --- (MATLAB)
Python translation: 2025
"""

from typing import TYPE_CHECKING
import numpy as np
from scipy.optimize import linprog

if TYPE_CHECKING:
    from .polytope import Polytope

def isemptyobject(P: 'Polytope') -> bool:
    """
    Checks if a polytope object is fully empty.

    In the context of this library's Polytope class:
    - An empty H-representation (no constraints) means the polytope is the entire space.
    - An empty V-representation (no vertices) means the polytope is the empty set.

    This function returns true only if the object is "fully empty",
    meaning it has neither a valid H-rep nor a valid V-rep defined
    at the time of calling.
    """

    # Check if H-representation is effectively empty
    has_h_rep = P._has_h_rep and (
        (P._A is not None and P._A.size > 0) or
        (P._Ae is not None and P._Ae.size > 0)
    )

    # Check if V-representation is effectively empty
    has_v_rep = P._has_v_rep and (P._V is not None and P._V.size > 0)

    # The object is considered "empty" in this specific sense if neither are present
    return not has_h_rep and not has_v_rep


def _check_empty_halfspace(P) -> bool:
    """
    Check if polytope in halfspace representation is empty
    
    This is done by checking if the system A*x <= b has any feasible solution.
    """
    
    # Quick check: if any constraint has negative RHS and positive normal
    # then the polytope might be empty
    
    try:
        # Use linear programming to check feasibility
        # We solve: find x such that A*x <= b
        # If no solution exists, the polytope is empty
        
        n = P.A.shape[1]  # dimension
        m = P.A.shape[0]  # number of constraints
        
        # Objective function (doesn't matter for feasibility, use zero)
        c = np.zeros(n)
        
        # Solve LP: min 0^T x subject to A*x <= b
        result = linprog(c, A_ub=P.A, b_ub=P.b.flatten(), 
                        bounds=None, method='highs')
        
        # If LP finds no feasible solution, polytope is empty
        if not result.success:
            if 'infeasible' in str(result.message).lower():
                return True
        
        # Additional check: verify the solution
        if result.success and result.x is not None:
            # Check if the solution actually satisfies constraints
            violations = P.A @ result.x - P.b.flatten()
            if np.any(violations > 1e-9):  # Allow small numerical tolerance
                return True
            else:
                return False
        
    except Exception:
        # If LP solver fails, try alternative check
        pass
    
    # Alternative check: look for obviously infeasible constraints
    # Check if any two constraints are contradictory
    
    try:
        for i in range(P.A.shape[0]):
            for j in range(i + 1, P.A.shape[0]):
                # Check if constraints i and j are contradictory
                # This happens when A[i,:] ≈ -A[j,:] but b[i] + b[j] < 0
                
                ai = P.A[i, :]
                aj = P.A[j, :]
                bi = P.b[i, 0] if P.b.ndim > 1 else P.b[i]
                bj = P.b[j, 0] if P.b.ndim > 1 else P.b[j]
                
                # Check if normals are opposite
                if np.allclose(ai, -aj, atol=1e-12):
                    # Constraints are: ai^T x <= bi and aj^T x <= bj
                    # Which becomes: ai^T x <= bi and -ai^T x <= bj
                    # Or: ai^T x <= bi and ai^T x >= -bj
                    # Inconsistent if bi < -bj
                    if bi < -bj - 1e-12:
                        return True
                        
    except Exception:
        pass
    
    # Conservative default: assume not empty
    return False 