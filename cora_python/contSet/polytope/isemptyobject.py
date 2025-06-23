"""
isemptyobject - checks if a polytope object is fully empty;
   if the H representation is given, this represents R^n since there are
   no constraints excluding any point
   if, instead, the V representation is given, this represents the empty
   set since there are no vertices

Syntax:
    res = isemptyobject(P)

Inputs:
    P - polytope object

Outputs:
    res - true/false

Example: 
    P = polytope([1 0;-1 0;0 1;0 -1],[3;0;3;-4]);
    isemptyobject(P); % false

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger
Written:       25-July-2023
Last update:   14-July-2024 (MW, support V representation)
               25-February-2025 (TL, bug fix, .inf/.empty cases)
Last revision: ---
"""

from typing import TYPE_CHECKING
import numpy as np
from scipy.optimize import linprog

if TYPE_CHECKING:
    from .polytope import Polytope


def isemptyobject(P: 'Polytope') -> bool:
    """
    Checks if a polytope object is fully empty.
    
    Args:
        P: polytope object
        
    Returns:
        res: true if polytope is empty object, false otherwise
    """
    
    # no inequality or equality constraints
    res_H = not P._has_h_rep or (
        (P._b is None or P._b.size == 0) and 
        (P._be is None or P._be.size == 0)
    )
    
    # no vertices
    res_V = not P._has_v_rep or (P._V is None or P._V.size == 0)
    
    # combine information
    res = res_H and res_V
    
    return res


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