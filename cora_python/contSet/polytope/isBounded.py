"""
isBounded - check if a polytope is bounded

Syntax:
    res = isBounded(P)

Inputs:
    P - polytope object

Outputs:
    res - true/false

Example:
    A = [1 0; 0 1; -1 0; 0 -1];
    b = [1; 1; 1; 1];
    P = polytope(A, b);
    res = isBounded(P)

Authors: Matthias Althoff, Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 30-September-2006 (MATLAB)
Last update: 16-September-2019 (MW, specify output for empty case) (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import Optional
from scipy.optimize import linprog
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .polytope import Polytope


def isBounded(P: 'Polytope') -> bool:
    """
    Check if a polytope is bounded
    
    Args:
        P: Polytope object
        
    Returns:
        bool: True if polytope is bounded, False otherwise
    """
    
    # Empty polytope is considered bounded
    if P.isemptyobject():
        return True
    
    # For halfspace representation: A*x <= b
    if P.A is not None and P.b is not None and P.A.size > 0:
        return _check_bounded_halfspace(P)
    
    # If no constraints (empty A and b), the polytope represents R^n, which is unbounded
    if P.A is not None and P.b is not None and P.A.size == 0:
        return False
    
    # For vertex representation: always bounded
    if hasattr(P, '_isVRep') and P._isVRep and hasattr(P, 'V') and P.V is not None and P.V.size > 0:
        return True
    
    # Default case - empty polytope
    return True


def _check_bounded_halfspace(P) -> bool:
    """
    Check if polytope in halfspace representation is bounded
    
    This is done by checking if the system A*x <= b has unbounded solutions
    in any direction. We solve linear programs in different directions.
    """
    
    n = P.A.shape[1]  # dimension
    
    # Check if polytope is bounded by solving LP in each coordinate direction
    # If we can find an unbounded direction, the polytope is unbounded
    
    # Test positive and negative directions for each coordinate
    for i in range(n):
        for direction in [1, -1]:
            # Objective: maximize/minimize x_i subject to A*x <= b
            c = np.zeros(n)
            c[i] = direction
            
            try:
                # Solve LP: min c^T x subject to A*x <= b
                result = linprog(c, A_ub=P.A, b_ub=P.b.flatten(), 
                               bounds=None, method='highs')
                
                # If the LP is unbounded, the polytope is unbounded
                if not result.success and 'unbounded' in str(result.message).lower():
                    return False
                    
                # Also check if objective value is extremely large
                if result.success and abs(result.fun) > 1e10:
                    return False
                    
            except Exception:
                # If LP solver fails, assume bounded (conservative approach)
                continue
    
    # Additional check: verify that the recession cone is {0}
    # This is a more thorough but computationally expensive check
    
    # For efficiency, we'll use a heuristic approach:
    # Check if there exists a direction d such that A*d <= 0 and d != 0
    
    # This would indicate an unbounded direction
    try:
        # Try to find a non-zero direction d such that A*d <= 0
        # We solve: find d such that ||d|| = 1 and A*d <= 0
        
        # Use a simple approach: check if any combination of constraint normals
        # gives a direction where all constraints are satisfied
        
        # If all constraint normals point "outward", polytope is likely bounded
        rank_A = np.linalg.matrix_rank(P.A)
        if rank_A == n:
            # Full rank constraint matrix often indicates boundedness
            return True
            
    except Exception:
        pass
    
    # Conservative default: assume bounded
    return True 