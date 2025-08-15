"""
priv_feasiblePoint - returns a feasible point of a polytope if one exists

This mirrors MATLAB's helper by solving a feasibility LP:
  find x such that A x <= b and Ae x = be

Returns (x, success) where x is (n,1) or None.
"""

import numpy as np
from typing import Tuple, Optional
from cora_python.g.functions.matlab.converter.CORAlinprog import CORAlinprog

def priv_feasiblePoint(A: np.ndarray, b: np.ndarray, Ae: np.ndarray, be: np.ndarray, n: int) -> Tuple[Optional[np.ndarray], bool]:
    """
    Computes a feasible point of a polytope.
    """
    # linprog requires a cost function c. For feasibility, we can use a zero vector.
    c = np.zeros(n)

    print(f"DEBUG (priv_feasiblePoint): A shape: {A.shape}, b shape: {b.shape}")
    print(f"DEBUG (priv_feasiblePoint): Ae shape: {Ae.shape}, be shape: {be.shape}")
    print(f"DEBUG (priv_feasiblePoint): n: {n}")
    print(f"DEBUG (priv_feasiblePoint): A content:\n{A}")
    print(f"DEBUG (priv_feasiblePoint): b content:\n{b}")
    # Ensure b and be are 1D arrays for linprog
    b_flat = b.flatten() if b.size > 0 else None
    be_flat = be.flatten() if be.size > 0 else None

    try:
        # Note: bounds are crucial. Default (0, None) restricts variables to non-negative.
        # We need (None, None) for unconstrained variables.
        from scipy.optimize import linprog
        res = linprog(c=c, A_ub=A, b_ub=b_flat, A_eq=Ae, b_eq=be_flat, 
                      bounds=[(None, None)] * n, method='highs')
        
        print(f"DEBUG (priv_feasiblePoint): linprog success: {res.success}")
        print(f"DEBUG (priv_feasiblePoint): linprog status: {res.status}")
        print(f"DEBUG (priv_feasiblePoint): linprog message: {res.message}")
        print(f"DEBUG (priv_feasiblePoint): linprog x: {res.x}")
        
        if res.success:
            return res.x.reshape(-1, 1), True
        else:
            # linprog.status can indicate infeasibility (status 2)
            if res.status == 2: # Infeasible
                return None, False
            else: # Other failures (unbounded, numerical error, etc.)
                # Treat other non-success as infeasible for now, or raise specific error if needed
                return None, False
    except Exception as e:
        print(f"ERROR (priv_feasiblePoint): Exception during linprog call: {e}")
        return None, False


