import numpy as np
from scipy.optimize import linprog
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cora_python.contSet.polytope.polytope import Polytope

def priv_emptySet(P: 'Polytope', tol: float) -> bool:
    """
    Checks if a polytope is empty by solving a linear program.

    Args:
        P: Polytope object
        tol: Tolerance for numerical comparisons

    Returns:
        bool: True if the polytope is empty, False otherwise.
    """
    # Ensure halfspace representation is available
    P.constraints()

    n = P.dim()

    # Handle 0-dimensional polytope separately as linprog struggles with n=0
    if n == 0:
        # A 0-dimensional polytope with no constraints is considered an empty set
        # in CORA, aligning with MATLAB's default polytope() constructor behavior.
        # If it has any constraints (even trivial ones), it would imply a contradiction
        # in 0D space, hence also empty.
        return True 

    # Objective function (can be any, as we are only interested in feasibility)
    c = np.zeros(n)

    # Inequality constraints: A_ub @ x <= b_ub
    A_ub = P.A if P.A is not None else np.array([]).reshape(0, n)
    b_ub = P.b.flatten() if P.b is not None else np.array([])

    # Equality constraints: A_eq @ x == b_eq
    A_eq = P.Ae if P.Ae is not None else np.array([]).reshape(0, n)
    b_eq = P.be.flatten() if P.be is not None else np.array([])

    # Convert empty arrays to None for linprog compatibility
    A_ub_lp = A_ub if A_ub.size > 0 else None
    b_ub_lp = b_ub if b_ub.size > 0 else None
    A_eq_lp = A_eq if A_eq.size > 0 else None
    b_eq_lp = b_eq if b_eq.size > 0 else None

    # Solve the linear program
    res = linprog(
        c,
        A_ub=A_ub_lp,
        b_ub=b_ub_lp,
        A_eq=A_eq_lp,
        b_eq=b_eq_lp,
        bounds=(None, None),
        options={'disp': False, 'tol': tol}
    )

    # A polytope is empty if the LP is infeasible
    return res.status == 2 