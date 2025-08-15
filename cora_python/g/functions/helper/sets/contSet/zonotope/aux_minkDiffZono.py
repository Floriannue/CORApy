"""
aux_minkDiffZono - helper implementation of the Minkowski difference core step for zonotopes

This is the canonical implementation used by both tests and the high-level
`cora_python.contSet.zonotope.minkDiff` API.
"""

import numpy as np
from cora_python.contSet.zonotope.zonotope import Zonotope
from cora_python.contSet.polytope.polytope import Polytope
from cora_python.g.functions.matlab.converter.CORAlinprog import CORAlinprog
from cora_python.g.functions.helper.sets.contSet.zonotope.aux_tightenHalfspaces import aux_tightenHalfspaces
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def aux_minkDiffZono(minuend: Zonotope, subtrahend: Zonotope, method: str) -> Zonotope:
    """Compute Minkowski difference using the approach in [1].

    Args:
        minuend: Zonotope minuend
        subtrahend: Zonotope subtrahend
        method: One of {'inner', 'outer', 'outer:coarse', 'approx', 'exact'}

    Returns:
        Zonotope: Result zonotope
    """

    # Obtain halfspace representation of the minuend's polytope hull
    P = Polytope(minuend)
    H_orig_twice = P.A
    K_orig_twice = P.b
    H_orig = H_orig_twice[: H_orig_twice.shape[0] // 2, :]

    # Number of subtrahend generators
    subtrahend_gens = subtrahend.G.shape[1]

    # Intersect polytopes according to Theorem 3 of [1]
    delta_K = H_orig_twice @ subtrahend.center()
    for i in range(subtrahend_gens):
        delta_K = delta_K + np.abs(H_orig_twice @ subtrahend.generators()[:, i : i + 1])
    K_orig_new = K_orig_twice - delta_K

    C = H_orig
    d = K_orig_new[: K_orig_new.shape[0] // 2, :]

    # Compute center
    c = minuend.center() - subtrahend.center()

    # Obtain minuend generators
    G = minuend.generators()

    # Reverse computation from halfspace generation
    n = minuend.dim()
    if method == "inner" or (method == "exact" and n == 2):
        delta_d = d - C @ minuend.c + C @ subtrahend.c
        A_abs = np.abs(C @ G)
        dims = A_abs.shape[1]
        # Vector of cost function
        f = np.linalg.norm(minuend.G, 2, axis=0)
        # A_abs x <= delta_d && x >= 0
        problem = {
            "f": -f,
            "Aineq": np.vstack([A_abs, -np.eye(dims)]),
            "bineq": np.vstack([delta_d, np.zeros((dims, 1))]),
            "Aeq": None,
            "beq": None,
            "lb": None,
            "ub": None,
        }
        alpha, _, exitflag = CORAlinprog(problem)[:3]
        if alpha is None or exitflag != 1:
            return Zonotope.empty(n)

    elif method in ["outer", "outer:coarse"]:
        # Reduce delta_d using linear programming
        if method == "outer":
            d_shortened = aux_tightenHalfspaces(H_orig_twice, K_orig_new)
        else:
            d_shortened = K_orig_new

        # Is set empty?
        if d_shortened is None or d_shortened.size == 0:
            # Return empty set with correct dimensions
            return Zonotope.empty(n)
        else:
            # Vector of cost function
            f = np.linalg.norm(minuend.generators(), 2, axis=0)
            # Obtain unrestricted A_abs and delta_d
            C = H_orig
            d = d_shortened[: d_shortened.shape[0] // 2, :]
            delta_d = d - C @ minuend.center() + C @ subtrahend.center()
            A_abs = np.abs(C @ G)
            dims = A_abs.shape[1]
            # A_abs x >= delta_d && x >= 0
            problem = {
                "f": f,
                "Aineq": np.vstack([-A_abs, -np.eye(dims)]),
                "bineq": np.vstack([-delta_d, np.zeros((dims, 1))]),
                "Aeq": None,
                "beq": None,
                "lb": None,
                "ub": None,
            }
            alpha, _, exitflag = CORAlinprog(problem)[:3]
            # alpha unused here; feasible region defines outer approximation

    elif method == "approx":
        delta_d = d - C @ minuend.center() + C @ subtrahend.center()
        A_abs = np.abs(C @ G)
        # Use pseudoinverse to compute an approximation
        alpha = np.linalg.pinv(A_abs) @ delta_d

    else:
        # Should already be caught before in higher level
        raise CORAerror("CORA:specialError", f"Unknown method: '{method}'")

    # Instantiate Z
    G_new = minuend.generators() @ np.diag(alpha.flatten())
    # Remove all zero columns
    G_new = G_new[:, ~np.all(G_new == 0, axis=0)]
    return Zonotope(c, G_new)



