import numpy as np
import cvxpy as cp
from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid
from numpy.linalg import svd, inv
from scipy.linalg import sqrtm

def priv_distanceEllipsoid(E1_orig: Ellipsoid, E2_orig: Ellipsoid) -> float:
    """
    Computes the distance between two ellipsoids by solving a QCQP.
    """
    if E1_orig.dim != E2_orig.dim:
        raise ValueError("Ellipsoids must have the same dimension.")

    if not E1_orig.isFullDim() and not E2_orig.isFullDim():
        raise ValueError("At least one ellipsoid must be non-degenerate.")

    # --- Pre-processing ---
    # Shift E1 to the origin for simplicity
    q1_orig = E1_orig.q
    E1 = E1_orig + (-q1_orig)
    E2 = E2_orig + (-q1_orig)

    # Ensure E1 is the non-degenerate ellipsoid
    if not E1.isFullDim():
        E1, E2 = E2, E1

    n = E1.dim
    nt = n
    x2_rem = np.array([])
    E2_calc = E2

    # Handle degeneracy in E2
    if not E2.isFullDim():
        nt = E2.rank()
        if nt == 0:
            # If E2 is a point, use the point-ellipsoid distance function
            return priv_distancePoint(E1, E2.q)

        T, _, _ = svd(E2.Q)

        # Manually transform E1 and E2 to bypass the matmul bug
        q1_trans = T.T @ E1.q
        Q1_trans = T.T @ E1.Q @ T
        E1 = Ellipsoid(Q1_trans, q1_trans)

        q2_trans = T.T @ E2.q
        Q2_trans = T.T @ E2.Q @ T
        E2_trans = Ellipsoid(Q2_trans, q2_trans)
        
        x2_rem = E2_trans.q[nt:]
        E2_calc = E2_trans.project(np.arange(nt))

    # --- QCQP Formulation using CVXPY ---
    x = cp.Variable(n)
    
    # Objective: Minimize distance from x to E2
    # ||sqrtm(inv(Q2)) * (x_proj - q2_proj)||^2 + ||(1/sqrt(TOL)) * (x_rem - q2_rem)||^2
    Q2_sqrt_inv = sqrtm(inv(E2_calc.Q))
    
    obj_p1 = cp.sum_squares(Q2_sqrt_inv @ (x[:nt] - E2_calc.q.flatten()))
    obj_p2 = cp.sum_squares((1/np.sqrt(E1.TOL)) * (x[nt:] - x2_rem.flatten())) if x2_rem.size > 0 else 0
    objective = cp.Minimize(obj_p1 + obj_p2)

    # Constraint: x must be in E1
    # ||sqrtm(inv(Q1)) * (x - q1)||^2 <= 1
    Q1_sqrt_inv = sqrtm(inv(E1.Q))
    constraints = [cp.sum_squares(Q1_sqrt_inv @ (x - E1.q.flatten())) <= 1]

    # Solve the problem
    prob = cp.Problem(objective, constraints)
    prob.solve()

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise RuntimeError(f"CVXPY could not solve the problem, status: {prob.status}")

    val = prob.value - 1
    return val

# This function will be needed by the final distance dispatcher, so we import it here
from .priv_distancePoint import priv_distancePoint 