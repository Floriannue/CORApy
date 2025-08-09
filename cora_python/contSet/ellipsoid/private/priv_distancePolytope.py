import numpy as np
from scipy.optimize import minimize
from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid
from cora_python.contSet.polytope.polytope import Polytope

def priv_distancePolytope(E: Ellipsoid, P: Polytope) -> float:
    """
    priv_distancePolytope - computes the distance from an ellipsoid to a
    polytope by solving a quadratic program.

    Inputs:
        E - ellipsoid object
        P - polytope object

    Outputs:
        val - distance between ellipsoid and polytope
    """
    
    # Handle degenerate ellipsoids approximately: if point, use point distance; if low rank, project to support direction
    if not E.isFullDim():
        if E.rank() == 0:
            # Re-use polytope distance method if E is just a point
            return P.distance(E.q)
        # Use center-to-polytope distance minus support radius in worst-case normal direction
        P.constraints()
        A = P.A
        b = P.b.reshape(-1,1)
        if A.size == 0:
            return 0.0
        # Choose most violated constraint normal
        violations = (A @ E.q - b).flatten()
        idx = int(np.argmax(violations))
        normal = A[idx:idx+1,:]
        normn = np.linalg.norm(normal)
        if normn == 0:
            return 0.0
        n_unit = normal / normn
        r_offset, _ = E.supportFunc_(n_unit.T, 'upper')
        r_center = float(n_unit @ E.q)
        r = r_offset - r_center
        d_center = float(n_unit @ E.q - b[idx]/normn)
        return d_center - r

    # Ensure polytope constraints are available and normalized
    P.constraints()
    A = P.A
    b = P.b

    # Fast path: P represents a single half-space (one inequality, no equalities)
    if A is not None and b is not None and A.shape[0] == 1 and (not hasattr(P, 'Ae') or P.Ae is None or P.Ae.size == 0):
        a = A.reshape(1, -1)
        a_norm = np.linalg.norm(a)
        if a_norm == 0:
            return 0.0
        a_n = a / a_norm
        plane_offset = float(b.reshape(-1, 1) / a_norm)
        center_proj = float(a_n @ E.q)
        rad = float(np.sqrt(a_n @ E.Q @ a_n.T)) if E.Q.size > 0 else 0.0
        # If outside: positive distance (signed), else <= 0
        return (center_proj - rad) - plane_offset
    
    # Normalize halfspace representation
    fac = 1. / np.sqrt(np.sum(A**2, axis=1))
    A_norm = A * fac[:, np.newaxis]
    b_norm = b.flatten() * fac

    # --- Solve QP using scipy.optimize.minimize ---
    # We want to find point x in P that minimizes (x-q)'Q^{-1}(x-q)
    
    Q_inv = np.linalg.inv(E.Q)
    q = E.q.flatten()

    # Objective function: 1/2 * x'Hx + f'x
    H = 2 * Q_inv
    f = -2 * Q_inv @ q

    def objective(x):
        return 0.5 * x.T @ H @ x + f.T @ x

    # Constraints: A_norm * x <= b_norm
    constraints = [{'type': 'ineq', 'fun': lambda x: (b_norm - A_norm @ x).flatten()}]

    # Initial guess (center of the ellipsoid)
    x0 = q
    
    # Solve the QP
    res = minimize(objective, x0, method='SLSQP', constraints=constraints)
    print(f"[DEBUG] priv_distancePolytope: res.x = {res.x}")
    print(f"[DEBUG] priv_distancePolytope: res.fun = {res.fun}")

    # Post-process result to match MATLAB output
    # objval_ is the raw objective value from the solver
    objval_ = res.fun
    # MATLAB's quadprog returns an objval that needs adjustment
    # The final distance is objval - 1
    # objval = objval_ + q' * Q_inv * q
    objval = objval_ + q.T @ Q_inv @ q
    
    val = objval - 1

    return val 