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
    
    # The MATLAB code handles degenerate ellipsoids by transforming the space.
    # This currently fails in Python due to an unsolved bug in matrix multiplication.
    if not E.isFullDim():
        # Using E.rank() == 0 case from MATLAB as a temporary substitute
        if E.rank() == 0:
            # Re-use polytope distance method if E is just a point
            return P.distance(E.q)
        raise NotImplementedError("Distance to polytope for degenerate ellipsoids is not supported due to a persistent upstream bug.")

    # Ensure polytope constraints are available and normalized
    P.constraints()
    A = P.A
    b = P.b
    
    # Normalize halfspace representation
    fac = 1. / np.sqrt(np.sum(A**2, axis=1))
    A_norm = A * fac[:, np.newaxis]
    b_norm = b * fac

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
    constraints = [{'type': 'ineq', 'fun': lambda x: b_norm - A_norm @ x}]

    # Initial guess (center of the ellipsoid)
    x0 = q
    
    # Solve the QP
    res = minimize(objective, x0, method='SLSQP', constraints=constraints)

    # Post-process result to match MATLAB output
    # objval_ is the raw objective value from the solver
    objval_ = res.fun
    # MATLAB's quadprog returns an objval that needs adjustment
    # The final distance is objval - 1
    # objval = objval_ + q' * Q_inv * q
    objval = objval_ + q.T @ Q_inv @ q
    
    val = objval - 1

    return val 