import numpy as np
from scipy.linalg import sqrtm
from numpy.linalg import svd
from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid

def priv_distancePoint(E: Ellipsoid, Y: np.ndarray) -> np.ndarray:
    """
    priv_distancePoint - computes the distance from an ellipsoid to an array
    of points

    Syntax:
        dist = priv_distancePoint(E,Y)

    Inputs:
        E - ellipsoid object
        Y - point or matrix of points (d x n)

    Outputs:
        dist - distance(s) between ellipsoid and points
    """

    dist_deg_dims = np.zeros(Y.shape[1])
    
    # Make copies to modify
    E_calc = E.copy()
    Y_calc = Y.copy()

    if not E_calc.isFullDim():
        nt = E_calc.rank()
        if nt == 0:
            # if only center remains, compute distance using ellipsoid equation
            return np.sum(((Y_calc - E_calc.q) / np.sqrt(E_calc.TOL))**2, axis=0) - 1

        T, _, _ = svd(E_calc.Q)
        
        E_trans = T.T @ E_calc
        Y_trans = T.T @ Y_calc
        
        Y_rem = Y_trans[nt:, :]
        x_rem = E_trans.q[nt:]
        
        # Re-assign E_calc and Y_calc for the final calculation step
        E_calc = E_trans.project(np.arange(nt))
        Y_calc = Y_trans[:nt, :]
        
        dist_deg_dims = np.sum(((Y_rem - x_rem[:, np.newaxis]) / np.sqrt(E_calc.TOL))**2, axis=0)

    # For full-dim part: (x-q)'*inv(Q)*(x-q)
    # which is ||sqrtm(inv(Q))*(x-q)||_2^2 = ||inv(sqrtm(Q))*(x-q)||_2^2
    # Using A\B = inv(A) @ B
    Q_sqrt = sqrtm(E_calc.Q)
    try:
        # Faster and more stable than inv
        temp = np.linalg.solve(Q_sqrt, Y_calc - E_calc.q)
    except np.linalg.LinAlgError:
        # Fallback for singular matrices
        temp = np.linalg.pinv(Q_sqrt) @ (Y_calc - E_calc.q)

    dist_nondeg_dims = np.sum(temp**2, axis=0)
    
    # Concatenate and subtract 1 (so that 0 means touching)
    dist = dist_nondeg_dims + dist_deg_dims - 1
    
    return dist 