"""
priv_zonotopeContainment_SadraddiniTedrakeDual - Solves the containment problem using the
   method described in [1, Theorem 3.5] (dual formulation of
   priv_zonotopeContainment_SadraddiniTedrake).

Syntax:
   res, cert, scaling = priv_zonotopeContainment_SadraddiniTedrakeDual(Z1, Z2, tol, scalingToggle)

Inputs:
   Z1 - zonotope object, inbody
   Z2 - zonotope object, circumbody
   tol - tolerance for the containment check
   scalingToggle - if set to True, scaling will be computed

Outputs:
   res - True/False
   cert - see logic below
   scaling - the smallest number 'scaling', such that scaling*(Z2 - center(Z2)) + center(Z2) contains Z1 (upper bound)

References:
   [1] A. Kulmburg, M. Althoff.: Approximability of the Containment Problem for Zonotopes and Ellipsotopes, 2024

Authors:       Adrian Kulmburg
Python port:   AI Assistant
"""
import numpy as np
from scipy.optimize import linprog

def priv_zonotopeContainment_SadraddiniTedrakeDual(Z1, Z2, tol, scalingToggle):
    # Extract generators and centers
    G_inbody = Z1.G if hasattr(Z1, 'G') else Z1.generators()
    G_circum = Z2.G if hasattr(Z2, 'G') else Z2.generators()
    c_inbody = Z1.c if hasattr(Z1, 'c') else Z1.center()
    c_circum = Z2.c if hasattr(Z2, 'c') else Z2.center()
    G_inbody = np.hstack([G_inbody, (c_inbody - c_circum).reshape(-1, 1)])
    m_inbody = G_inbody.shape[1]
    m_circum = G_circum.shape[1]
    n = G_inbody.shape[0]

    # Build dual constraints
    I_m = np.eye(m_inbody)
    ones_m = np.ones((m_inbody, 1))
    # Kronecker products
    pos_constraint = np.hstack([np.kron(I_m, G_circum.T), np.kron(ones_m, -np.eye(m_circum))])
    neg_constraint = np.hstack([np.kron(I_m, -G_circum.T), np.kron(ones_m, -np.eye(m_circum))])
    summation = np.hstack([np.zeros((1, n * m_inbody)), np.ones((1, m_circum))])
    A = np.vstack([pos_constraint, neg_constraint])
    b = np.zeros((2 * m_circum * m_inbody,))
    Aeq = summation
    beq = np.array([1.0])
    # MATLAB: cost = [-G_inbody(:);sparse(m_circum,1)];
    # MATLAB's G_inbody(:) flattens column-major, so use order='F' (Fortran order)
    cost = np.hstack([-G_inbody.flatten(order='F'), np.zeros(m_circum)])

    # Linear program
    bounds = [(None, None)] * (n * m_inbody + m_circum)
    try:
        result = linprog(cost, A_ub=A, b_ub=b, A_eq=Aeq, b_eq=beq, bounds=bounds, method='highs')
        if not result.success:
            scaling = np.inf
            res = False
            cert = True
            return res, cert, scaling
        X = result.x[:n * m_inbody]
        scaling = -result.fun
        # MATLAB: X = reshape(X(1:n*m_inbody), [n m_inbody]);
        # MATLAB reshape uses column-major, so use order='F' (Fortran order)
        X_mat = X.reshape((n, m_inbody), order='F')
        res = scaling <= 1 + tol
        if res:
            cert = True
        elif scaling > np.sqrt(m_inbody):
            cert = True
        elif np.all((X_mat == X_mat[:, [0]]) | (X_mat == -X_mat[:, [0]])):
            cert = True
        else:
            cert = False
        return res, cert, scaling
    except Exception:
        scaling = np.inf
        res = False
        cert = True
        return res, cert, scaling 